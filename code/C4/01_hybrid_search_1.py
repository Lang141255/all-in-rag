import json
import os
import numpy as np
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchReuqest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

'''
目标：把“龙”的图片/文本元数据写进 Milvus，给每条文档算 BGE-M3 的“密集向量 + 稀疏向量”，创建两套索引，然后分别做密集检索、稀疏检索，以及用 RRF（Reciprocal Rank Fusion），做混合检索融合
数据流：JSON -> 拼接文本 -> BGE-M3 生成 dense/sparse -> 插入到 Milvus -> （可选）过滤 category -> 搜索 -> RRF 融合 -> 输出
索引：
    稀疏 SPARSE_INVERTED_INDEX + IP
    密集 AUTOINDEX + IP
注意：脚本最后会把 Collection 释放并删除（只是演示），真实场景请注释掉
'''
# 1. 初始化设置
'''
连接本地/服务器端 Milvus
指定集合名、数据文件路径
connections.connect 用于底层连接；脚本里还创建了 MilvusClient（高层封装）
'''
COLLECTION_NAME = "dragon_hybrid_demo"
MILVUS_URI = "http://localhost:19530" # 服务器模式
DATA_PATH = "../../data/C4/metadata/dragon.json" # 相对路径
BATCH_SIZE = 50

# 2. 连接 Milvus 并初始化嵌入模型
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
'''
BGE-M3 会同时产出：
    dense：浮点数组（如1024维，依具体实现为准）
    sparse：稀疏词袋/词权重（CSR结构）
ef.dim["dense"] 用来决定 dense_vector 字段的维度，用于定义 schema
这一步是混合检索的基础：语义由密集向量捕获，关键词精确匹配由稀疏向量捕获
'''
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")

# 3. 创建 Collection
# 用 MilvusClient 做集合存在性与删除
# 如果同名集合存在，直接删除，保证后面是一个全新集合
milvus_client = MilvusClient(uri=MILVUS_URI)
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)

'''
定义了一个主键 + 多个属性字段 + 两个向量字段的 schema:
    文本/标签类（title, description, category, location, environment 等）
    向量字段：
        sparse_vector: SPARSE_FLOAT_VECTOR
        dense_vector: FLOAT_VECTOR (维度等于 BGE-M3 的 dense 维度)
'''
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

# 如果集合不存在，则创建它及索引
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")
    # 创建集合
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 创建成功。")

    # 4. 创建
    print("--> 正在为新集合创建索引...")
    # 对稀疏向量建倒排稀疏索引（SPARSE_INVERTED_INDEX）, 度量是IP
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    # 对稠密向量建 AUTOINDEX（Milvus 自适应索引），度量也是 IP
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

# 无论是否刚创建，重新拿一个 Collection 对象，并把集合加载进内存
# 这一步在你的脚本里是先 load 再插入
collection = Collection(COLLECTION_NAME)

# 5. 加载数据并插入
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

# 如果没有数据，就从你的 JSON 文件里读入 dataset（一个 list）
if collection.is_empty:
    print(f"--> Collection 为空，开始插入数据...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 组装要嵌入的“文本”，拼接 title/description/location/environment，放进 docs
    # 原始对象放进 metadata
    docs, metadata = [], []
    for item in dataset:
        parts = [
            item.get('title', ''),
            item.get('description', ''),
            item.get('location', ''),
            item.get('environment', ''),
        ]

        docs.append(' '.join(filter(None, parts)))
        metadata.append(item)

    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    print(f"--> 正在生成向量嵌入...")
    # 一次性给所有 docs 编码，得到稀疏与稠密两个“数组”（稀疏通常是 CSR 矩阵）
    embeddings = ef(docs)
    print("--> 向量生成完成。")

    print("--> 正在分批插入数据...")
    # 为每个字段准备批量数据
    img_ids = [doc["img_id"] for doc in metadata]
    paths = [doc["path"] for doc in metadata]
    titles = [doc["description"] for doc in metadata]
    descriptions = [doc["description"] for doc in metadata]
    categories = [doc["category"] for doc in metadata]
    locations = [doc["location"] for doc in metadata]
    enviroments = [doc["environment"] for doc in metadata]

    # 获取向量
    sparse_vectors = embeddings["sparse"]
    dense_vectors = embeddings["dense"]

    # 插入数据
    collection.insert([
        img_ids,
        paths,
        titles,
        descriptions,
        categories,
        locations,
        enviroments,
        sparse_vectors,
        dense_vectors
    ])

    # flush() 把新增数据落盘，使后续可见
    collection.flush()
    print(f"--> 数据插入完成，总数：{collection.num_entities}")
else:
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")

# 6. 执行搜索
search_query = "悬崖上的巨龙"
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
top_k = 5

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: {'search_filter'}")

query_embeddings = ef([search_query])
dense_vec = query_embeddings["dense"][0]
sparse_vec = query_embeddings["sparse"]._getrow[0]

# 打印向量信息
print("\n=== 向量信息 ===")
print(f"密集向量维度: {len(dense_vec)}")
print(f"密集向量前5个元素: {dense_vec[:5]}")
print(f"密集向量范数: {np.linalg.norm(dense_vec):.4f}")

print(f"\n稀疏向量维度: {sparse_vec.shape[1]}")
print(f"稀疏向量非零元素数量: {sparse_vec.nnz}")
print("稀疏向量前5个非零元素:")
for i in range(min(5, sparse_vec.nnz)):
    print(f"    -   索引: {sparse_vec.indices[i]}, 值: {sparse_vec.data[i]:.4f}")
density = (sparse_vec.nnz / sparse_vec.shape[1] * 100)
print(f"\n稀疏向量密度: {density:.8f}%")

# 定义搜索参数
search_params = {"metric_type": "IP", "params": {}}

# 先执行单独的搜索
'''
在 dense_vector 上用 IP 做 ANN 检索
用结构化过滤器保留三类 category
打印前 top_k 的标题、路径、描述开头
'''
print("\n--- [单独] 密集向量搜索结果 ---")
dense_results = collection.search(
    [dense_vec],
    anns_field = "dense_vector",
    param = search_params,
    limit = top_k, 
    expr = search_filter,
    output_fields = ["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(dense_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

'''
在 sparse_vector 上用 IP 和稀疏倒排索引做搜索
同样打印结果
'''
print("\n--- [单独] 稀疏向量搜索结果 ---")
sparse_results = collection.search(
    [sparse_vec],
    anns_field="sparse_vector",
    param = search_params,
    limit = top_k, 
    expr = search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

'''
构造两条 ANN 搜索请求（稀疏/稠密），交给 hybrid_search 在服务端并发检索后做 RRF 融合（参数 k=60）
输出最终融合后的前 top_k
'''
print("\n--- [混合] 稀疏+密集向量搜索结果 ---")
# 创建 RRF 融合器
rerank = RRFRanker(k=60)

# 创建搜索请求
dense_req = AnnSearchReuqest([dense_vec], "dense_vector", search_params, limit=top_k)
sparse_req = AnnSearchReuqest([sparse_vec], "sparse_vector", search_params, limit=top_k)

# 执行混合搜索
results = collection.hybrid_seach(
    [sparse_req, dense_req],
    rerank = rerank,
    limit = top_k,
    output_fields = ["title", "path", "description", "category", "location", "environment"]
)

# 打印最终结果
for i, hit in enumerate(results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# 7. 清理资源
# 释放内存并删除集合。这让你的脚本是“建->查->删”的一次性演示流程
milvus_client.release_collection(Collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")

'''
分析代码为什么在密集向量检索和稀疏向量检索中，排名第三的驯龙高手在混合检索中反而排在了第五？

简短说：RRF 融合看“名次的一致性”，不是看单路的分数高不高
“驯龙高手” 在密集/稀疏各自都不差，但只要其中一条通道相对更靠后，而“霸王龙/奶龙”在两条通道都“中上”，RRF 就会把“更均衡的”排在前面，所以出现了第3 -> 第5的下滑

为什么会出现这种“掉名次”
RRF 不看原始 distance，只看名次。单路第3但另一条调出更靠后名次，综合得分就被“双4名次”的对手超越
候选池太小：你在 AnnSearchRequest(..., limit=top_k) 里用的是 5。每路只拿前 5 做融合，名次一两位的抖动对最终名词影响就很大
k = 60 很“温和”：两路名次差一两名的分差非常接近，所以“均衡中上”的项目更容易占优
'''