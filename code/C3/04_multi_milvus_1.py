import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

'''
用 Visualized-BGE 把图像（以及图文组合查询）编码成向量 -> 存进 Milvus -> 建 HNSW 索引 -> 做一次“图+文本”的多模态检索 -> 把结果拼成一张大图展示 -> 最后清理集合

总体流程图：
1. 读取本地数据集 ../../data/C3/dragon/*.png
2. 用 Visualized_BGE 把每张图片编码为向量
3. 在 Milvus 里创建集合（id，向量字段， image_path）
4. 插入所有图片的向量与路径
5. 用 HNSW（COSINE） 为向量字段建索引并加载
6. 生成图+文本查询向量（query.png + “一条龙”）
7. 在 Milvus 搜索，取 top-5
8. 把“查询图 + 检索到的前几张图”拼接成一张可视化大图
9. 释放并删除集合（演示用，非持久化）
'''
# 1. 初始化设置
# 底座是 BGE 文本嵌入模型，权重 *.pth 是视觉分支 + 融合部分（Visualized-BGE）
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
# 要检索的图像所在目录；脚本默认看 dragon 子文件夹
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"
# Milvus 服务地址（HTTP）。确保 Milvus 已启动并允许 HTTP 端口访问
MILVUS_URI = "http://localhost:19530"

# 2. 定义工具（编码器和可视化函数）
class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""
    def __init__(self, model_name: str, model_path: str):
        # Visualized_BGE 会把图像或图+文本编码到统一语义空间（便于跨模态相似度搜索）
        self.model = Visualized_BGE(model_name=model_name, model_weight=model_path)
        # eval() 进入推理模式；配合 torch.no_grad() 避免梯度与显存开销
        self.model_eval()

    # 多模态查询：把“图像 + 文本”一起喂入，让查询向量更语义化
    # 返回 Python list（Milvus 期望原生数值数组）
    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)

        return query_emb.tolist()[0]

    # 库内向量：纯图像编码，用于入库
    # 注意：这里直接把文件路径传给 encode。这依赖于 Visualized_BGE 的实现是否支持“路径字符串”输入
    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)

        return query_emb.tolist()[0]

# 结果可视化（OpenCV + PIL 拼图）
def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int=300, row_count: int = 3) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化"""
    # 布局：左侧一列显示查询图（只在底部显示一张，方便对齐），右侧是 row_count × row_count 的网络放检索结果；每张结果图四周加黑边并在左上角标注序号
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    # RGB/BGR：PIL 读的是 RGB，这里用 [:, :, ::-1] 变成 BGR，符合 OpenCV 的通道惯例；最后 cv2.imwrite 正常输出颜色
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count -1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4), (img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row: start_row + img_height, start_col: start_col + img_width] = bordered_retrieved

        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 查询图放左侧“竖条”的最底部一格，红色边框，右侧是候选图的网格
    return np.hstack([query_display_area, panoramic_image])

# 3. 初始化客户端
# 首先初始化 Milvus 客户端，然后定义 Collection 的 Schema，它规定了集合的数据结构
print("--> 正在初始化编码器和Milvus客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 4. 创建 Milvus Collection
# 演示脚本每次重建集合，避免脏数据（生产不应频繁 drop）
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

image_list = glob(os.path.join(DATA_DIR, 'dragon', '*.png'))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
# 动态确定向量维度 dim （由模型返回的向量长度决定）
dim = len(encoder.encode_image(image_list[0]))

fields = [
    # id：主键，自增；
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # vector：浮点向量字段（Milvus 的向量索引）
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    # image_path：为可视化/回显准备的元数据
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

# 创建集合 Schema
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构：")
print(schema)

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构：")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

# 5. 准备并插入数据
# 创建好 Collectionn 后，需要将数据填充进去。通过遍历指定目录下的所有图片，将它们逐一编码成向量，然后与图片路径一起组织成符合 Schema 结构的格式，最后批量插入到 Collection 中
# 逐图编码 -> 组装成字典列表 -> 一次性插入
# insert_count 打印确认条数
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")

# 6. 创建索引
# 为了实现快速检索，需要为向量字段创建索引。这里选择 HNSW 索引，它在召回率和查询性能之间有着很好的平衡。
# 创建索引后，必须调用 load_collection 将集合加载到内存中才能进行搜索
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
index_params = milvus_client.prepare_index_params()
# HNSW：图索引，适合在线检索；M（每点连边数）与 efConstruction（建索引时的候选集大小）影响索引质量与速度
# 度量：COSINE。注意在 Milvus 中 distance 的定义与返回（是"距离=1-cos"还是"相似度=cos"）随接口版本可能不同
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情：")
print(milvus_client.desribe_index(collection_name=COLLECTION_NAME, index_name="vector"))
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")

# 7. 执行多模态检索
# 定义一个包含图片和文本的组合查询，将其编码为查询向量，然后调用 search 方法在 Milvus 中执行近似最近邻搜索
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
# 查询向量融合了图像与文字信息，相比纯图像/纯文本更“语义贴近”
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

# 检索参数：limit=5 返回前5
# 输出包含id、distance 及 entity（元数据）
search_results = milvus_client.search(
    collection_name = COLLECTION_NAME,
    data = [query_vector],
    output_fields = ["image_path"],
    limit = 5,
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
)[0]

retrieved_images = []
print("检索结果：")
for i, hit in enumerate(search_results):
    print(f"    Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])

# 8. 可视化与清理
# 最后，将检索到的图片路径用于可视化，生成一张直观的结果对比图。在完成所有操作后，应该释放 Milvus 中的资源，包括从内存中卸载 Collection 和删除整个 Collection
print(f"\n--> 正在可视化结果并清理资源")
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    # 保存并尝试用系统默认查看器打开
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到：{combined_image_path}")
    Image.open(combined_image_path).show()

# 演示完就释放并删除集合（生产中一般保留）
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内容村中释放 Collection：'{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection：'{COLLECTION_NAME}'")
