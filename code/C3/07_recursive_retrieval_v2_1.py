import os
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

'''
这段代码把 movie.xlsx 里每个工作表（如“年份_1994”）做成两类向量化文档：
    摘要文档：一条很短的“该表时某年电影信息”的说明，用来路由“应该查询哪张表”
    内容文档：把整张表转成文本（df.to_string()），用来回答细节问题
查询时走两步：
1. 在摘要索引里做向量检索，找到最相关的工作表名（比如“年份_1994”）
2. 在内容索引里，按元数据 sheet_name = 年份_1994 进行精确过滤+ 向量检索，把那一年的表文本喂给 LLM 生成最终答案
'''
# 配置模型
# 之后创建的索引/检索都会默认使用这套 LLM 与嵌入模型
Settings.llm = OpenAILike(
    model = "deepseek-chat",
    api_base = "https://api.deepseek.com/v1",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    is_chat_model = True,
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1. 加载和预处理数据
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

summary_docs = []
content_docs = []

print("开始加载和处理Excel文件...")
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # 数据清洗
    # 针对中文列“评分人数”做清洗：删除尾缀“人评价” -> 转数值 -> 缺失设0 -> 转 int
    if '评分人数' in df.columns:
        df['评分人数'] = df['评分人数'].astype(str).str.replace('人评价', '').str.strip()
        df['评分人数'] = pd.to_numeric(df['评分人数'], errors='coerce').fillna(0).astype(int)

    '''
    摘要文档：短文本 + 元数据 sheet_name。将被嵌入到“摘要索引”里，供路由使用
    内容文档：df.to_string() 把全表序列化成纯文本（包含列名与每列）。同样附上 sheet_name，之后用作过滤镜
    注意：把整表转成文本很直观，但大的表会很长，可能挤占 LLM 上下文
    '''
    # 创建摘要文档（用于路由）
    # 摘要：用于“路由到哪张表”
    year = sheet_name.replace('年份_', '')
    summary_text = f'这个表格包含了年份为 {year} 的电影信息，包括电影名称、导演、评分、评分人数等'
    summary_doc = Document(
        text = summary_text,
        metadata = {"sheet_name": sheet_name}
    )
    summary_docs.append(summary_doc)

    # 创建内容文档（用于最终问答）
    # 内容：整张表转为字符串，供最终回答使用
    content_text = df.to_string(index=False)
    content_doc = Document(
        text = content_text,
        metadata = {"sheet_name": sheet_name}
    )
    content_docs.append(content_doc)

print("数据加载和处理完成。 \n")

# 2. 构建向量索引
# 使用默认的内存 SimpleVectorStore，它支持元数据过滤

# 2.1 为摘要创建索引
# 摘要索引：非常小（每张表 1 条摘要）
summary_index = VectorStoreIndex(summary_docs)

# 2.2 为内容创建索引
# 内容索引：每张表 1 条大文本。默认四合院内存型 SimpleVectorStore，无需额外后端
# 索引阶段会调用 Settings.embed_model 对文档 text 做嵌入并存储，同时保存 metadata
content_index = VectorStoreIndex(content_docs)

print("摘要索引和内容索引构建完成。\n")

# 3. 定义两步式查询逻辑
def query_safe_recursive(query_str):
    print(f"--- 开始执行查询 ---")
    print(f"查询: {query_str}")

    # 第一步：路由 - 在摘要索引中找到最相关的表格
    print("\n 第一步：在摘要 索引中进行路由 ...")
    # VectorIndexRetriever(..., similarity_top_k=1)：只取最相关的 1 条摘要
    summary_retriever = VectorIndexRetriever(index=summary_index, similarity_top_k=1)
    retrieved_nodes = summary_retriever(query_str)

    if not retrieved_nodes:
        return "抱歉，未能找到相关的电影年份信息。"
    
    # 获取匹配到的工作表名称
    # 从命中节点的 metadata 里拿出 sheet_name（例如“年份_1994”），作为“目标表”
    matched_sheet_name = retrieved_nodes[0].node.metadata['sheet_name']
    print(f"路由结果：匹配到工作表 -> {matched_sheet_name}")

    # 第二步：检索 - 在内容索引中根据工作表名称过滤并检索具体内容
    print("\n第二步：在内容索引中检索具体信息...")
    # 第二步：在“内容索引”里，用元数据过滤锁定到该工作表，再做向量检索
    '''
    MetadataFilters + ExactMatchFilter：在第二阶段把候选文档严格限制为 sheet_name == 命中表名，确保 LLM 只看到目标年份的文本
    RetrieverQueryEngine 的工作流：
    1. 调用 content_retriever 取回匹配节点（这里 top_k = 1，通常就是整张表的那条文本）
    2. 把检索到的文本拼成上下文，附着上原始问题调用 Settings.llm 生成答案
    '''
    content_retriever = VectorIndexRetriever(
        index = content_index,
        similarity_top_k = 1, # 通常只返回最匹配的整个表格即可
        filters = MetadataFilters(
            filters = [ExactMatchFilter(key="sheet_name", value=matched_sheet_name)]
        )
    )

    # 创建查询引擎并执行查询
    # 构建查询引擎并执行：由引擎先检索文档片段，再调用 LLM 生成答案
    query_engine = RetrieverQueryEngine.from_args(content_retriever)
    response = query_engine.query(query_str)

    print("--- 查询执行结束 ---\n")
    return response

# 4. 执行查询
# 第一步路由基本会命中“年份_1994”的摘要
# 第二步在“年份_1994”的表文本上，LLM通过阅读/比对数字来生成“评分人数最少的电影”的答案
query = "1994年评分人数最少的电影时哪一部？"
response = query_safe_recursive(query)

print(f"最终回答: {response}")
