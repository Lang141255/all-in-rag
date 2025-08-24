import os
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import IndexNode
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

'''
整体思路：实现一个两层的检索/问答路由：
1. 顶层（向量检索）
    先把每个 Excel 工作表（例如“年份_1994”）概括成一句摘要节点，用 VectorStoreIndex 建索引。查询进来后，顶层检索会从这些摘要里挑出最相关的年份
2. 底层（表格问答）
    根据顶层挑中的年份，把同名的 PandasQueryEngine （对应该年的 DataFrame）取出来，让 LLM 在该 DataFrame 上生成/执行 pandas 代码，直接回答可运算的问题
这就是 RecursiveRetriever 的用途：先找哪张表，再在那张表里算答案
'''
# 配置模型
Settings.llm = OpenAILike(
    model = "deepseek-chat",
    api_base = "https://api.deepseek.com/v1",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    is_chat_model = True,
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1. 加载数据并为每个工作表创建查询引擎和摘要节点
# 加载 Excel & 为每个 sheet 建表格问答引擎
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

df_query_engines = {}
all_nodes = []

# 每个工作表一个DataFrame，一个 PandasQueryEngine
# 重要：PandasQueryEngine 允许任意代码执行（它会 eval pandas 代码），只在可信环境使用
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # 为当前工作表（DataFrame）创建一个 PandasQueryEngine
    # 底层：每张表一个 Pandas 引擎
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)

    # 顶层：每张表一条“摘要节点”
    # 为当前工作表创建一个摘要节点（IndexNode）
    # IndexNode.index_id 设置为 sheet_name，这样顶层检索路由出的节点能和 query_engine_dict 的 key 对得上
    year = sheet_name.replace('年份_', '')
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)

    # 存储工作表名称到其查询引擎的映射
    # df_query_engines: 把每个表的 query_engine 存起来，键名必须和 index_id 一致，否则路由不到
    # 记录“sheet 名 -> 底层引擎”的映射
    df_query_engines[sheet_name] = query_engine

# 2. 创建顶层索引（只包含摘要节点）
# 对摘要文本做嵌入并建立一个小向量库
vector_index = VectorStoreIndex(all_nodes)

# 3. 创建递归检索器
# 3.1 创建顶层检索器，用于在摘要节点中检索
# as_retriever(): 查询时仅挑出最相关的一个年份摘要（即路由到一张表）
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# 3.2 创建递归检索器
'''
路由机制：
1. 顶层向量检索用摘要挑出最佳 IndexNode;
2. 取该节点的 index_id（也就是 sheet 名），在 query_engine_dict 里找到对应的 PandasQueryEngine;
3. 把原始问题交给这个底层引擎取运行（生成/执行 pandas 代码）
'''
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict ={"vector": vector_retriever},  # 给顶层检索器起个名字"vector"
    query_engine_dict = df_query_engines,
    verbose = True,
)

# 4. 创建查询引擎
# RetrieverQueryEngine.from_args(...) 把这套路由器包装成一个统一的 query() 接口方便调用
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# 5. 执行查询
# 顶层会把这个问题与每条摘要算相似度（由嵌入决定），基本会命中"年份_1994"
query = "1994年评分人数最少的电影是哪一部？"
print(f"查询: {query}")
response = query_engine.query(query)
print(f"回答: {response}")