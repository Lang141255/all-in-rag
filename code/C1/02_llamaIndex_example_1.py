import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# load_dotenv()会把同目录（或上级路径）中的 .env 环境变量加载到进程（如DEEPSEEK_API_KEY）
load_dotenv()

# 设置全局默认 LLM 与 Embedding
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 读取单个 Markdown 文件，返回 [Document, ...]
# Document 含文本、metadata（如文件名、路径）等
docs = SimpleDirectoryReader(input_files=['../../data/C1/markdown/easy-rl-chatpter1.md']).load_data()

'''
核心步骤：
1. 把 Document 切块成 Node（内部会按默认分块器切分）
2. 用 Settings.embed_model 对每个 Node 生成向量
3. 存入默认内存向量存储（SimpleVectorStore）
得到的 index 就是“可检索”的向量索引
'''
index = VectorStoreIndex.from_documents(docs)

'''
把索引封装成问答引擎：
    收到查询 -> 对查询做 embedding -> 在向量库里找相似节点（top-k）
    把检索到的上下文与问题交给 Settings.llm 做综合回答
默认 top_k=2 或 3
'''
query_engine = index.as_query_engine()

# 打印内部用到的提示词模板（一个字典）
print(query_engine.get_prompts())
# 可以基于这些 key 定位并替换模板
print(query_engine.query('文中举了哪些例子?'))