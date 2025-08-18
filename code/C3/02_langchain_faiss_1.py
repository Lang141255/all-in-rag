# 下面的代码演示了使用 LangChain 和 FAISS 完成一个完整的“创建 -> 保存 -> 加载 -> 查询”流程
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
docs = [Document(page_content=t) for t in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 创建向量存储并保存到本地
'''
from_documents (封装层):
    这是我们直接调用的方法。它的职责很简单：从输入的 Document 对象列表中提取出纯文本内容(page_content)和元数据(metadata)
    然后，它将这些提取出的信息传递给核心的 from_texts 方法

from_texts (向量化入口):
    这个方法是面向用户的核心入口。它接收文本列表，并执行关键的第一步：调用 embedding.embed_documents(texts)，将所有文本批量转换为向量
    完成向量化，它并却直接处理索引构建，而是将生成的向量和其他所有信息(文本、元数据等)传递给一个内部的辅助方法_from

__from (构建索引框架)：
    这是一个内部方法，负责搭建 FAISS 向量存储的“空框架”
    它会根据指定的距离策略(默认为L2欧氏距离)初始化一个空的 FAISS 索引结构(如 faiss.IndexFlatL2)
    同时，它也准备好了用于存储文档原文的 docstore 和用于连接 FAISS 索引与文档的 index_to_docstore_id 映射
    最后，它调用另一个内部方法 _add 来完成数据的填充

__add (填充数据)：
    这是真正执行数据添加操作的核心。它接收到向量、文本和元数据后，执行以下关键操作：
        添加向量：将向量列表转换为 FAISS 需要的 numpy 数组，并调用 self.index.add(vector) 将其批量添加到 FAISS 索引中
        存储文档：将文本和元数据打包成 Document 对象，存入 docstore
        建立映射：更新 index_to_docsore_id 字典，建立起 FAISS 内部的整数ID(如 0, 1, 2...) 到我们文档唯一 ID 的映射关系
'''
vectorstore = FAISS.from_documents(docs, embeddings)

local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 加载索引并执行查询
# 加载时需指定相同的嵌入模型，并允许反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 执行相似性搜索
query = "FAISS是做什么的？"
results = loaded_vectorstore.similarity_search(query, k=1)

print("f\n查询: '{query}'")
print("相似度最高的文档：")
for doc in results:
    print(f"- {doc.page_content}")