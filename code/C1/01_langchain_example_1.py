# 进行基础配置，包括导入必要的库、加载环境变量以及下载嵌入模型
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_spltters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv()

# 定义Markdown文件的路径
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件，加载该文件作为知识源
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
# 当不指定参数初始化 RecursiveCharacterTextSplitter()时，其默认行为旨在最大程度保留文本的语义结构
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 数据准备完成后，接下来构建向量索引
'''
中文嵌入模型：使用 HuggingFaceEmbeddings 加载之前在初始化设置中下载的中文嵌入模型
配置模型在CPU上运行，并启用嵌入归一化
'''
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-zh-v1.5",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': True}
)

# 构建向量存储
'''
将分割后的文本块（texts）通过初始化好的嵌入模型转换为向量表示，
然后使用 InMemoryVectorStore 将这些向量及其对应的原始文本内容添加进去，从而在内存中构建出一个向量索引
这个过程完成后，便构建了一个可供查询的知识索引
'''
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
# 使用 ChatPromptTemplate.from_template 创建一个结构化的提示模板
# 此模板指导 LLM 根据提供的上下文（context）回答用户的问题（question），并明确指出在信息不足时应如何回应
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}
                                          
问题:{question}
                                          
回答："""
                                          )

# 配置大语言模型
# 初始化 ChatDeepSeek 客户端，配置所用模型（deepseek-chat）、生成答案的温度参数（temperature=0.7）、最大Token数（max_tokens=2048）以及API密钥（从环境变量加载）
llm = ChatDeepSeek(
    model = 'deepseek-chat',
    temperature = 0.7,
    max_tokens = 2048,
    api_key = os.getenv('DEEPSEEK_API_KEY')
)

# 用户查询
# 设置一个具体的用户问题字符串
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
# 使用向量存储的 similarity_search 方法，根据用户问题在索引中查找最相关的 k 个文本块
retrieved_docs = vectorstore.similarity_search(question, k=3)
# 准备上下：将检索到的多个文本块的页面内容（doc.page_content）合并成一个单一的字符串，并使用双换行符（"\n\n"）分隔各个块，形成最终的上下文信息（docs_content）供大语言模型参考
# "\n\n"双换行符通常代表段落的结束和新段落的开始，这种格式有助于LLM将每个块视为一个独立的上下文来源，从而更好地理解和利用这些信息来生成回答
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

'''
调用LLM生成答案并输出
将用户问题（question）和先前准备好的上下文（docs_content）格式化到提示模板中，
然后调用ChatDeepSeek的invoke方法获取生成的答案
'''
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)