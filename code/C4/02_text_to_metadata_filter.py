import os
from langchain_deepseek import ChatDeepSeek 
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
try:
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    '''
    1. 预处理工作：遍历每个文档，手动提取需要的字段（如title, author, view_count, length），并构建一个干净、扁平化的新 metadata 字典
    2. 这个过程确保了后续的自查询检索器能够直接、可靠地访问这些字段
    3. 最后，将处理好的文档和元数据存入 Chroma 向量数据库中，为下一步的查询构建做好准备
    '''
    for doc in docs:
        original = doc.metadata
        
        # 提取基本元数据字段
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        doc.metadata = metadata
        bili.append(doc)
        
except Exception as e:
    print(f"加载BiliBili视频失败: {str(e)}")

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
'''
配置元数据字段（metadata_field_info）：这是与LLM沟通的蓝图。通过 AttributeInfo 为每个元数据字段定义名称、
类型和一份清晰的自然语言 description。LLM 将依赖这份描述来理解如何处理用户的查询
因此，一份准确、无歧义的描述很重要
'''
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数）",
        type="integer"
    )
]

# 4. 创建自查询检索器
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
    )

'''
自查询检索器（Self-Query Retriever）是 LangChain 中实现这一功能的核心组件。它的工作流程如下：
1. 定义元数据结构：首先，需要向LLM清晰地描述文档内容和每个元数据字段的含义及类型
2. 查询解析：当用户输入一个自然语言查询时，自查询检索器会调用LLM，将查询分解为两部分:
    查询字符串（Query String）:用于进行语义搜索的部分
    元数据过滤器（Metadata Filter）:从查询中提取出结构化的过滤条件
3. 执行查询: 检索器将解析出的查询字符串和元数据过滤器发送给向量数据库，执行一次同时包含语义搜索和元数据过滤的查询

from_llm 方法在底层执行了两个核心操作：
1. 加载查询构造器：利用传入的 llm、document_contents 和 metadata_field_info，创建一个专门的“查询构造链”
   这个链的核心职责是将用户的自然语言查询转换为一个通用的、结构化的查询对象
2. 获取内置翻译器：接着，检查使用的向量数据库（这里是 Chroma），并为其匹配一个内置的“翻译器”。
   这个翻译器负责将上一步生成的通用查询对象，翻译成 Chroma 数据库能够原生理解和执行的过滤语法


'''
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

# 5. 执行查询示例
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    '''
    执行内置翻译器（retriever.invoke）：最后，用自然语言发起调用。检索器内部会依次执行“构造”和“翻译”两个步骤，
    最终向 Chroma 发起一个同时包含语义搜索和精确元数据过滤的复合查询，从而返回最相关的结果
    '''
    results = retriever.invoke(query)
    if results:
        for doc in results:
            title = doc.metadata.get('title', '未知标题')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("="*50)
    else:
        print("未找到匹配的视频")
