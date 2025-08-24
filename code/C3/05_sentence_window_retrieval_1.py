import os
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

'''
对照实验：
    同一份 PDF
    用两种切分与检索策略分别构建索引并回答同一个问题
1. 句子窗口检索 (Sentence-Window Retrieval)
    先把文档按句子切分，以“命中句子”为中心拼接前后各 window_size 句形成局部窗口（局部上下文）
    将“窗口文本”放进 metadata （键叫"window"），而节点本体 text 仍是那句“命中句子”
    检索时取 top-k 节点，再用 MetadataReplacementPostProcessor 把节点的 text 替换为 "window"，从而在生成答案时使用“命中句子 加减 上下文”的内容
2. 常规分块检索（作为基线）
    用 SenntenceSplitter 把文档按长度（默认近似按 token/字符数）切块（如512）再构建向量索引与检索
    命中的就是“常规大块”的文本

直观看到：句子窗口通常给出更贴题、干净的证据片段，而常规分块有时冗余上下文较多
'''

# 1. 配置模型
# 用 OpenAI 兼容的 OpenAILike 直连 DeepSeek （省去了 llama-index-llms-deepseek 的版本问题）
# temperature=0.1 让语言模型更稳（更少发散）
# 嵌入模型选 BGE small(EN)，速度快、够用
Settings.llm = OpenAILike(
    model = "deepseek-chat",
    temperature = 0.1,
    api_base = "https://api.deepseek.com/v1",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    is_chat_model = True
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 2. 加载文档
# 直接给定单个 PDF 文件路径
# 直接给定单个 PDF 文件路径，返回 documents（列表，每个文档包含 text 与元数据）
documents = SimpleDirectoryReader(
    input_files = ["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# 3. 创建节点与构建索引
# 3.1 句子窗口索引
# 先给句子切分；对每个“锚句”，把前后各 3 句拼接成窗口文本，存在 node.metadata["window"]
# original_text_metadata_key="original_text" 会把整段原文也存进 metadata（便于溯源）
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size = 3,
    window_metadata_key = "window",
    original_text_metadata_key = "original_text",
)

sentence_nodes = node_parser.get_nodes_from_documents(documents)
# 再把这些“句子节点”喂给 VectorStoreIndex，它会计算嵌入并建一个内存向量索引（未指定外部向量库时，默认内存实现）
sentence_index = VectorStoreIndex(sentence_nodes)

# 3.2 常规分块索引（基准）
# 按块长（约512）切分。是否“按 token”或“按字符”取决于幻觉中可用的 tokenizer (LlamaIndex 会尽量按 token 近似处理)
# 同样建立一个内存向量索引
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)

# 4. 构建查询引擎
# 两个引擎都取相似度 top-2
# 关键：句子窗口引擎套了一个 MetadataReplacementPostProcessor("window")，把检索到节点的 nodex.text 替换成 node.metadata["window"]，从而回答更贴近“命中句子 加减 上下文”的内容
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k = 2,
    node_postprocessors = [
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 5. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"查询：{query}\n")

print("--- 句子窗口检索结果 ---")
window_response = sentence_query_engine.query(query)
print(f"回答: {window_response}\n")

print("--- 常规检索结果 ---")
base_response = base_query_engine.query(query)
print(f'回答: {base_response}')