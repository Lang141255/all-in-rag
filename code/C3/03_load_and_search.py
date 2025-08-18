import argparse             # 命令行参数解析
from typing import List

from llama_index.core import (
    StorageContext,             # 告诉LlamaIndex去哪个目录找持久化的docstore、index 元数据、向量库等
    load_index_from_storage,    # 把该目录里的索引对象恢复出来
    Settings,                   # 全局配置入口，这里用来指定查询时要用的嵌入模型
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 把全局嵌入模型设为 BGE 中文小模型
def setup_embeddings(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """
    配置与构建索引时一致的嵌入模型
    一定要与建库时模型保持一致，否则相似度会失真
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name)


def load_index(persist_dir: str):
    """
    从持久化目录加载 VectorStoreIndex
    """ 
    storage_ctx = StorageContext.from_defaults(persist_dir=persist_dir) # 指向持久化目录
    index = load_index_from_storage(storage_ctx)    # 读取该目录下保存的各种存储
    return index

def similarity_search(index, query: str, top_k: int = 3):
    """
    使用 Retriever 做相似性检索，返回 NodeWithScore 列表
    """
    # index.as_retriever(...) 得到检索器 -> retrieve(query) 返回 List[NodeWithScore]
    '''
    NodeWithScore 里有：
        node: 被命中的文档片段（可 node.get_content()取文本，node.metadata 取元信息）
        score: 相似度分数，一般越大越相关
    '''
    retriever = index.as_retriever(similarity_top=top_k)
    nodes_with_scores = retriever.retrieve(query)
    return nodes_with_scores

def format_hits(hits) -> List[str]:
    """
    将检索结构转为可读字符串列表
    """
    formatted = []
    for i, nws in enumerate(hits, 1):
        node = nws.node
        score = getattr(nws, "score", None)
        # .get_content() 取出原始文本；也可用 node.text
        content = node.get_content().strip().replace("\n", " ")
        # 截断展示
        preview = content if len(content) <= 120 else content[:117] + "..."
        formatted.append(f"[{i}] score={score:.4f} text: {preview}")

    return formatted

def main():
    parser = argparse.ArgumentParser(
        description="Load a persisted LlamaIndex and run similarity search."
    )

    parser.add_argument(
        "query",       # qurey 是位置参数（必填）
        type=str,
        help="什么是LlamaIndex?"
    )

    parser.add_argument(
        "--persist",    # --persist 指明索引目录
        type=str,
        default="./llamaindex_index_store",
        help="索引持久化目录（与保存时一致）"
    )

    parser.add_argument(
        "--model",      # --model 指明嵌入模型（务必与建库一致）
        type=str,
        default="BAAI/bge-small-zh-v1.5",
        help="嵌入模型名称（需与建库时保持一致）"
    )

    parser.add_argument(
        "--topk",       # --topk 控制返回条数
        type=int,
        default=3,
        help="返回最相似的条目数量",
    )

    args = parser.parse_args()

    # 1) 配置嵌入(务必与建库时一致)
    setup_embeddings(args.model)

    # 2) 加载索引
    index = load_index(args.persist)

    # 3) 相似性检索
    hits = similarity_search(index, args.query, args.topk)

    print(f"\nQuery: {args.query}")
    if not hits:
        print("没有检索到结果")
        return 
    
    print(f"Top-{args.topk} 相似结果：")
    for line in format_hits(hits):
        print(line)

if __name__ == '__main__':
    main()