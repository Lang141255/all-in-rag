from unstructured.partition.auto import partition

# PDF 文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
# partition函数使用自动文件类型检测，内部会根据文件类型路由到对应的专用函数（如PDF文件会调用partition_pdf）
# 如过需要更专业的PDF处理，可以直接使用from unstructured.partition.pdf import partition_pdf，它提供更多PDF特有的参数选项，如OCR语言设置、图像提取、表格结构推理等高级功能，同时性能更优
elements = partition(
    filename = pdf_path,        # 文档文件路径，支持本地文件路径
    content_type = "application/pdf"  # 可选参数，指定MIME类型，可绕过自动文件类型检测
)

# 打印解析结果
print(f"解析完成：{len(elements)} 个元素，{sum(len(str(e)) for e in elements)}")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f'元素类型：{dict(types)}')

# 显示所有元素
print("\n所有元素：")
for i, element in enumerate(elements, 1):
    print(f'Element {i} ({element.category}):')
    print(element)
    print("=" * 60)