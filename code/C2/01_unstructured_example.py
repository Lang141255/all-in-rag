from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from collections import Counter
from time import perf_counter

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# # 使用Unstructured加载并解析PDF文档
# elements = partition(
#     filename=pdf_path,
#     content_type="application/pdf"
# )

# # 打印解析结果
# print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

def summarize(elements, label):
    total_chars = sum(len(str(e)) for e in elements)
    types = Counter(e.category for e in elements)
    print(f"[{label}] 解析完成：{len(elements)} 个元素，{total_chars} 字符")
    print(f'[{label}] 元素类型：{dict(types)}')

    # 可按需展示若干元素
    for i, e in enumerate(elements[:10], 1):
        print(f"\n[{label}] Element {i} ({e.category}):\n{e}\n" + "=" * 60)

t0 = perf_counter()
elements_hi = partition_pdf(
    filename=pdf_path,
    strategy="hi_res"
)

t1 = perf_counter()
summarize(elements_hi, f"hi_res 用时 {t1 - t0:.2f}s")

t2 = perf_counter()
elements_ocr = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only",
)

t3 = perf_counter()
summarize(elements_ocr, f"ocr_only 用时 {t3 - t2:.2f}s")


# # 统计元素类型
# from collections import Counter
# types = Counter(e.category for e in elements)
# print(f"元素类型: {dict(types)}")

# # 显示所有元素
# print("\n所有元素:")
# for i, element in enumerate(elements, 1):
#     print(f"Element {i} ({element.category}):")
#     print(element)
#     print("=" * 60)