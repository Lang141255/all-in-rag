import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

# Visualized_BGE 是通过将图像token嵌入集成到BGE文本嵌入框架中构建的通用多模态嵌入模型，具备处理超越纯文本的多模态数据的灵活性
'''
模型参数：
    model_name_bge: 指定底层BGE文本嵌入模型，继承其强大的文本表示能力
    model_weight: Visual BGE的预训练权重文件，包含视觉编码器参数
多模态编码能力：Visual BGE提供了编码多模态数据的多样性，支持纯文本、纯图像或图文组合的格式：
    纯文本编码：保持原始BGE模型的强大文本嵌入能力
    纯图像编码：使用基于EVA-CLIP的视觉编码器处理图像
    图文联合编码：将图像和文本特征融合到统一的向量空间
'''
model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")

model.eval()

with torch.no_grad():
    text_emb = model.encode(text="蓝鲸")
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="蓝鲸")
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="蓝鲸")

# 计算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度：{multi_emb_1.shape}")
print(f"图像向量维度：{img_emb_1.shape}")
print(f"多模态向量示例(前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例(前10个元素):   {img_emb_1[0][:10]}")