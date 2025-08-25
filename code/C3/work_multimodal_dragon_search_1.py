import json
import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from dataclass import dataclass

@dataclass
class DragonImage:
    """龙类图像数据类"""
    img_id: str
    path: str
    title: str
    description: str
    category: str
    location: str
    environment: str
    combat_details: Dict[str, Any] = None
    secene_info: Dict[str, Any] = None

class DragonDataset:
    """龙类图像数据集管理类"""
    def __init__(self, data_dir: str, metadata_path: str):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.images: List[DragonImage] = []
        self._load_metadata()

    def _load_metadata(self):
        """加载图像元数据"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for img_data in data:
                # 确保图片路径是完整的
                if not img_data['path'].startswith(self.data_dir):
                    img_data['path'] = os.path.join(self.data_dir, img_data['path'].split('/')[-1])
                self.images.append(DragonImage(**img_data))

    def get_image_paths(self) -> List[str]:
        """获取所有图像路径"""
        return [img.path for img in self.images]
    
    def get_metadata_by_path(self, path: str) -> DragonImage:
        """根据路径获取元数据"""
        for img in self.images:
            if img.path == path:
                return img
            
        return None
    
    def get_text_content(self, img: DragonImage) -> str:
        """获取图像的文本描述内容"""
        parts = [
            img.title, img.description,
            img.location, img.environment
        ]

        if img.combat_details:
            parts.extend(img.combat_details.get('combat_style', []))
            parts.extend(img.combat_details.get('abilities_used', []))
        if img.secene_info:
            parts.append(img.secene_info.get('time_of_day', ''))

        return ' '.join(filter(None, parts))

class Encoder:
    """编码器类，用于将图像和文本编码为向量"""
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str = None, text: str = None) -> list[float]:
        """编码查询（支持图像+文本或仅文本）"""
        with torch.no_grad():
            if image_path and text:
                query_emb = self.model.encode(image=image_path, text=text)
            elif image_path:
                querb_emb = self.model.encode(image=image_path)
            elif text:
                query_emb = self.model.encode(text=text)
        
        return query_emb.tolist()[0]

    def encode_multimodal(self, image_path: str, text: str) -> list[float]:
        """编码多模态内容（图像+文本）"""
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]
    
def visualizer_results(query_image_path: str, retrieved_results: list, img_height: int = 300, img_width: int = 300, row_count: int = 3) -> np.ndarray:
    """从检索到的结果创建一个全景图用于可视化"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    if query_image_path and os.path.exists(query_image_path):
        query_pil = Image.open(query_image_path).convert("RGB")
        query_cv = np.array(query_pil)[:, :, ::-1]
        resized_query = cv2.resize(query_cv, (img_width, img_height))
        bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
        cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # 处理检索到的图像
    for i, result in enumerate(retrieved_results):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        img_path = result['image_path']
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row: start_row + img_height, start_col: start_col + img_width] = bordered_retrieved

        # 添加索引号和相似度
        cv2.putText(panoramic_image, f"{i+1}", (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(panoramic_image, f"{result['distance']:.3f}", (start_col + 10, start_row + img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return np.hstack([query_display_area, panoramic_image])

# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3/dragon"
METADATA_PATH = "multimodal_dragon_demo"
COLLECTION_NAME = "multimodal_dragon_demo"
MILVUS_URI = "http://localhost:19530"

# 2. 初始化数据集和编码器
print("--> 正在初始化数据集...")
dataset = DragonDataset(DATA_DIR, METADATA_PATH)
print(f"加载了 {len(dataset.images)} 张龙类图像")

print("--> 正在初始化编码器和Milvus客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 3. 创建 Milvus Collection
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

# 获取向量维度
sample_text = dataset.get_text_content(dataset.images[0])
sample_path = dataset.images[0].path
dim = len(encoder.encode_multimodal(sample_path, sample_text))

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name='img_id', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='image_path', dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name='category', dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name='location', dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name='environment', dtype=DataType.VARCHAR, max_length=64),
]

schema = CollectionSchema(fields, description="多模态龙类图像检索")

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")

# 4. 准备并插入数据
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
for img_data in tqdm(dataset.images, desc='生成多模态嵌入'):
    # 结合图像和文本信息生成向量
    text_content = dataset.get_text_content(img_data)
    vector = encoder.encode_multimodal(img_data.path, text_content)

    data_to_insert.append({
        "vector": vector,
        "img_id": img_data.img_id,
        "image_path": img_data.path,
        "title": img_data.title,
        "description": img_data.description,
        "category": img_data.category,
        "location": img_data.location,
        "environment": img_data.environment
    })

if data_to_insert:
    result = milvus_client.insert(collection_name = COLLECTION_NAME, data = data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")