from typing import Dict, Any
import logging
from tools import YA_MCPServer_Tool
from core.yolo.person_reid import _count_person_vector_appearances

logger = logging.getLogger(__name__)

@YA_MCPServer_Tool(
    name="extract_person_features",
    title="Extract Person Features",
    description="从人物图片中提取特征向量"
)
def extract_features(image_path: str) -> dict:
    """
    从人物图片中提取特征向量

    Args:
        image_path: 人物图片路径

    Returns:
        人物特征向量，或在出错时返回错误信息
    """
    try:
        from core.yolo.yolo_core import extract_person_features, get_image_and_bbox
        image, bbox = get_image_and_bbox(image_path)
        features = extract_person_features(image, bbox)
        features_list = features.tolist()  # 将特征转换为列表
        return {"features": features_list} if features is not None else {"error": "无法提取特征"}
    except Exception as e:
        logger.error(f"提取人物特征时出错: {str(e)}")
        return {"error": str(e)}
    
    
    
@YA_MCPServer_Tool(
    name="count_person_database_appearances",
    title="Count Person Appearances in Database",
    description="统计数据库中特定人物在图片集出现的次数"
)
async def count_person_database_appearances(person_name: str, image_folder: str) -> Dict[str, Any]:
    """
    统计数据库中特定人物在图片集出现的次数

    Args:
        person_name: 人名
        image_folder: 图片集文件夹路径

    Returns:
        包含统计结果的字典
    """
    try:
        from core.database.db import query_person_vectors
        features = query_person_vectors(person_name)[2]  # 获取特征向量列表
        results = _count_person_vector_appearances(
            target_vector=features,
            image_folder=image_folder
        )
        return results
    except Exception as e:
        logger.error(f"统计数据库中特定人物出现次数时出错: {str(e)}")
        return {"error": str(e)}