from typing import Dict, Any, List
import logging
from tools import YA_MCPServer_Tool
from core.yolo.person_reid import count_person_appearances, _count_person_vector_appearances

# 配置日志
logger = logging.getLogger(__name__)

@YA_MCPServer_Tool(
    name="count_person_image_appearances",
    title="Count Person Appearances (ReID)",
    description="使用统计特定人物在图片集中出现的次数"
)
async def person_appearances(
    target_image: str,
    image_folder: str,
    conf_threshold: float = 0.8,
    reid_threshold: float = 0.7,
    max_images: int = 100,
    save_matched_crops: bool = False,
    output_folder: str = "out_put"
) -> Dict[str, Any]:
    """
    使用统计特定人物在图片集中出现的次数
    
    Args:
        target_image: 目标人物图片路径
        image_folder: 要搜索的图片集文件夹路径
        conf_threshold: YOLO检测置信度阈值 (默认: 0.7)
        reid_threshold: 重识别相似度阈值 (默认: 0.6)
        max_images: 最大处理图片数量，None表示处理所有 (默认: 100)
        save_matched_crops: 是否保存匹配到的人物裁剪图 (默认: False)
        output_folder: 输出文件夹路径，如果save_matched_crops为True则必需
    
    Returns:
        包含统计结果的字典
    """
    
    results = count_person_appearances(
        target_image=target_image,
        image_folder=image_folder,
        conf_threshold=conf_threshold,
        reid_threshold=reid_threshold,
        max_images=max_images,
        save_matched_crops=save_matched_crops,
        output_folder=output_folder
    )  
    
    return results


@YA_MCPServer_Tool(
    name="count_person_vector_appearances",
    title="Count Person Appearances by Vector",
    description="统计特定人物特征向量在图片集中出现的次数"
)
async def person_vector_appearances(
    target_vector:  List[float],
    image_folder: str,
    conf_threshold: float = 0.8,
    reid_threshold: float = 0.7,
    max_images: int = 100,
    save_matched_crops: bool = False,
    output_folder: str = "out_put"
) -> Dict[str, Any]:
    """
    使用特征向量统计特定人物在图片集中出现的次数

    Args:
        target_vector: 目标人物特征向量
        image_folder: 要搜索的图片集文件夹路径
        conf_threshold: YOLO检测置信度阈值 (默认: 0.7)
        reid_threshold: 重识别相似度阈值 (默认: 0.6)
        max_images: 最大处理图片数量 (默认: 100)
        save_matched_crops: 是否保存匹配到的人物裁剪图 (默认: False)
        output_folder: 输出文件夹路径，如果save_matched_crops为True则必需

    Returns:
        包含统计结果的字典
    """

    results = _count_person_vector_appearances(
        target_vector=target_vector,
        image_folder=image_folder,
        conf_threshold=conf_threshold,
        reid_threshold=reid_threshold,
        max_images=max_images,
        save_matched_crops=save_matched_crops,
        output_folder=output_folder
    )

    return results
