import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch
from PIL import Image
import logging
from modules.yolo_models.loader import get_detect_model
from torchvision import transforms
import torchreid

logger = logging.getLogger(__name__)

# OSNet x1_0 默认输出特征维度
FEATURE_DIM = 512

def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """计算两个特征向量的余弦相似度"""
    # 检查维度一致性，提供更友好的错误信息
    f1 = np.asarray(feat1)
    f2 = np.asarray(feat2)
    if f1.ndim != 1 or f2.ndim != 1:
        raise ValueError(f"feature vectors must be 1-D arrays, got {f1.ndim}D and {f2.ndim}D")
    if f1.shape[0] != f2.shape[0]:
        raise ValueError(f"feature dimension mismatch: {f1.shape[0]} vs {f2.shape[0]}")
    return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-6))


def extract_person_features(image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    """
    从检测框中提取人物特征
    
    Args:
        image: BGR格式的图像
        bbox: 边界框 [x1, y1, x2, y2]
    
    Returns:
        人物特征向量，失败返回None
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # 确保坐标在图像范围内
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # 裁剪人物区域
        person_crop = image[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            logger.warning(f"裁剪区域为空: bbox={bbox}")
            return None
        
        # 转换为RGB并创建PIL图像
        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(person_crop_rgb)
        
        # 预处理
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("正在初始化ReID模型 (OSNet)...")
        # 初始化ReID模型
        reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        reid_model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reid_model = reid_model.to(device)
        logger.info(f"使用设备: {device}")
        
        person_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            features = reid_model(person_tensor)
        
        # 转换为numpy数组并归一化
        features = features.cpu().numpy().flatten()
        features = features / np.linalg.norm(features)
        
        return features
        
    except Exception as e:
        logger.error(f"提取人物特征时出错: {str(e)}")
        return None



def get_image_and_bbox(image_path: str) -> Tuple[np.ndarray, List[float]]:
    """
    从图片路径中加载图像并返回边界框

    Args:
        image_path (str): 图片路径

    Returns:
        Tuple[np.ndarray, List[float]]: BGR格式的图像和边界框 [x1, y1, x2, y2]
    """
    try:
        
        target_img = cv2.imread(image_path)
        if target_img is None:
            raise ValueError(f"无法加载图片: {image_path}")

        # 使用目标检测模型获取边界框
        model = get_detect_model()
        results = model(image_path)
        
        for result in results:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                 for box in result.boxes:
                      if int(box.cls[0]) == 0:
                            bbox = box.xyxy[0].tolist()
                            if float(box.conf[0]) > 0.5:
                                return target_img, bbox
                     
        # 提取第一个边界框
        if len(results) > 0:
            bbox = results[0]['bbox']
            return target_img, bbox
        else:
            raise ValueError("未检测到任何边界框")

    except Exception as e:
        logger.error(f"处理图片时出错: {str(e)}")
        raise
