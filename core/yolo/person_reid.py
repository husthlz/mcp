import os
import glob
import cv2
import numpy as np
from typing import Dict, Any
import logging
from modules.yolo_models import get_detect_model
from core.yolo.yolo_core import compute_similarity, extract_person_features, FEATURE_DIM

logger = logging.getLogger(__name__)
    
def count_person_appearances(
    target_image: str,
    image_folder: str,
    conf_threshold: float = 0.5,
    reid_threshold: float = 0.7,
    max_images: int = 100,
    save_matched_crops: bool = False,
    output_folder: str = "out_put"
) -> Dict[str, Any]:
    """
    核心功能：统计特定人物在图片集中出现的次数
    
    Args:
        target_image: 目标人物图片路径
        image_folder: 要搜索的图片集文件夹路径
        conf_threshold: YOLO检测置信度阈值 (默认: 0.5)
        reid_threshold: 重识别相似度阈值 (默认: 0.7)
        max_images: 最大处理图片数量，None表示处理所有 (默认: None)
        save_matched_crops: 是否保存匹配到的人物裁剪图 (默认: False)
        output_folder: 输出文件夹路径，如果save_matched_crops为True则必需

    Returns:
        包含统计结果的字典
    """
    if not os.path.exists(target_image):
        return {
            "success": False,
            "error": f"目标图片不存在: {target_image}",
            "suggestions": "请检查目标图片路径是否正确"
        }
    
    if not os.path.exists(image_folder):
        return {
            "success": False,
            "error": f"图片集文件夹不存在: {image_folder}",
            "suggestions": "请检查图片集文件夹路径是否正确"
        }
    
    if save_matched_crops and not output_folder:
        return {
            "success": False,
            "error": "当save_matched_crops为True时，必须提供output_folder",
            "suggestions": "请指定输出文件夹路径"
        }
    
    # 创建输出文件夹（如果需要）
    if save_matched_crops and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "matched_crops"), exist_ok=True)
    
    try:
        # 初始化模型
        logger.info("正在初始化YOLO检测模型...")
        yolo_model = get_detect_model()
        
        # 步骤1: 提取目标人物特征
        logger.info(f"正在处理目标图片: {target_image}")
        target_img = cv2.imread(target_image)
        
        if target_img is None:
            return {
                "success": False,
                "error": f"无法读取目标图片: {target_image}",
                "suggestions": "请确保图片格式正确且未损坏"
            }
        
        # 使用YOLO检测目标图片中的人物
        target_results = yolo_model(target_image)
        target_features_list = []
        target_bboxes = []
        
        for result in target_results:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # person class
                        bbox = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        
                        if confidence >= conf_threshold:
                            features = extract_person_features(target_img, bbox)
                            if features is not None:
                                target_features_list.append(features)
                                target_bboxes.append({
                                    "bbox": bbox,
                                    "confidence": confidence
                                })
        
        if not target_features_list:
            return {
                "success": False,
                "error": "未在目标图片中检测到人物",
                "suggestions": "请确保目标图片中包含清晰的人物，或降低conf_threshold"
            }
        
        logger.info(f"目标图片中检测到 {len(target_features_list)} 个人物")
        
        # 步骤2: 获取图片集中的所有图片
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        
        # 限制处理数量
        if max_images and max_images > 0:
            image_files = image_files[:max_images]
        
        logger.info(f"找到 {len(image_files)} 张图片待处理")
        
        # 步骤3: 遍历图片集进行匹配
        results = {
            "success": True,
            "target_image": os.path.basename(target_image),
            "target_persons": len(target_features_list),
            "total_images": len(image_files),
            "images_with_matches": 0,
            "total_appearances": 0,
            "matches": [],
            "statistics": {
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 1.0,
                "confidence_stats": {}
            }
        }
        
        all_similarities = []
        
        for idx, image_path in enumerate(image_files):
            if idx % 10 == 0:
                logger.info(f"处理进度: {idx}/{len(image_files)}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"无法读取图片: {image_path}")
                continue
            
            # YOLO检测
            detections = yolo_model(image_path)
            
            image_matches = []
            
            for result in detections:
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # person
                            bbox = box.xyxy[0].tolist()
                            confidence = float(box.conf[0])
                            
                            if confidence >= conf_threshold:
                                # 提取当前人物特征
                                person_features = extract_person_features(image, bbox)
                                
                                if person_features is not None:
                                    # 若person_features维度不匹配，跳过该检测
                                    if person_features.ndim != 1 or person_features.shape[0] != FEATURE_DIM:
                                        logger.warning(f"检测到的特征维度不匹配，跳过: {person_features.shape}")
                                        continue
                                    # 与所有目标人物特征比较
                                    best_similarity = 0.0
                                    best_target_idx = -1
                                    
                                    for target_idx, target_features in enumerate(target_features_list):
                                        similarity = compute_similarity(person_features, target_features)
                                        if similarity > best_similarity:
                                            best_similarity = similarity
                                            best_target_idx = target_idx
                                    
                                    if best_similarity > reid_threshold:
                                        match_info = {
                                            "bbox": bbox,
                                            "detection_confidence": confidence,
                                            "reid_similarity": best_similarity,
                                            "matched_target_idx": best_target_idx
                                        }
                                        image_matches.append(match_info)
                                        all_similarities.append(best_similarity)
                                        
            
            if image_matches:
                results["images_with_matches"] += 1
                # 确保每张图片只统计一次出现
                results["total_appearances"] += 1
                results["matches"].append({
                    "image": os.path.basename(image_path),
                    "match_count": len(image_matches),
                    "persons": image_matches
                })
        
        # 计算统计信息
        if all_similarities:
            results["statistics"]["avg_similarity"] = float(np.mean(all_similarities))
            results["statistics"]["max_similarity"] = float(np.max(all_similarities))
            results["statistics"]["min_similarity"] = float(np.min(all_similarities))
        
        # 添加处理信息
        results["processing_info"] = {
            "conf_threshold": conf_threshold,
            "reid_threshold": reid_threshold,
            "target_person_count": len(target_features_list),
            "processed_images": len(image_files)
        }
        
        # 添加使用建议
        # results["recommendations"] = generate_recommendations(results)
        
        logger.info(f"处理完成！共找到 {results['total_appearances']} 次匹配")
        
        return results
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "suggestions": "请检查日志获取详细错误信息"
        }
        
        
        
        
def _count_person_vector_appearances(
    target_vector:  list[float],
    image_folder: str,
    conf_threshold: float = 0.5,
    reid_threshold: float = 0.7,
    max_images: int = 100,
    save_matched_crops: bool = False,
    output_folder: str = "out_put"
) -> Dict[str, Any]:
    """
    输入特征向量统计特定人物在图片集中出现的次数
    
    Args:
        target_vector: 目标人物特征向量
        image_folder: 要搜索的图片集文件夹路径
        conf_threshold: YOLO检测置信度阈值 (默认: 0.5)
        reid_threshold: 重识别相似度阈值 (默认: 0.7)
        max_images: 最大处理图片数量，None表示处理所有 (默认: None)
        save_matched_crops: 是否保存匹配到的人物裁剪图 (默认: False)
        output_folder: 输出文件夹路径，如果save_matched_crops为True则必需

    Returns:
        包含统计结果的字典
    """
    if not os.path.exists(image_folder):
        return {
            "success": False,
            "error": f"图片集文件夹不存在: {image_folder}",
            "suggestions": "请检查图片集文件夹路径是否正确"
        }
    
    if save_matched_crops and not output_folder:
        return {
            "success": False,
            "error": "当save_matched_crops为True时，必须提供output_folder",
            "suggestions": "请指定输出文件夹路径"
        }
    
    # 创建输出文件夹（如果需要）
    if save_matched_crops and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "matched_crops"), exist_ok=True)
    
    try:
        # 初始化模型
        logger.info("正在初始化YOLO检测模型...")
        yolo_model = get_detect_model()
        

        # 转换为numpy数组并校验维度
        target_vector = np.array(target_vector)
        if target_vector.ndim != 1:
            return {"success": False, "error": f"目标向量必须为一维数组，收到 {target_vector.ndim} 维"}
        if target_vector.shape[0] != FEATURE_DIM:
            return {"success": False, "error": f"目标向量维度错误：期望 {FEATURE_DIM}，收到 {target_vector.shape[0]}"}

        target_features_list = [target_vector]
        
        # 步骤2: 获取图片集中的所有图片
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        
        # 限制处理数量
        if max_images and max_images > 0:
            image_files = image_files[:max_images]
        
        logger.info(f"找到 {len(image_files)} 张图片待处理")
        
        # 步骤3: 遍历图片集进行匹配
        results = {
            "success": True,
            "target_persons": len(target_features_list),
            "total_images": len(image_files),
            "images_with_matches": 0,
            "total_appearances": 0,
            "matches": [],
            "statistics": {
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 1.0,
                "confidence_stats": {}
            }
        }
        
        all_similarities = []
        
        for idx, image_path in enumerate(image_files):
            if idx % 10 == 0:
                logger.info(f"处理进度: {idx}/{len(image_files)}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"无法读取图片: {image_path}")
                continue
            
            # YOLO检测
            detections = yolo_model(image_path)
            
            image_matches = []
            
            for result in detections:
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # person
                            bbox = box.xyxy[0].tolist()
                            confidence = float(box.conf[0])
                            
                            if confidence >= conf_threshold:
                                # 提取当前人物特征
                                person_features = extract_person_features(image, bbox)
                                
                                if person_features is not None:
                                    # 若person_features维度不匹配，跳过该检测
                                    if person_features.ndim != 1 or person_features.shape[0] != FEATURE_DIM:
                                        logger.warning(f"检测到的特征维度不匹配，跳过: {person_features.shape}")
                                        continue
                                    # 与所有目标人物特征比较
                                    best_similarity = 0.0
                                    best_target_idx = -1
                                    
                                    for target_idx, target_features in enumerate(target_features_list):
                                        similarity = compute_similarity(person_features, target_features)
                                        if similarity > best_similarity:
                                            best_similarity = similarity
                                            best_target_idx = target_idx
                                    
                                    if best_similarity > reid_threshold:
                                        match_info = {
                                            "bbox": bbox,
                                            "detection_confidence": confidence,
                                            "reid_similarity": best_similarity,
                                            "matched_target_idx": best_target_idx
                                        }
                                        image_matches.append(match_info)
                                        all_similarities.append(best_similarity)
                                        
            
            if image_matches:
                results["images_with_matches"] += 1
                # 确保每张图片只统计一次出现
                results["total_appearances"] += 1
                results["matches"].append({
                    "image": os.path.basename(image_path),
                    "match_count": len(image_matches),
                    "persons": image_matches
                })
        
        # 计算统计信息
        if all_similarities:
            results["statistics"]["avg_similarity"] = float(np.mean(all_similarities))
            results["statistics"]["max_similarity"] = float(np.max(all_similarities))
            results["statistics"]["min_similarity"] = float(np.min(all_similarities))
        
        # 添加处理信息
        results["processing_info"] = {
            "conf_threshold": conf_threshold,
            "reid_threshold": reid_threshold,
            "target_person_count": len(target_features_list),
            "processed_images": len(image_files)
        }
        
        # 添加使用建议
        # results["recommendations"] = generate_recommendations(results)
        
        logger.info(f"处理完成！共找到 {results['total_appearances']} 次匹配")
        
        return results
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "suggestions": "请检查日志获取详细错误信息"
        }