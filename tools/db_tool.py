from tools import YA_MCPServer_Tool
from core.database.db import save_person_features, update_person_features, query_person_vectors
from core.yolo.yolo_core import get_image_and_bbox, extract_person_features, FEATURE_DIM

@YA_MCPServer_Tool(
    name="yolo_db_insert",
    title="YOLO Database insert",
    description="插入一条人物特征向量到数据库中。",
)
def db_insert(person_name: str, features: list) -> dict:
    """
    插入一条人物特征向量到数据库中。

    Args:
        person_name (str): 人名
        features (list): 人物特征向量列表

    Returns:
        dict: 包含操作结果的字典，例如 {"success": True} 或 {"error": "错误信息"}
    """

    try:
        # 校验特征向量维度
        if not isinstance(features, list):
            return {"error": "features must be a list of floats"}
        if len(features) != FEATURE_DIM:
            return {"error": f"feature length mismatch: expected {FEATURE_DIM}, got {len(features)}"}

        success = save_person_features(person_name, features)
        if success:
            return {"success": True}
        else:
            return {"error": "存储失败"}
    except Exception as e:
        return {"error": str(e)}



@YA_MCPServer_Tool(
    name="yolo_db_update",
    title="YOLO Database update",
    description="更新一条人物特征向量到数据库中。",
)
def db_update(person_name: str, features: list) -> dict:
    """
    更新一条人物特征向量到数据库中。

    Args:
        person_name (str): 人名
        features (list): 人物特征向量列表

    Returns:
        dict: 包含操作结果的字典，例如 {"success": True} 或 {"error": "错误信息"}
    """
    
    try:
        # 校验特征向量维度
        if not isinstance(features, list):
            return {"error": "features must be a list of floats"}
        if len(features) != FEATURE_DIM:
            return {"error": f"feature length mismatch: expected {FEATURE_DIM}, got {len(features)}"}

        success = update_person_features(person_name, features)
        if success:
            return {"success": True}
        else:
            return {"error": "更新失败"}
    except Exception as e:
        return {"error": str(e)}

@YA_MCPServer_Tool(
    name="image_person_insert",
    title="Image and Insert",
    description="从图片中提取人物特征并插入到数据库中。",
)
def image_and_insert(person_name: str, image_path: str) -> dict:
    """
    从图片中提取人物特征并插入到数据库中。

    Args:
        person_name (str): 人名
        image_path (str): 图片路径

    Returns:
        dict: 包含操作结果的字典，例如 {"success": True} 或 {"error": "错误信息"}
    """
    try:
        image, bbox = get_image_and_bbox(image_path)
        features = extract_person_features(image, bbox)
        if features is None:
            return {"error": "无法提取特征"}
        features_list = features.tolist()  # 将特征转换为列表
        success = save_person_features(person_name, features_list)
        if success:
            return {"success": True}
        else:
            return {"error": "存储失败"}
    except Exception as e:
        return {"error": str(e)}
    
@YA_MCPServer_Tool(
    name="person_query",
    title="person_query",
    description="从数据库中查询人物特征向量。",
)
def db_query(person_name: str) -> dict:
    """
    从数据库中查询人物特征向量。

    Args:
        person_name (str): 人名

    Returns:
        dict: 包含查询结果的字典
    """
    try:
        db_path = 'core/database/person_recognition.db'
        vector_data = query_person_vectors(person_name, db_path)
        if vector_data:
            features_list = vector_data[2]  # 获取特征列表
            return {"success": True, "features": features_list}
        else:
            return {"error": "未找到该人物特征"}
    except Exception as e:
        return {"error": str(e)}