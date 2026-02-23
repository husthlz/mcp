from typing import Any, Dict
from pathlib import Path

from modules.yolo_models import get_classify_model, get_detect_model, get_segment_model
from tools import YA_MCPServer_Tool


# 定义保存目录
SAVE_DIR = Path("test/results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


@YA_MCPServer_Tool(
    name="yolo_tool", 
    title="YOLO Tool", 
    description="使用yolo进行检测、分割或分类"
)
async def yolo_tool(image_path: str, operation: str, conf: float = 0.5) -> Dict[str, Any]:
    """
    operation: str - 可选值为 'detect', 'segment', 'classify'
    """
    if operation == "detect":
        model = get_detect_model()
        result_filename = "test/detect_result.jpg"
    elif operation == "segment":
        model = get_segment_model()
        result_filename = "test/segment_result.jpg"
    elif operation == "classify":
        model = get_classify_model()
        result_filename = "test/classify_result.jpg"
    else:
        return {"error": "Invalid operation. Choose from 'detect', 'segment', or 'classify'."}

    paths = [image_path]
    results = model(paths, conf=conf)
    results.save_dir = SAVE_DIR 

    for result in results:
        result.save(filename=result_filename)

    return {
        "message": f"{operation.capitalize()} operation completed", 
        "results": [str(result) for result in results]
    }