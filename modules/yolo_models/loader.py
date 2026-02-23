# 模型文件下载与加载，权重保存在本模块 weights/ 目录下

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

# 本模块下的权重目录
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# 当前工具使用的模型文件名
MODEL_DETECT = "yolo26n.pt"
MODEL_SEGMENT = "yolo26n-seg.pt"
MODEL_CLASSIFY = "yolo26n-cls.pt"

# 下载使用的 release，与 ultralytics 保持一致
ASSETS_RELEASE = "v8.4.0"


def _ensure_model(filename: str) -> Path:
    """若权重不存在则从官方 assets 下载到 WEIGHTS_DIR，返回权重文件路径。"""
    path = WEIGHTS_DIR / filename
    attempt_download_asset(path, release=ASSETS_RELEASE)
    return path


def _get_model(filename: str) -> YOLO:
    """下载（如需）并加载模型，返回 YOLO 实例。"""
    path = _ensure_model(filename)
    return YOLO(str(path))


def get_detect_model() -> YOLO:
    """获取检测模型（yolo26n.pt）。"""
    return _get_model(MODEL_DETECT)


def get_segment_model() -> YOLO:
    """获取分割模型（yolo26n-seg.pt）。"""
    return _get_model(MODEL_SEGMENT)


def get_classify_model() -> YOLO:
    """获取分类模型（yolo26n-cls.pt）。"""
    return _get_model(MODEL_CLASSIFY)
