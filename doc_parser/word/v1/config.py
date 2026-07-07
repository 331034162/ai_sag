"""
Word 文档解析器配置
==================
"""

import logging

# ============================================================
# 日志
# ============================================================
def setup_logging(level: int = logging.INFO):
    """配置日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    for _noise in ['faiss', 'paddle', 'ppocr', 'onnxruntime']:
        logging.getLogger(_noise).setLevel(logging.WARNING)


# ============================================================
# 图片 OCR
# ============================================================
# Word 中嵌入图片的最小面积阈值（像素²），低于此值跳过 OCR（避免小图标浪费算力）
IMAGE_MIN_AREA_FOR_OCR = 5000

# OCR 后端：rapidocr（速度快）或 paddleocr（精度高）
DEFAULT_OCR_BACKEND = "rapidocr"

# ============================================================
# 签章检测
# ============================================================
ENABLE_SIGNING_DETECTION: bool = True

SIGNING_KEYWORDS: list[str] = [
    '签字', '签章', '公章', '财务专用章', '法人名章',
    "单位公章", "盖章", "专用章",
]

logger = logging.getLogger(__name__)
