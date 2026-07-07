"""
图像 OCR 配置（统一管理）
======================
集中管理图像处理、OCR、表格识别的所有可配置常量。
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
    # 抑制第三方库的噪音日志
    for _noise in ['faiss', 'paddle', 'ppocr', 'onnxruntime']:
        logging.getLogger(_noise).setLevel(logging.WARNING)


# ============================================================
# [基础] OCR 相关
# ============================================================

OCR_CONFIDENCE_THRESHOLD = 0.5      # OCR 识别结果置信度过滤阈值
OCR_PAGE_DPI = 600                   # 页面级 OCR 渲染 DPI
OCR_PYMUPDF_DPI = 600                # pymupdf4llm OCR 回调渲染 DPI


# ============================================================
# [基础] OCR 文字插入字体（OCR 回调专用）
# ============================================================

OCR_FONT_NAME = "ocr_font"
OCR_FONT_MIN_SIZE = 10


# ============================================================
# [基础] 水印检测
# ============================================================

WATERMARK_CONTOUR_THRESHOLD = 50
WATERMARK_BG_WHITE_RATIO = 0.85
WATERMARK_BG_DARK_RATIO = 0.01
WATERMARK_BG_CONTENT_RATIO = 0.01


# ============================================================
# [基础] 签章/印章检测
# ============================================================

STAMP_MIN_AREA = 200
STAMP_MAX_SIZE = 300
STAMP_ELLIPSE_RATIO = (0.7, 1.4)
STAMP_MIN_FIT_RATIO = 0.3
STAMP_MIN_SOLIDITY = 0.4



