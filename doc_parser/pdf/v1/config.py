"""
全局配置
========
集中管理所有可配置常量，避免魔法数字散落各处

注意：图像/OCR 相关配置已迁移至 doc_parser.image.config，
此文件保留 PDF 专属配置，并从 image 包重新导出共享配置以保持向后兼容
"""

import logging

# ============================================================
# 从 image 包重新导出共享配置（向后兼容）
# ============================================================
from ...image.config import (  # noqa: F401
    OCR_CONFIDENCE_THRESHOLD,
    OCR_PAGE_DPI,
    OCR_PYMUPDF_DPI,
    OCR_FONT_NAME,
    OCR_FONT_MIN_SIZE,
    WATERMARK_CONTOUR_THRESHOLD,
    WATERMARK_BG_WHITE_RATIO,
    WATERMARK_BG_DARK_RATIO,
    WATERMARK_BG_CONTENT_RATIO,
    STAMP_MIN_AREA,
    STAMP_MAX_SIZE,
    STAMP_ELLIPSE_RATIO,
    STAMP_MIN_FIT_RATIO,
    STAMP_MIN_SOLIDITY,
    setup_logging,
)


# ============================================================
# PDF 专属配置
# ============================================================
IMAGE_ONLY_RATIO = 0.5                # 图片页占比超过此值判定为扫描件（image_only）
                                     # 低于此值为混合型（mixed）或纯文本型（text）
