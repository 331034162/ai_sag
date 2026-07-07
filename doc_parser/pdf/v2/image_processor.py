"""
图像预处理组件
==============
水印检测与去除、签章检测、OCR 预处理（锐化/去噪/增强）、图片保存

注意：实际实现已迁移至 doc_parser.image.processor，
此文件保留用于向后兼容，所有导出均从 image 包重新导出
"""

from ...image.processor import (  # noqa: F401
    WatermarkHandler,
    StampDetector,
    ImagePreprocessor,
    ImageProcessor,
)

__all__ = [
    "WatermarkHandler",
    "StampDetector",
    "ImagePreprocessor",
    "ImageProcessor",
]
