"""
PDF 文档解析器 V1
================
"""

from .models import ImageInfo, PageContent, PDFResult
from .detector import PDFTypeDetector
from .image_processor import (
    WatermarkHandler,
    StampDetector,
    ImagePreprocessor,
    ImageProcessor,
)
from ...image.ocr import OCRBackend, BaseOCREngine, get_ocr_engine
from .parser import PDFParser, parse_pdf, parse_directory
from .formatter import (
    print_summary,
    print_page_text,
    save_markdown,
    save_text,
)

__all__ = [
    "ImageInfo",
    "PageContent",
    "PDFResult",
    "PDFTypeDetector",
    "WatermarkHandler",
    "StampDetector",
    "ImagePreprocessor",
    "ImageProcessor",
    "OCRBackend",
    "BaseOCREngine",
    "get_ocr_engine",
    "PDFParser",
    "parse_pdf",
    "parse_directory",
    "print_summary",
    "print_page_text",
    "save_text",
    "save_markdown",
]
