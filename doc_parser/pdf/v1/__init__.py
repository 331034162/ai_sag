"""
PDF 文档解析器 V1
================
"""

from ai_sag.doc_parser.pdf.v1.models import ImageInfo, PageContent, PDFResult
from ai_sag.doc_parser.pdf.v1.detector import PDFTypeDetector
from ai_sag.doc_parser.pdf.v1.image_processor import (
    WatermarkHandler,
    StampDetector,
    ImagePreprocessor,
    ImageProcessor,
)
from ai_sag.doc_parser.image.ocr import OCRBackend, BaseOCREngine, get_ocr_engine
from ai_sag.doc_parser.pdf.v1.parser import PDFParser, parse_pdf, parse_directory
from ai_sag.doc_parser.pdf.v1.formatter import (
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
