"""
PDF 文档解析器 V2
================
新增：PyMuPDF 表格检测 + 合并单元格处理
"""

from .models import ImageInfo, PageContent, PDFResult, PDFTable
from .detector import PDFTypeDetector
from .image_processor import (
    WatermarkHandler,
    StampDetector,
    ImagePreprocessor,
    ImageProcessor,
)
from ...image.ocr import OCRBackend, BaseOCREngine, get_ocr_engine
from .parser import PDFParser, parse_pdf, parse_directory
from .table_handler import (
    table_to_markdown, build_table_metadata,
    extract_tables_from_struct_tree, is_tagged_pdf, StructTableResult,
)
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
    "PDFTable",
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
    "table_to_markdown",
    "build_table_metadata",
    "extract_tables_from_struct_tree",
    "is_tagged_pdf",
    "StructTableResult",
    "print_summary",
    "print_page_text",
    "save_text",
    "save_markdown",
]
