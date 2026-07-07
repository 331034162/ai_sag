"""
PDF 文档解析器 V2
================
新增：PyMuPDF 表格检测 + 合并单元格处理
"""

from ai_sag.doc_parser.pdf.v2.models import ImageInfo, PageContent, PDFResult, PDFTable
from ai_sag.doc_parser.pdf.v2.detector import PDFTypeDetector
from ai_sag.doc_parser.pdf.v2.image_processor import (
    WatermarkHandler,
    StampDetector,
    ImagePreprocessor,
    ImageProcessor,
)
from ai_sag.doc_parser.image.ocr import OCRBackend, BaseOCREngine, get_ocr_engine
from ai_sag.doc_parser.pdf.v2.parser import PDFParser, parse_pdf, parse_directory
from ai_sag.doc_parser.pdf.v2.table_handler import (
    table_to_markdown, build_table_metadata,
    extract_tables_from_struct_tree, is_tagged_pdf, StructTableResult,
)
from ai_sag.doc_parser.pdf.v2.formatter import (
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
