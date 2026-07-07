"""
PDF 文档解析子包
==============
提供完整的 PDF 解析能力：
- 数据模型: ImageInfo, PageContent, PDFTable, PDFResult
- 类型检测: PDFTypeDetector
- 图像处理: WatermarkHandler, StampDetector, ImagePreprocessor, ImageProcessor
- OCR 引擎: PaddleOCR / RapidOCR (通过 ocr_backend 参数切换)
- 表格检测: PyMuPDF 表格检测 + 合并单元格处理（V2 新增）
- 核心解析: PDFParser
- 结果输出: print_summary, save_text, save_markdown
- 便捷函数: parse_pdf, parse_directory

默认使用 V2 版本（含表格检测），也可通过 doc_parser.pdf.v1 显式使用 V1 版本。
"""

from ai_sag.doc_parser.pdf.v2.models import ImageInfo, PageContent, PDFTable, PDFResult
from ai_sag.doc_parser.pdf.v2.detector import PDFTypeDetector
from ai_sag.doc_parser.pdf.v2.image_processor import WatermarkHandler, StampDetector, ImagePreprocessor, ImageProcessor
from ai_sag.doc_parser.image.ocr import OCRBackend, BaseOCREngine, get_ocr_engine
from ai_sag.doc_parser.pdf.v2.parser import PDFParser
from ai_sag.doc_parser.pdf.v2.formatter import (
    print_summary,
    print_page_text,
    save_markdown,
    save_text,
)
from ai_sag.doc_parser.pdf.v2.parser import parse_pdf, parse_directory

__all__ = [
    "ImageInfo",
    "PageContent",
    "PDFTable",
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
