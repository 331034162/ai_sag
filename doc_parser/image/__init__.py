"""
图像 OCR 解析子包
================
统一的图片 OCR 解析能力，支持：
- 文字识别（OCR）
- 水印 / 签章检测
- 表格识别（可选，由 ENABLE_TABLE_RECOGNITION 控制）

对外统一导出：
    from ai_sag.doc_parser.image import ImageParser, get_ocr_engine, ...
    from ai_sag.doc_parser.image import recognize_tables_in_image, TableRecognitionResult, ...
"""

from ai_sag.doc_parser.image.parser import (  # noqa: F401
    ImageParser,
    parse_image,
    parse_directory,
    ocr_for_pymupdf,
)
from ai_sag.doc_parser.image.ocr import get_ocr_engine  # noqa: F401
from ai_sag.doc_parser.image.processor import ImageProcessor  # noqa: F401
from ai_sag.doc_parser.image.models import (  # noqa: F401
    ImageOCRResult,
    TableCell,
    TableRecognitionResult,
)
from ai_sag.doc_parser.image.table.config import (  # noqa: F401
    ENABLE_TABLE_RECOGNITION,
    TABLE_RECOGNITION_BACKEND,
)
from ai_sag.doc_parser.image.table.recognizer import (  # noqa: F401
    PaddleTableRecognizer,
    VisualTableRecognizer,
    recognize_tables_in_image,
    get_table_recognizer,
)
from ai_sag.doc_parser.image.table.formatter import (  # noqa: F401
    table_to_markdown,
)
from ai_sag.doc_parser.image.table.layout import (  # noqa: F401
    sort_blocks_by_reading_order,
    reconstruct_structured_text,
)

# 结果格式化与输出
from ai_sag.doc_parser.image.formatter import (  # noqa: F401
    print_summary,
    print_ocr_text,
    save_text,
    save_summary,
)

__all__ = [
    "ImageParser", "parse_image", "parse_directory",
    "get_ocr_engine", "ImageProcessor",
    "ImageOCRResult",
    "TableCell", "TableRecognitionResult",
    "ENABLE_TABLE_RECOGNITION", "TABLE_RECOGNITION_BACKEND",
    "PaddleTableRecognizer", "VisualTableRecognizer",
    "recognize_tables_in_image", "get_table_recognizer",
    "table_to_markdown",
    "sort_blocks_by_reading_order", "reconstruct_structured_text",
    "print_summary", "print_ocr_text", "save_text", "save_summary",
]
