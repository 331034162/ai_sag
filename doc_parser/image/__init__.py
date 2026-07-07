"""
图像 OCR 解析子包
================
统一的图片 OCR 解析能力，支持：
- 文字识别（OCR）
- 水印 / 签章检测
- 表格识别（可选，由 ENABLE_TABLE_RECOGNITION 控制）

对外统一导出：
    from . import ImageParser, get_ocr_engine, ...
    from . import recognize_tables_in_image, TableRecognitionResult, ...
"""

from .parser import (  # noqa: F401
    ImageParser,
    parse_image,
    parse_directory,
    ocr_for_pymupdf,
)
from .ocr import get_ocr_engine  # noqa: F401
from .processor import ImageProcessor  # noqa: F401
from .models import (  # noqa: F401
    ImageOCRResult,
    TableCell,
    TableRecognitionResult,
)
from .table.config import (  # noqa: F401
    ENABLE_TABLE_RECOGNITION,
    TABLE_RECOGNITION_BACKEND,
)
from .table.recognizer import (  # noqa: F401
    PaddleTableRecognizer,
    VisualTableRecognizer,
    recognize_tables_in_image,
    get_table_recognizer,
)
from .table.formatter import (  # noqa: F401
    table_to_markdown,
)
from .table.layout import (  # noqa: F401
    sort_blocks_by_reading_order,
    reconstruct_structured_text,
)

# 结果格式化与输出
from .formatter import (  # noqa: F401
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
