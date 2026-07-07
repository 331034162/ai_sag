"""
表格识别子包
===========
从图像中检测并提取表格结构化数据。

支持三种后端：
1. PaddleTableRecognizer — 基于 PaddleOCR PP-Structure
2. VisualTableRecognizer — 基于几何线条分析
3. PositionTableRecognizer — 基于 OCR 块位置聚类（推荐）

对外统一导出：
    from . import recognize_tables_in_image, TableRecognitionResult, ...
    from . import ENABLE_TABLE_RECOGNITION, TABLE_RECOGNITION_BACKEND, ...
"""

from .config import (  # noqa: F401
    ENABLE_TABLE_RECOGNITION,
    TABLE_RECOGNITION_BACKEND,
)
from .models import (  # noqa: F401
    TableCell,
    TableRecognitionResult,
)
from .recognizer import (  # noqa: F401
    TableRecognizer,
    PaddleTableRecognizer,
    VisualTableRecognizer,
    PositionTableRecognizer,
    get_table_recognizer,
    recognize_tables_in_image,
)
from .formatter import (  # noqa: F401
    table_to_markdown,
    tables_to_text,
)
from .layout import (  # noqa: F401
    sort_blocks_by_reading_order,
    reconstruct_structured_text,
    detect_tables_from_blocks,
    cluster_into_rows,
    detect_column_boundaries,
    detect_table_regions,
)

__all__ = [
    "ENABLE_TABLE_RECOGNITION", "TABLE_RECOGNITION_BACKEND",
    "TableCell", "TableRecognitionResult",
    "TableRecognizer",
    "PaddleTableRecognizer", "VisualTableRecognizer", "PositionTableRecognizer",
    "get_table_recognizer", "recognize_tables_in_image",
    "table_to_markdown", "tables_to_text",
    "sort_blocks_by_reading_order", "reconstruct_structured_text",
    "detect_tables_from_blocks",
    "cluster_into_rows", "detect_column_boundaries", "detect_table_regions",
]
