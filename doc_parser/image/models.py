"""
图像 OCR 数据模型
=================

- ImageOCRResult: OCR 解析结果（含可选的表格识别数据）
- TableCell / TableRecognitionResult: 表格相关（re-export 自 table 子包）
"""

from dataclasses import dataclass, field

from ai_sag.doc_parser.image.table.models import (  # noqa: E402
    TableCell,
    TableRecognitionResult,
)


@dataclass
class ImageOCRResult:
    """图片 OCR 解析结果"""
    file_path: str
    file_name: str = ""
    ocr_text: str = ""
    has_watermark: bool = False
    watermark_info: dict = field(default_factory=dict)
    has_stamp: bool = False
    stamp_info: dict = field(default_factory=dict)
    preprocessed: bool = False
    ocr_backend: str = ""
    metadata: dict = field(default_factory=dict)
    tables: list[TableRecognitionResult] = None  # 表格识别结果列表（可选）

    def __post_init__(self):
        if not self.file_name and self.file_path:
            import os
            self.file_name = os.path.basename(self.file_path)
        if self.tables is None:
            self.tables = []


__all__ = [
    "ImageOCRResult",
    "TableCell",
    "TableRecognitionResult",
]
