"""
数据结构定义
============
"""

import json
from dataclasses import dataclass, field


@dataclass
class SheetRaw:
    """单个 Sheet 的原始单元格数据"""
    sheet_name: str
    max_row: int = 0
    max_col: int = 0
    merged_cells: list[str] = field(default_factory=list)
    cells: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sheet_name": self.sheet_name,
            "max_row": self.max_row,
            "max_col": self.max_col,
            "merged_cells": self.merged_cells,
            "cells": self.cells,
        }


@dataclass
class ExcelRaw:
    """Excel 原始解析结果"""
    file_path: str
    file_name: str
    total_sheets: int
    sheets: list[SheetRaw] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "total_sheets": self.total_sheets,
            "sheets": [s.to_dict() for s in self.sheets],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)
