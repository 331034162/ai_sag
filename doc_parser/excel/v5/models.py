"""
数据结构定义
============
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    """一个表格段落：表头 + 数据行（二维数组）"""
    headers: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"headers": self.headers, "rows": self.rows}


@dataclass
class SheetData:
    """单个 Sheet 的解析结果"""
    sheet_name: str
    title: str = ""
    sections: list[Section] = field(default_factory=list)
    form_fields: list[str] = field(default_factory=list)
    signing_info: list[str] = field(default_factory=list)
    formulas: list[dict] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0

    def to_dict(self) -> dict:
        d: dict = {
            "sheet_name": self.sheet_name,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
        }
        if self.form_fields:
            d["form_fields"] = self.form_fields
        if self.signing_info:
            d["signing_info"] = self.signing_info
        if self.formulas:
            d["formulas"] = self.formulas
        d["row_count"] = self.row_count
        d["col_count"] = self.col_count
        return d


@dataclass
class ExcelData:
    """Excel 解析结果"""
    file_path: str
    file_name: str
    total_sheets: int
    sheets: list[SheetData] = field(default_factory=list)
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
