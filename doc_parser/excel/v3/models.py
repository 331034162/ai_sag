"""
数据结构定义
============
"""

import json
from dataclasses import dataclass, field, asdict


@dataclass
class TableSection:
    """一个表格段落"""
    headers: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SheetJSON:
    """单个 Sheet 的 JSON 解析结果"""
    sheet_name: str
    title: str = ""
    sections: list[TableSection] = field(default_factory=list)
    form_fields: list[str] = field(default_factory=list)
    signing_info: list[str] = field(default_factory=list)
    comments: list[dict[str, str]] = field(default_factory=list)
    formulas: list[dict[str, str]] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class ExcelJSON:
    """Excel JSON 解析结果"""
    file_path: str
    file_name: str
    total_sheets: int
    sheets: list[SheetJSON] = field(default_factory=list)
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
