"""
数据结构定义
============
"""

import json
from dataclasses import dataclass, field


@dataclass
class SheetCSV:
    """单个 Sheet 的 CSV 文本"""
    sheet_name: str
    csv_text: str = ""          # CSV 格式文本（逗号分隔，换行符分行）
    row_count: int = 0          # 数据行数
    col_count: int = 0          # 列数

    def to_dict(self) -> dict:
        return {
            "sheet_name": self.sheet_name,
            "csv_text": self.csv_text,
            "row_count": self.row_count,
            "col_count": self.col_count,
        }


@dataclass
class ExcelCSV:
    """Excel CSV 解析结果"""
    file_path: str
    file_name: str
    total_sheets: int
    sheets: list[SheetCSV] = field(default_factory=list)
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

    def to_csv_text(self) -> str:
        """将所有 sheet 的 CSV 文本拼接为一个完整文本，用 sheet 名称分隔"""
        parts = []
        for s in self.sheets:
            parts.append(f"=== {s.sheet_name} ===")
            parts.append(s.csv_text)
        return "\n".join(parts)
