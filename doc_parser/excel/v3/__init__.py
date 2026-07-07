"""
Excel V3 JSON 解析子包
======================
V3 版本的 Excel 解析能力（JSON 输出）：
- 数据模型: SheetJSON, ExcelJSON, TableSection
- 配置: ENABLE_SIGNING_DETECTION, INCLUDE_HIDDEN, SIGNING_KEYWORDS
- 核心解析: ExcelParser
- 结果输出: print_summary, save_json
- 便捷函数: parse_excel, parse_directory
"""

from ai_sag.doc_parser.excel.v3.models import SheetJSON, ExcelJSON, TableSection
from ai_sag.doc_parser.excel.v3.config import ENABLE_SIGNING_DETECTION, INCLUDE_HIDDEN, SIGNING_KEYWORDS
from ai_sag.doc_parser.excel.v3.parser import ExcelParser
from ai_sag.doc_parser.excel.v3.formatter import print_summary, save_json
from ai_sag.doc_parser.excel.v3.parser import parse_excel, parse_directory

__all__ = [
    "SheetJSON",
    "ExcelJSON",
    "TableSection",
    "ENABLE_SIGNING_DETECTION",
    "INCLUDE_HIDDEN",
    "SIGNING_KEYWORDS",
    "ExcelParser",
    "parse_excel",
    "parse_directory",
    "print_summary",
    "save_json",
]
