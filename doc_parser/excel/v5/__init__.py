"""
Excel V5 二维数组 JSON 解析子包
================================
V5 版本的 Excel 解析能力（二维数组 + 原始类型 + 公式值）：
- 数据模型: Section, SheetData, ExcelData
- 配置: ENABLE_SIGNING_DETECTION, INCLUDE_HIDDEN, SIGNING_KEYWORDS
- 核心解析: ExcelParser
- 结果输出: print_summary, save_json
- 便捷函数: parse_excel, parse_directory
"""

from ai_sag.doc_parser.excel.v5.models import Section, SheetData, ExcelData
from ai_sag.doc_parser.excel.v5.config import ENABLE_SIGNING_DETECTION, INCLUDE_HIDDEN, SIGNING_KEYWORDS
from ai_sag.doc_parser.excel.v5.parser import ExcelParser
from ai_sag.doc_parser.excel.v5.formatter import print_summary, save_json
from ai_sag.doc_parser.excel.v5.parser import parse_excel, parse_directory

__all__ = [
    "Section",
    "SheetData",
    "ExcelData",
    "ENABLE_SIGNING_DETECTION",
    "INCLUDE_HIDDEN",
    "SIGNING_KEYWORDS",
    "ExcelParser",
    "parse_excel",
    "parse_directory",
    "print_summary",
    "save_json",
]
