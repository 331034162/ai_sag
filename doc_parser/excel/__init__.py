"""
Excel 文档解析子包
=================
提供完整的 Excel 解析能力：
- 数据模型: SheetContent, ExcelResult
- 配置: ENABLE_SIGNING_DETECTION, SIGNING_KEYWORDS
- 核心解析: ExcelParser
- 结果输出: print_summary, save_text, save_markdown
- 便捷函数: parse_excel, parse_directory
"""

from ai_sag.doc_parser.excel.v2.models import SheetContent, ExcelResult
from ai_sag.doc_parser.excel.v2.config import ENABLE_SIGNING_DETECTION, INCLUDE_HIDDEN, SIGNING_KEYWORDS
from ai_sag.doc_parser.excel.v2.parser import ExcelParser
from ai_sag.doc_parser.excel.v2.formatter import print_summary, save_markdown, save_text
from ai_sag.doc_parser.excel.v2.parser import parse_excel, parse_directory

__all__ = [
    "SheetContent",
    "ExcelResult",
    "ENABLE_SIGNING_DETECTION",
    "INCLUDE_HIDDEN",
    "SIGNING_KEYWORDS",
    "ExcelParser",
    "parse_excel",
    "parse_directory",
    "print_summary",
    "save_text",
    "save_markdown",
]
