"""
数据结构定义
============
"""

from dataclasses import dataclass, field


@dataclass
class SheetContent:
    """单个 Sheet 的解析结果"""
    sheet_name: str
    title: str = ""               # 标题行文本
    markdown_text: str = ""        # Markdown 表格文本
    form_fields: list[str] = field(default_factory=list)   # 表单字段 (如 "供应商：武汉XX")
    signing_info: list[str] = field(default_factory=list)  # 签章信息 (如 "申请人签字：张三")
    row_count: int = 0            # 数据行数（不含表头）
    col_count: int = 0            # 列数


@dataclass
class ExcelResult:
    """Excel 解析结果"""
    file_path: str
    file_name: str
    total_sheets: int
    sheets: list[SheetContent] = field(default_factory=list)
    full_text: str = ""           # 所有 sheet 纯文本合并
    markdown_text: str = ""       # 完整 Markdown（含目录、分 sheet）
    metadata: dict = field(default_factory=dict)
