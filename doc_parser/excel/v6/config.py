"""
Excel V6 CSV 格式解析器配置
===========================
"""

import logging

# 隐藏行列过滤：True=跳过隐藏行列不输出，False=隐藏行列也输出
INCLUDE_HIDDEN: bool = False

# 空单元格：True=输出空单元格，False=跳过空单元格
INCLUDE_EMPTY_CELLS: bool = True

# 结构识别开关：True=识别标题行/表单行/签章行/分组表头行（用于保护这些行不被误填充）
# False=不做结构识别，所有行视为普通数据行
ENABLE_STRUCTURE_DETECTION: bool = True

# 纵向合并向下填充开关：True=对行方向合并（同列跨多行）向下填充起点值，保留分组上下文
# False=非起点单元格一律留空（原始行为）
# 注意：开启时若 ENABLE_STRUCTURE_DETECTION 也开启，则标题/表单/签章/分组表头行不做填充
ENABLE_MERGE_FILL_DOWN: bool = True

# 签章关键词（用于识别签章行，签章行的纵向合并不做向下填充）
SIGNING_KEYWORDS: list[str] = [
    '签字', '签章', '公章', '财务专用章', '法人名章',
    "单位公章", "盖章", "专用章",
]

# 日志
logger = logging.getLogger(__name__)