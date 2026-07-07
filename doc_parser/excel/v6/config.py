"""
Excel V6 CSV 格式解析器配置
===========================
"""

import logging

# 隐藏行列过滤：True=跳过隐藏行列不输出，False=隐藏行列也输出
INCLUDE_HIDDEN: bool = False

# 空单元格：True=输出空单元格，False=跳过空单元格
INCLUDE_EMPTY_CELLS: bool = True

# 日志
logger = logging.getLogger(__name__)
