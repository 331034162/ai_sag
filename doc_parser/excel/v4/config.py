"""
Excel V4 原始单元格 JSON 解析器配置
====================================
"""

import logging

# 隐藏行列过滤：True=跳过隐藏行列不输出，False=隐藏行列也输出
INCLUDE_HIDDEN: bool = False

# 空单元格：True=输出空单元格为 null，False=跳过空单元格（更紧凑）
INCLUDE_EMPTY_CELLS: bool = True

# 日志
logger = logging.getLogger(__name__)
