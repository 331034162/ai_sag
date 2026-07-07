"""
Excel 解析器配置
================
"""

import logging

# 签章检测开关：True=将签章行从表格中剥离为元数据，False=签章行当作普通数据行
ENABLE_SIGNING_DETECTION: bool = True

# 签章关键词
SIGNING_KEYWORDS: list[str] = [
    '签字', '签章', '公章', '财务专用章', '法人名章',
    "单位公章", "盖章", "专用章",
]

# 日志
logger = logging.getLogger(__name__)
