"""
Excel V5 二维数组 JSON 解析器配置
==================================
"""

import logging

# 签章检测开关
ENABLE_SIGNING_DETECTION: bool = True

# 隐藏行列过滤
INCLUDE_HIDDEN: bool = False

# 签章关键词
SIGNING_KEYWORDS: list[str] = [
    '签字', '签章', '公章', '财务专用章', '法人名章',
    "单位公章", "盖章", "专用章",
]

# 日志
logger = logging.getLogger(__name__)
