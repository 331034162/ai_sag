"""
结果输出函数
============
"""

import os
import json
import logging

from .models import ExcelCSV

logger = logging.getLogger(__name__)


def print_summary(result: ExcelCSV):
    """打印解析摘要"""
    print("=" * 60)
    print(f"文件: {result.file_name}")
    print(f"工作表数: {result.total_sheets}")

    for s in result.sheets:
        info = f"  - {s.sheet_name}: {s.row_count}行 × {s.col_count}列"
        info += f", {len(s.csv_text)} 字符"
        print(info)

    print("=" * 60)


def save_csv_text(result: ExcelCSV, output_path: str):
    """保存 CSV 文本到文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        f.write(result.to_csv_text())
    logger.info(f"CSV 已保存: {output_path}")
