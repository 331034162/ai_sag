"""
结果输出函数
============
"""

import os
import json
import logging

from ai_sag.doc_parser.excel.v4.models import ExcelRaw

logger = logging.getLogger(__name__)


def print_summary(result: ExcelRaw):
    """打印解析摘要"""
    print("=" * 60)
    print(f"文件: {result.file_name}")
    print(f"工作表数: {result.total_sheets}")

    for s in result.sheets:
        non_empty = sum(1 for v in s.cells.values() if v is not None)
        info = f"  - {s.sheet_name}: {s.max_row}行 × {s.max_col}列, {non_empty}个非空单元格"
        if s.merged_cells:
            info += f", {len(s.merged_cells)}个合并区域"
        print(info)

    print("=" * 60)


def save_json(result: ExcelRaw, output_path: str,
              indent: int = 2, ensure_ascii: bool = False):
    """
    保存 JSON 到文件。

    Args:
        result: 解析结果
        output_path: 输出文件路径
        indent: JSON 缩进空格数
        ensure_ascii: 是否转义非 ASCII 字符
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=indent, ensure_ascii=ensure_ascii)
    logger.info(f"JSON 已保存: {output_path}")
