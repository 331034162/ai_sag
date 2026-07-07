"""
结果输出函数
============
"""

import os
import json
import logging

from ai_sag.doc_parser.excel.v3.models import ExcelJSON

logger = logging.getLogger(__name__)


def print_summary(result: ExcelJSON):
    """打印解析摘要"""
    print("=" * 60)
    print(f"文件: {result.file_name}")
    print(f"工作表数: {result.total_sheets}")

    for s in result.sheets:
        sections_info = f", {len(s.sections)}个表格段" if s.sections else ""
        info = f"  - {s.sheet_name}: {s.row_count}行 × {s.col_count}列{sections_info}"
        if s.form_fields:
            info += f", {len(s.form_fields)}个表单字段"
        if s.signing_info:
            info += f", {len(s.signing_info)}条签章信息"
        if s.comments:
            info += f", {len(s.comments)}条批注"
        if s.formulas:
            info += f", {len(s.formulas)}个公式"
        print(info)

    print("=" * 60)


def save_json(result: ExcelJSON, output_path: str,
              indent: int = 2, ensure_ascii: bool = False):
    """
    保存 JSON 到文件。

    Args:
        result: 解析结果
        output_path: 输出文件路径
        indent: JSON 缩进空格数，默认 2
        ensure_ascii: 是否转义非 ASCII 字符，默认 False（保留中文等原始字符）
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=indent, ensure_ascii=ensure_ascii)
    logger.info(f"JSON 已保存: {output_path}")
