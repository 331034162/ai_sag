"""
结果输出函数
============
"""

import os
import logging

from ai_sag.doc_parser.excel.v2.models import ExcelResult

logger = logging.getLogger(__name__)


def print_summary(result: ExcelResult):
    """打印解析摘要"""
    print("=" * 60)
    print(f"文件: {result.file_name}")
    print(f"工作表数: {result.total_sheets}")
    print(f"Markdown 长度: {len(result.markdown_text)} 字符")

    for sc in result.sheets:
        info = f"  - {sc.sheet_name}: {sc.row_count}行 × {sc.col_count}列"
        if sc.form_fields:
            info += f", {len(sc.form_fields)}个表单字段"
        if sc.signing_info:
            info += f", {len(sc.signing_info)}条签章信息"
        if sc.comments:
            info += f", {len(sc.comments)}条批注"
        if sc.formulas:
            info += f", {len(sc.formulas)}个公式"
        print(info)

    print("=" * 60)


def save_markdown(result: ExcelResult, output_path: str):
    """保存 Markdown 到文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.markdown_text)
    logger.info(f"Markdown 已保存: {output_path}")


def save_text(result: ExcelResult, output_path: str):
    """保存纯文本到文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.full_text)
    logger.info(f"纯文本已保存: {output_path}")
