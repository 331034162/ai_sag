"""
Excel 工作表转 CSV 格式
========================

将每个 Sheet 转为标准 CSV 文本：
- 逗号分隔，双引号包裹含逗号/换行/双引号的字段
- 合并单元格：起点填值，非起点留空
- 值经过格式化处理（货币、百分比、日期等）
- 公式单元格输出计算值
"""

import csv
import io
from typing import Any, Optional, List, Dict, Tuple

from openpyxl.utils import get_column_letter

from .config import INCLUDE_HIDDEN, INCLUDE_EMPTY_CELLS


# ============================================================
# 隐藏行列检测
# ============================================================

def _get_visible_rows_cols(ws) -> Tuple[List[int], List[int]]:
    """获取可见（非隐藏）的行号和列号列表"""
    hidden_rows = set()
    for r, dim in ws.row_dimensions.items():
        if dim.hidden:
            hidden_rows.add(r)

    hidden_cols = set()
    for col_key, dim in ws.column_dimensions.items():
        if dim.hidden:
            if isinstance(col_key, int):
                hidden_cols.add(col_key)
            else:
                from openpyxl.utils import column_index_from_string
                hidden_cols.add(column_index_from_string(col_key))

    visible_rows = sorted(r for r in range(1, (ws.max_row or 0) + 1) if r not in hidden_rows)
    visible_cols = sorted(c for c in range(1, (ws.max_column or 0) + 1) if c not in hidden_cols)
    return visible_rows, visible_cols


# ============================================================
# 合并单元格信息采集
# ============================================================

def _collect_merged_ranges(ws) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    采集合并单元格信息。

    Returns:
        origin_map: {(row, col): (min_row, min_col)} 每个合并区域内单元格指向其起点
    """
    origin_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for merged_range in ws.merged_cells.ranges:
        min_r, max_r = merged_range.min_row, merged_range.max_row
        min_c, max_c = merged_range.min_col, merged_range.max_col

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if r == min_r and c == min_c:
                    continue
                origin_map[(r, c)] = (min_r, min_c)

    return origin_map


# ============================================================
# 单元格值格式化
# ============================================================

def _format_value(value: Any, number_format: str) -> str:
    """格式化单元格值，使其对 LLM 可读"""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()

    if isinstance(value, bool):
        return "是" if value else "否"

    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            return str(value)

    if isinstance(value, (int, float)):
        fmt = number_format or ""

        for symbol, prefix in [("¥", "¥"), ("￥", "¥"), ("$", "$"), ("€", "€")]:
            if symbol in fmt:
                return f"{prefix}{value:,.2f}"

        if "%" in fmt:
            pct = value * 100
            if pct == int(pct):
                return f"{int(pct)}%"
            return f"{pct:.2f}%"

        if "#,##0" in fmt or "#,#0" in fmt:
            if ".00" in fmt or ".0" in fmt:
                return f"{value:,.2f}"
            if isinstance(value, float) and value != int(value):
                return f"{value:,.2f}"
            return f"{int(value):,}"

        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            return str(round(value, 10))

        return str(value)

    return str(value)


# ============================================================
# 主处理函数
# ============================================================

def sheet_to_csv(ws, sheet_title: Optional[str] = None,
                 include_hidden: bool = False,
                 include_empty: bool = True) -> dict:
    """
    将单个 Sheet 转换为 CSV 文本。

    Args:
        ws: 工作表对象（data_only=True 加载）
        sheet_title: 工作表标题
        include_hidden: 是否包含隐藏行列
        include_empty: 是否包含空行

    Returns:
        dict: {sheet_name, csv_text, row_count, col_count}
    """
    title = sheet_title or ws.title

    if ws.max_row is None or ws.max_column is None or ws.max_row == 0:
        return {
            "sheet_name": title,
            "csv_text": "",
            "row_count": 0,
            "col_count": 0,
        }

    # 获取可见行列
    if include_hidden:
        visible_rows = list(range(1, ws.max_row + 1))
        visible_cols = list(range(1, ws.max_column + 1))
    else:
        visible_rows, visible_cols = _get_visible_rows_cols(ws)

    if not visible_rows or not visible_cols:
        return {
            "sheet_name": title,
            "csv_text": "",
            "row_count": 0,
            "col_count": len(visible_cols),
        }

    # 合并单元格信息
    origin_map = _collect_merged_ranges(ws)

    # 按行输出 CSV
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    row_count = 0

    for row_idx in visible_rows:
        row_values = []
        has_value = False

        for col_idx in visible_cols:
            # 合并区域内非起点 → 空字符串
            if (row_idx, col_idx) in origin_map:
                row_values.append("")
                continue

            cell = ws.cell(row=row_idx, column=col_idx)
            value = cell.value

            if value is None or (isinstance(value, str) and value.strip() == ""):
                row_values.append("")
                continue

            number_format = cell.number_format or "General"
            formatted = _format_value(value, number_format)
            row_values.append(formatted)
            has_value = True

        # 跳过全空行
        if not has_value and not include_empty:
            continue

        writer.writerow(row_values)
        if has_value:
            row_count += 1

    csv_text = output.getvalue().rstrip("\r\n")

    return {
        "sheet_name": title,
        "csv_text": csv_text,
        "row_count": row_count,
        "col_count": len(visible_cols),
    }
