"""
Word 表格解析处理器
=================
从 python-docx 的 Table 对象中提取完整表格结构（含合并单元格），
复用与 Excel 解析相同的 采集→格式化→合并处理→结构检测→Markdown输出 流水线。

python-docx 中合并单元格的 XML 特征：
- 水平合并（gridSpan）: <w:tc> 的 <w:tcPr><w:gridSpan w:val="N"/>
- 垂直合并（vMerge）: <w:tcPr><w:vMerge w:val="restart"/>（起始）
                     <w:tcPr><w:vMerge/>（继续，val 属性缺失）

注意：python-docx 的 row.cells 对合并单元格会返回相同的 Cell 对象，
      必须通过 XML 解析才能获得精确的网格位置。
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from .config import ENABLE_SIGNING_DETECTION, SIGNING_KEYWORDS

logger = logging.getLogger(__name__)

# Word XML 命名空间
_NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
}


# ============================================================
# 数据结构（与 excel converter.py 保持一致）
# ============================================================

@dataclass
class MergeInfo:
    """一个合并区域的完整信息"""
    min_row: int
    max_row: int
    min_col: int
    max_col: int

    @property
    def row_span(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def col_span(self) -> int:
        return self.max_col - self.min_col + 1

    def contains(self, row: int, col: int) -> bool:
        return self.min_row <= row <= self.max_row and self.min_col <= col <= self.max_col

    def is_origin(self, row: int, col: int) -> bool:
        return row == self.min_row and col == self.min_col


@dataclass
class CellMeta:
    """单个单元格的完整信息"""
    raw_value: str = ""
    formatted: str = ""
    final: str = ""
    merge: Optional[MergeInfo] = None
    is_merge_origin: bool = False
    col_span: int = 1
    row_span: int = 1


# ============================================================
# XML 辅助函数
# ============================================================

def _get_grid_span(tc_element) -> int:
    """从 <w:tc> XML 元素中获取 gridSpan 值"""
    tc_pr = tc_element.find('w:tcPr', _NS)
    if tc_pr is not None:
        gs = tc_pr.find('w:gridSpan', _NS)
        if gs is not None:
            val = gs.get(f'{{{_NS["w"]}}}val')
            if val:
                try:
                    return int(val)
                except ValueError:
                    pass
    return 1


def _get_vmerge(tc_element) -> Optional[str]:
    """
    从 <w:tc> XML 元素中获取 vMerge 状态。

    Returns:
        "restart"  - 垂直合并的起始单元格
        "continue" - 垂直合并的延续单元格
        None       - 不参与垂直合并
    """
    tc_pr = tc_element.find('w:tcPr', _NS)
    if tc_pr is not None:
        vm = tc_pr.find('w:vMerge', _NS)
        if vm is not None:
            val = vm.get(f'{{{_NS["w"]}}}val')
            if val == 'restart':
                return 'restart'
            # val 为 None 或缺失 → 表示 continue
            return 'continue'
    return None


def _get_cell_text(cell) -> str:
    """提取单元格的纯文本（保留换行）"""
    paragraphs = cell.paragraphs
    parts = []
    for p in paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)
    return "<br>".join(parts) if parts else ""


# ============================================================
# 阶段1：信息采集 — 从 Word Table 构建网格
# ============================================================

def collect_table_grid(table) -> Tuple[
    int, int,
    Dict[Tuple[int, int], CellMeta],
    List[MergeInfo]
]:
    """
    解析 Word 表格的完整网格结构，包括合并单元格。

    通过逐行扫描 XML 中的 gridSpan 和 vMerge 属性，精确还原表格的逻辑网格，
    而非 python-docx 返回的可能有重复 Cell 对象的简化视图。

    Returns:
        (num_rows, num_cols, cell_map, merge_list)
    """
    rows = table.rows
    num_rows = len(rows)
    if num_rows == 0:
        return 0, 0, {}, []

    # 第一步：逐行扫描 XML，构建原始网格
    # grid[row][col] = (cell_object, grid_span, vmerge_state)
    raw_grid: List[List[Tuple]] = []
    max_cols = 0

    for row in rows:
        row_cells = []
        col_cursor = 0
        for tc in row._tr.findall('w:tc', _NS):
            # 跳过被上方 vMerge continue 占据的位置
            # （col_cursor 需要前进，但不放新 cell）
            gs = _get_grid_span(tc)
            vm = _get_vmerge(tc)
            # 获取该 tc 对应的 python-docx Cell 对象
            # row.cells 基于 tc 在 _tr 中的索引，这里直接用 xml 找
            from docx.table import _Cell
            cell_obj = _Cell(tc, table)
            row_cells.append((cell_obj, gs, vm, col_cursor))
            col_cursor += gs
        raw_grid.append(row_cells)
        if col_cursor > max_cols:
            max_cols = col_cursor

    num_cols = max_cols

    # 第二步：构建完整的 (row, col) → CellMeta 映射和合并信息
    cell_map: Dict[Tuple[int, int], CellMeta] = {}
    merge_list: List[MergeInfo] = []

    # 先扫描所有垂直合并，建立 vMerge 区域
    # vmerge_starts: {col: row} 当前活跃的 restart 位置
    vmerge_starts: Dict[int, Tuple[int, int]] = {}  # col -> (start_row, start_col)

    for row_idx in range(num_rows):
        col_idx = 0
        for cell_obj, gs, vm, start_col in raw_grid[row_idx]:
            col_idx = start_col

            # 先处理 vMerge 结束逻辑（在设置新 start 之前）
            # vm=None 时，该列之前活跃的 vMerge 在此行结束（不含此行）
            # vm='restart' 时，如果同列已有活跃的 vMerge，先关闭旧的再开新的
            if vm is None or vm == 'restart':
                for c in range(col_idx, min(col_idx + gs, num_cols)):
                    if c == col_idx and c in vmerge_starts:
                        start_r, start_c = vmerge_starts.pop(c)
                        if start_r < row_idx:
                            merge_list.append(MergeInfo(
                                min_row=start_r + 1, max_row=row_idx,
                                min_col=start_c + 1, max_col=start_c + gs,
                            ))

            if vm == 'restart':
                # 新的垂直合并起点
                vmerge_starts[col_idx] = (row_idx, col_idx)
            elif vm == 'continue':
                # 垂直合并延续：不属于新 cell，属于上方的合并区域
                pass

    # 扫描结束后，关闭所有仍然活跃的 vMerge
    for col, (start_r, start_c) in vmerge_starts.items():
        mi = MergeInfo(
            min_row=start_r + 1, max_row=num_rows,
            min_col=start_c + 1, max_col=start_c + 1,
        )
        merge_list.append(mi)

    # 处理水平合并（gridSpan > 1 且非 vMerge continue）
    for row_idx in range(num_rows):
        for cell_obj, gs, vm, start_col in raw_grid[row_idx]:
            if gs > 1 and vm != 'continue':
                # 检查是否已与某个 merge_list 中的区域合并
                # 水平合并：找已有的包含此 cell 的 merge，扩展其 col 范围
                found = False
                for mi in merge_list:
                    if mi.contains(row_idx + 1, start_col + 1) and mi.is_origin(row_idx + 1, start_col + 1):
                        # 扩展列范围
                        mi.max_col = max(mi.max_col, start_col + gs)
                        found = True
                        break
                if not found:
                    merge_list.append(MergeInfo(
                        min_row=row_idx + 1, max_row=row_idx + 1,
                        min_col=start_col + 1, max_col=start_col + gs,
                    ))

    # 构建 cell_merge_map
    cell_merge_map: Dict[Tuple[int, int], MergeInfo] = {}
    for mi in merge_list:
        for r in range(mi.min_row, mi.max_row + 1):
            for c in range(mi.min_col, mi.max_col + 1):
                cell_merge_map[(r, c)] = mi

    # 第三步：填充 CellMeta
    for row_idx in range(num_rows):
        for cell_obj, gs, vm, start_col in raw_grid[row_idx]:
            r = row_idx + 1  # 1-based
            c = start_col + 1

            if vm == 'continue':
                # 延续单元格：文本置空，由合并逻辑填充
                merge = cell_merge_map.get((r, c))
                is_origin = merge is not None and merge.is_origin(r, c)
                cell_map[(r, c)] = CellMeta(
                    raw_value="",
                    merge=merge,
                    is_merge_origin=is_origin,
                    col_span=merge.col_span if is_origin else 1,
                    row_span=merge.row_span if is_origin else 1,
                )
                continue

            text = _get_cell_text(cell_obj)
            merge = cell_merge_map.get((r, c))
            is_origin = merge is not None and merge.is_origin(r, c)

            # 水平合并时，填充所有被占列
            cell_map[(r, c)] = CellMeta(
                raw_value=text,
                merge=merge,
                is_merge_origin=is_origin,
                col_span=merge.col_span if is_origin else 1,
                row_span=merge.row_span if is_origin else 1,
            )

            # 水平合并的后续列：标记为合并的一部分
            for dc in range(1, gs):
                cc = c + dc
                if (r, cc) not in cell_map:
                    cell_map[(r, cc)] = CellMeta(
                        raw_value="",
                        merge=merge,
                        is_merge_origin=False,
                        col_span=1,
                        row_span=1,
                    )

    # 补充网格中未被填充的空位
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if (r, c) not in cell_map:
                merge = cell_merge_map.get((r, c))
                is_origin = merge is not None and merge.is_origin(r, c)
                cell_map[(r, c)] = CellMeta(
                    raw_value="",
                    merge=merge,
                    is_merge_origin=is_origin,
                    col_span=merge.col_span if is_origin else 1,
                    row_span=merge.row_span if is_origin else 1,
                )

    return num_rows, num_cols, cell_map, merge_list


# ============================================================
# 阶段2：格式化（简化版，Word 无 number_format）
# ============================================================

def format_all_cells(cell_map: Dict[Tuple[int, int], CellMeta]):
    """格式化所有单元格（Word 中主要是文本处理）"""
    for cm in cell_map.values():
        cm.formatted = (cm.raw_value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>").strip()


# ============================================================
# 阶段3：合并处理
# ============================================================

def apply_merge_logic(cell_map: Dict[Tuple[int, int], CellMeta], merge_list: List[MergeInfo]):
    """
    根据合并信息处理每个单元格的最终输出值。
    规则同 excel converter.py。
    """
    origin_values: Dict[Tuple[int, int], str] = {}
    for mi in merge_list:
        origin = cell_map.get((mi.min_row, mi.min_col))
        if origin:
            origin_values[(mi.min_row, mi.min_col)] = origin.formatted

    for (row, col), cm in cell_map.items():
        if cm.merge is None or cm.is_merge_origin:
            cm.final = cm.formatted
        else:
            mi = cm.merge
            origin_val = origin_values.get((mi.min_row, mi.min_col), "")
            if col == mi.min_col:
                # 同列不同行 → 行方向 → 填充值
                cm.final = origin_val
            else:
                # 同行不同列 → 列方向 → 留空
                cm.final = ""


# ============================================================
# 阶段4：构建行数据 + 结构检测
# ============================================================

def build_rows(cell_map: Dict[Tuple[int, int], CellMeta], num_rows: int, num_cols: int):
    """构建行数据列表"""
    rows = []
    for row_idx in range(1, num_rows + 1):
        row_data = []
        for col_idx in range(1, num_cols + 1):
            cm = cell_map.get((row_idx, col_idx))
            row_data.append(cm.final if cm else "")
        rows.append((row_idx, row_data))
    return rows


def is_empty_row(row_data: list) -> bool:
    return all(v.strip() == "" for v in row_data)


def _has_col_merges(row_idx: int, cell_map: Dict, num_cols: int) -> bool:
    for c in range(1, num_cols + 1):
        cm = cell_map.get((row_idx, c))
        if cm and cm.is_merge_origin and cm.col_span > 1:
            return True
    return False


def _is_title_row(row_idx: int, row_data: list, cell_map: Dict, num_cols: int) -> bool:
    non_empty = [v for v in row_data if v.strip()]
    if len(non_empty) != 1:
        return False
    for c in range(1, num_cols + 1):
        cm = cell_map.get((row_idx, c))
        if cm and cm.is_merge_origin and cm.col_span >= num_cols * 0.5:
            return True
    return False


def _is_form_row(row_idx: int, cell_map: Dict, num_cols: int) -> bool:
    for c in range(1, num_cols + 1):
        cm = cell_map.get((row_idx, c))
        if cm and cm.is_merge_origin and cm.col_span > 1:
            val = cm.formatted
            if "：" in val or ":" in val:
                return True
    return False


def _is_signing_row(row_data: list) -> bool:
    for v in row_data:
        v = v.strip()
        if not v:
            continue
        if any(kw in v for kw in SIGNING_KEYWORDS):
            return True
        if '审批' in v:
            if re.search(r'审批[：:]', v):
                return True
            if len(v) <= 10 and v.endswith('审批'):
                return True
    return False


def _is_group_header(row_idx: int, rows: list, cell_map: Dict, num_cols: int) -> bool:
    if not _has_col_merges(row_idx, cell_map, num_cols):
        return False

    # 检查该行在所有合并区域之外是否有非空内容
    # 如果合并列之外还有实质数据，说明这不是纯粹的分组行，而是数据行
    # （例如: R1 = [供应商资质信息(合并2列), 公司名称, 武汉示例科技有限公司]）
    non_merge_non_empty = 0
    for c in range(1, num_cols + 1):
        cm = cell_map.get((row_idx, c))
        if cm is None:
            continue
        # 检查此格是否属于某个合并区域
        in_merge_span = False
        if cm.is_merge_origin and cm.col_span > 1:
            in_merge_span = True
        elif cm.merge and not cm.is_merge_origin:
            in_merge_span = True
        if not in_merge_span and cm.formatted.strip():
            non_merge_non_empty += 1

    if non_merge_non_empty >= 2:
        # 合并区域之外还有 2+ 个非空格 → 不是纯分组行，而是数据行
        return False

    row_pos = None
    for i, (ri, _) in enumerate(rows):
        if ri == row_idx:
            row_pos = i
            break
    if row_pos is None:
        return False

    for j in range(row_pos + 1, min(row_pos + 5, len(rows))):
        _, next_data = rows[j]
        if is_empty_row(next_data):
            continue
        next_row_idx = rows[j][0]
        next_has_merges = _has_col_merges(next_row_idx, cell_map, num_cols)
        if next_has_merges:
            return True
        non_empty = sum(1 for v in next_data if v.strip())
        if non_empty >= max(num_cols * 0.3, 2):
            # 下一行非空格足够多，但需判断它是"列名"还是"数据值"
            # 如果下一行含有明显的数据值（数字、日期、百分比等），
            # 说明当前行是键值对的分类标签，而非 group header
            data_value_count = 0
            for v in next_data:
                v = v.strip()
                if not v:
                    continue
                if re.match(r"^[\d,.¥￥$€%]+$", v):
                    data_value_count += 1
                elif re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", v):
                    data_value_count += 1
            if data_value_count >= 1:
                return False
            return True
        break

    return False


def _extract_form_fields(row_idx: int, cell_map: Dict, num_cols: int) -> List[str]:
    merges = []
    for c in range(1, num_cols + 1):
        cm = cell_map.get((row_idx, c))
        if cm and cm.is_merge_origin and cm.col_span > 1:
            merges.append((c, c + cm.col_span - 1, cm.formatted.strip()))

    fields = []
    for i, (start, end, val) in enumerate(merges):
        if not val or ("：" not in val and ":" not in val):
            continue
        next_val = ""
        for j in range(i + 1, len(merges)):
            ns, ne, nv = merges[j]
            if ns == end + 1:
                next_val = nv.replace("<br>", "")
                break
        if not next_val:
            for c in range(end + 1, num_cols + 1):
                cm = cell_map.get((row_idx, c))
                if cm and cm.formatted.strip():
                    next_val = cm.formatted.strip().replace("<br>", "")
                    break
        if next_val:
            fields.append(f"{val}{next_val}")
        else:
            fields.append(val)
    return fields


def _extract_signing_info(row_data: list) -> List[str]:
    parts = []
    for v in row_data:
        v = v.strip().replace("<br>", "")
        if not v or v == "签章区域":
            continue
        parts.append(v)

    results = []
    i = 0
    while i < len(parts):
        v = parts[i]
        if any(kw in v for kw in ['签章', '公章', '财务专用章', '法人名章']):
            results.append(v)
            i += 1
            continue
        if '：' in v or ':' in v:
            if i + 1 < len(parts) and '：' not in parts[i + 1] and ':' not in parts[i + 1]:
                results.append(f"{v}{parts[i + 1]}")
                i += 2
                continue
        results.append(v)
        i += 1

    return results


def _flatten_header(group_rows: List[Tuple[int, list]], leaf_data: list,
                    cell_map: Dict, num_cols: int,
                    first_col_value: str = None) -> list:
    """将多行表头展平为单行"""
    flattened = list(leaf_data)

    if first_col_value is not None and flattened[0].strip() == "":
        flattened[0] = first_col_value

    if not group_rows:
        return flattened

    for col_idx in range(num_cols):
        col_num = col_idx + 1
        leaf_name = flattened[col_idx].strip() if col_idx < len(flattened) else ""

        parts = []
        if leaf_name:
            parts.append(leaf_name)

        for g_row_idx, _ in reversed(group_rows):
            group_name = ""
            for g in range(1, num_cols + 1):
                cm = cell_map.get((g_row_idx, g))
                if cm and cm.is_merge_origin and cm.col_span > 1:
                    if cm.merge and cm.merge.contains(g_row_idx, col_num):
                        group_name = cm.formatted.strip()
                        break
            if group_name and group_name != leaf_name and group_name not in parts:
                parts.insert(0, group_name)

        if parts:
            flattened[col_idx] = "/".join(parts)
        else:
            flattened[col_idx] = ""

    return flattened


# ============================================================
# 阶段5：Markdown 输出
# ============================================================

def _escape_markdown(text: str) -> str:
    text = text.replace("|", "\\|")
    text = text.replace("*", "\\*")
    text = text.replace("_", "\\_")
    return text


def _col_letter(col_idx: int) -> str:
    """1-based 列索引 → 列字母（A, B, ..., Z, AA, ...）"""
    result = ""
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _output_table(lines: list, header: list, data_rows: list, num_cols: int):
    """输出一个 Markdown 表格"""
    while len(header) < num_cols:
        header.append("")
    for row in data_rows:
        while len(row) < num_cols:
            row.append("")

    # 裁剪尾部全空列
    while num_cols > 1:
        ci = num_cols - 1
        if header[ci].strip() == "" and all(r[ci].strip() == "" for r in data_rows):
            header.pop()
            for r in data_rows:
                r.pop()
            num_cols -= 1
        else:
            break

    # 空表头用列字母填充
    for i, h in enumerate(header):
        if h.strip() == "":
            header[i] = f"列{_col_letter(i + 1)}"

    if not data_rows and not any(v.strip() for v in header):
        return

    lines.append("| " + " | ".join(_escape_markdown(h) for h in header) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    for row in data_rows:
        lines.append("| " + " | ".join(_escape_markdown(v) for v in row) + " |")
    lines.append("")


# ============================================================
# 主入口
# ============================================================

def table_to_markdown(table, table_index: int = 0) -> dict:
    """
    将 python-docx Table 对象转换为 Markdown。

    Args:
        table: python-docx Table 对象
        table_index: 表格在文档中的顺序（0-based）

    Returns:
        {
            "markdown": str,       # Markdown 文本
            "title": str,          # 检测到的标题
            "form_fields": list,   # 表单字段
            "signing_info": list,  # 签章信息
            "row_count": int,
            "col_count": int,
        }
    """
    lines = []

    num_rows, num_cols, cell_map, merge_list = collect_table_grid(table)

    if num_rows == 0 or num_cols == 0:
        return {
            "markdown": "*（空表格）*\n",
            "title": "", "form_fields": [], "signing_info": [],
            "row_count": 0, "col_count": 0,
        }

    # 阶段2：格式化
    format_all_cells(cell_map)

    # 阶段3：合并处理
    apply_merge_logic(cell_map, merge_list)

    # 阶段4：构建行数据
    rows = build_rows(cell_map, num_rows, num_cols)

    # 找首列的行合并值
    first_col_merge_value = None
    for mi in merge_list:
        if mi.min_col == 1 and mi.row_span > 1:
            origin = cell_map.get((mi.min_row, mi.min_col))
            if origin:
                first_col_merge_value = origin.formatted
                break

    # 结构检测与分段
    title_texts = []
    form_metadata = []
    signing_metadata = []

    sections = []
    cur_groups = []
    cur_leaf = None
    cur_data = []

    def flush():
        if cur_leaf is not None:
            sections.append({
                "groups": list(cur_groups),
                "leaf": cur_leaf,
                "data": list(cur_data),
            })

    # 检测表格是否有真正的表头行：
    # 以下情况判定为无表头（键值对表格，用列字母作通用表头）：
    # 1. 第一个非空行含有垂直合并单元格（row_span > 1）
    # 2. 第一个非空行是纯水平合并行（所有非空格都属于合并区域），
    #    且下一行含有明显数据值（数字、日期等），说明当前行只是分类标签
    has_header_row = True
    for pos, (row_idx, row_data) in enumerate(rows):
        if is_empty_row(row_data):
            continue
        # 条件1：检查第一个非空行是否有垂直合并
        for c in range(1, num_cols + 1):
            cm = cell_map.get((row_idx, c))
            if cm and cm.is_merge_origin and cm.row_span > 1:
                has_header_row = False
                break
        if not has_header_row:
            break

        # 条件2：第一个非空行是纯水平合并行（所有非空格都在合并区域内）
        all_non_empty_in_merge = True
        has_any_merge = False
        for c in range(1, num_cols + 1):
            cm = cell_map.get((row_idx, c))
            if cm is None:
                continue
            if cm.formatted.strip():
                if cm.is_merge_origin and cm.col_span > 1:
                    has_any_merge = True
                elif cm.merge and not cm.is_merge_origin:
                    has_any_merge = True
                else:
                    all_non_empty_in_merge = False
                    break
        if has_any_merge and all_non_empty_in_merge:
            # 所有非空格都在合并区域内，检查下一行是否有数据值
            for j in range(pos + 1, min(pos + 3, len(rows))):
                _, next_data = rows[j]
                if is_empty_row(next_data):
                    continue
                data_value_count = 0
                for v in next_data:
                    v = v.strip()
                    if not v:
                        continue
                    if re.match(r"^[\d,.¥￥$€%]+$", v):
                        data_value_count += 1
                    elif re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", v):
                        data_value_count += 1
                if data_value_count >= 1:
                    has_header_row = False
                break
        break

    if not has_header_row:
        # 无表头模式：所有非特殊行都是数据行，用列字母作表头
        data_positions = []
        for pos, (row_idx, row_data) in enumerate(rows):
            if is_empty_row(row_data):
                continue
            if _is_title_row(row_idx, row_data, cell_map, num_cols):
                for c in range(1, num_cols + 1):
                    cm = cell_map.get((row_idx, c))
                    if cm and cm.is_merge_origin and cm.col_span >= num_cols * 0.5:
                        title_texts.append(cm.formatted)
                        break
                continue
            if ENABLE_SIGNING_DETECTION and _is_signing_row(row_data):
                for s in _extract_signing_info(row_data):
                    if s and s not in signing_metadata:
                        signing_metadata.append(s)
                continue
            if _is_form_row(row_idx, cell_map, num_cols):
                for f in _extract_form_fields(row_idx, cell_map, num_cols):
                    form_metadata.append(f)
                continue
            data_positions.append(pos)

        # 用列字母作为通用表头
        header = [_col_letter(c + 1) for c in range(num_cols)]
        data_rows = [rows[p][1] for p in data_positions]
        actual_cols = max(len(header), max((len(r) for r in data_rows), default=0))

        for t in title_texts:
            lines.append(f"**{t.strip()}**")
            lines.append("")

        if form_metadata:
            lines.append("**表单信息：**")
            for fm in form_metadata:
                lines.append(f"- {fm}")
            lines.append("")

        _output_table(lines, header, data_rows, actual_cols)

        if signing_metadata:
            lines.append("**签章信息：**")
            for sm in signing_metadata:
                lines.append(f"- {sm}")
            lines.append("")

        md = "\n".join(lines)
        title = title_texts[0] if title_texts else ""
        return {
            "markdown": md,
            "title": title,
            "form_fields": form_metadata,
            "signing_info": signing_metadata,
            "row_count": num_rows,
            "col_count": num_cols,
        }

    for pos, (row_idx, row_data) in enumerate(rows):
        if is_empty_row(row_data):
            continue

        if _is_title_row(row_idx, row_data, cell_map, num_cols):
            flush()
            cur_groups, cur_leaf, cur_data = [], None, []
            for c in range(1, num_cols + 1):
                cm = cell_map.get((row_idx, c))
                if cm and cm.is_merge_origin and cm.col_span >= num_cols * 0.5:
                    title_texts.append(cm.formatted)
                    break
            else:
                title_texts.append(next((v for v in row_data if v.strip()), ""))
            continue

        if ENABLE_SIGNING_DETECTION and _is_signing_row(row_data):
            flush()
            cur_groups, cur_leaf, cur_data = [], None, []
            for s in _extract_signing_info(row_data):
                if s and s not in signing_metadata:
                    signing_metadata.append(s)
            continue

        if _is_form_row(row_idx, cell_map, num_cols):
            flush()
            cur_groups, cur_leaf, cur_data = [], None, []
            for f in _extract_form_fields(row_idx, cell_map, num_cols):
                form_metadata.append(f)
            continue

        if _is_group_header(row_idx, rows, cell_map, num_cols):
            if cur_leaf is not None and cur_data:
                flush()
                cur_groups = []
                cur_data = []
            cur_groups.append(pos)
            cur_leaf = None
            continue

        # 叶子表头 vs 数据行
        has_merges = _has_col_merges(row_idx, cell_map, num_cols)
        non_empty = sum(1 for v in row_data if v.strip())
        numeric_count = sum(
            1 for v in row_data
            if v.strip() and re.match(r"^[\d,.¥￥$€%]+$", v.strip())
        )

        if not has_merges and non_empty >= max(num_cols * 0.3, 2):
            if cur_leaf is None:
                if cur_groups:
                    cur_leaf = pos
                elif numeric_count > non_empty * 0.4:
                    cur_leaf = pos
                else:
                    cur_leaf = pos
            else:
                cur_data.append(pos)
        else:
            if cur_leaf is not None:
                cur_data.append(pos)
            elif non_empty >= 2:
                cur_data.append(pos)

    flush()

    # 阶段5：输出
    for t in title_texts:
        lines.append(f"**{t.strip()}**")
        lines.append("")

    if form_metadata:
        lines.append("**表单信息：**")
        for fm in form_metadata:
            lines.append(f"- {fm}")
        lines.append("")

    for sec in sections:
        group_row_indices = sec["groups"]
        leaf_pos = sec["leaf"]
        data_positions = sec["data"]

        group_rows = [(rows[p][0], rows[p][1]) for p in group_row_indices]
        leaf_data = rows[leaf_pos][1] if leaf_pos is not None else []
        data_rows = [rows[p][1] for p in data_positions]

        header = _flatten_header(
            group_rows, leaf_data, cell_map, num_cols,
            first_col_value=first_col_merge_value,
        )
        actual_cols = max(len(header), max((len(r) for r in data_rows), default=0))
        _output_table(lines, header, data_rows, actual_cols)

    # 无段落时（简单表格）
    if not sections:
        data_positions = []
        leaf_pos = None
        for pos, (row_idx, row_data) in enumerate(rows):
            if is_empty_row(row_data):
                continue
            rt = None
            if _is_title_row(row_idx, row_data, cell_map, num_cols):
                rt = "title"
            elif ENABLE_SIGNING_DETECTION and _is_signing_row(row_data):
                rt = "signing"
            elif _is_form_row(row_idx, cell_map, num_cols):
                rt = "form"

            if rt == "title":
                for c in range(1, num_cols + 1):
                    cm = cell_map.get((row_idx, c))
                    if cm and cm.is_merge_origin and cm.col_span >= num_cols * 0.5:
                        title_texts.append(cm.formatted)
                        break
                continue
            if rt == "form":
                for f in _extract_form_fields(row_idx, cell_map, num_cols):
                    form_metadata.append(f)
                continue
            if rt == "signing":
                for s in _extract_signing_info(row_data):
                    if s and s not in signing_metadata:
                        signing_metadata.append(s)
                continue

            if leaf_pos is None:
                leaf_pos = pos
            else:
                data_positions.append(pos)

        if leaf_pos is not None:
            leaf_data = rows[leaf_pos][1]
            data_rows = [rows[p][1] for p in data_positions]
            header = _flatten_header([], leaf_data, cell_map, num_cols,
                                     first_col_value=first_col_merge_value)
            actual_cols = max(len(header), max((len(r) for r in data_rows), default=0))
            _output_table(lines, header, data_rows, actual_cols)

    if signing_metadata:
        lines.append("**签章信息：**")
        for sm in signing_metadata:
            lines.append(f"- {sm}")
        lines.append("")

    if not any(line.startswith("|") for line in lines) and not title_texts:
        lines.append("*（空表格）*")
        lines.append("")

    md = "\n".join(lines)
    title = title_texts[0] if title_texts else ""

    return {
        "markdown": md,
        "title": title,
        "form_fields": form_metadata,
        "signing_info": signing_metadata,
        "row_count": num_rows,
        "col_count": num_cols,
    }
