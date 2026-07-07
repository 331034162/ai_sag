"""
Word 表格解析处理器 — 网格填充法
===============================
核心思路：
1. 解析 XML → 提取元数据（行列数、每个 cell 的位置/合并/文本、合并区域）
2. 构建完整网格 (rows × cols) → 按 XML 信息填充每个位置
   - vMerge continue 用 origin 值填充（Excel 风格，每行语义完整）
   - gridSpan 文本填在起始列，后续列留空（被合并占据）
3. 表头检测：
   - has_header_row 预判：检查第一个非空行是否有垂直合并或纯水平合并
   - 无表头模式：用列字母 (A, B, C...) 作通用表头
   - 有表头模式：逐行分类（标题/签章/表单/分组表头/叶子表头/数据行）
4. 展平多行表头为 "分组名/叶子名" 格式
5. 生成 Markdown

python-docx 中合并单元格的 XML 特征：
- 水平合并（gridSpan）: <w:tc> 的 <w:tcPr><w:gridSpan w:val="N"/>
- 垂直合并（vMerge）: <w:tcPr><w:vMerge w:val="restart"/>（起始）
                     <w:tcPr><w:vMerge/>（继续，val 属性缺失）

注意：python-docx 的 row.cells 对合并单元格会返回相同的 Cell 对象，
      必须通过 XML 解析才能获得精确的网格位置。
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from .config import ENABLE_SIGNING_DETECTION, SIGNING_KEYWORDS

logger = logging.getLogger(__name__)

# Word XML 命名空间
_NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CellInfo:
    """XML 中单个 <w:tc> 的解析结果"""
    row: int = 0          # 0-based 行号
    col: int = 0          # 0-based 列号（起始列）
    grid_span: int = 1    # 横向合并列数
    v_merge: Optional[str] = None  # None / 'restart' / 'continue'
    text: str = ""        # 单元格文本
    row_span: int = 1     # 纵向合并行数（仅 vMerge=restart 时计算）


@dataclass
class MergeRegion:
    """一个合并区域的描述"""
    type: str = ""          # 'horizontal' 或 'vertical'
    origin_row: int = 0     # 0-based
    origin_col: int = 0     # 0-based
    row_span: int = 1
    col_span: int = 1
    text: str = ""


@dataclass
class TableMetadata:
    """表格的完整元数据"""
    total_rows: int = 0
    total_cols: int = 0
    cells: List[CellInfo] = field(default_factory=list)
    merge_regions: List[MergeRegion] = field(default_factory=list)


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
            return 'continue'
    return None


def _get_cell_text_from_xml(tc_element) -> str:
    """从 <w:tc> XML 元素中提取纯文本（保留换行）"""
    parts = []
    for p in tc_element.findall('.//w:p', _NS):
        text_parts = []
        for r in p.findall('w:r', _NS):
            t = r.find('w:t', _NS)
            if t is not None and t.text:
                text_parts.append(t.text)
        t = ''.join(text_parts).strip()
        if t:
            parts.append(t)
    return "<br>".join(parts) if parts else ""


# ============================================================
# 阶段1：解析 XML → 元数据
# ============================================================

def parse_table_metadata(table) -> TableMetadata:
    """
    解析 Word 表格的 XML，提取完整的结构化元数据。

    Returns:
        TableMetadata 对象
    """
    tr_list = table._tbl.findall('w:tr', _NS)
    num_rows = len(tr_list)
    if num_rows == 0:
        return TableMetadata()

    # 从 tblGrid 获取总列数
    grid_cols = table._tbl.findall('.//w:tblGrid/w:gridCol', _NS)
    num_cols = len(grid_cols)

    # 逐行扫描 XML，收集每个 cell 的信息
    cells: List[CellInfo] = []
    for ri, tr in enumerate(tr_list):
        col_idx = 0
        # 追踪上方 vMerge 占据的列（这些列在当前行没有独立的 cell）
        # 用 (row, col) 集合跟踪
        occupied_by_vmerge = set()

        tc_list = tr.findall('w:tc', _NS)
        for tc in tc_list:
            gs = _get_grid_span(tc)
            vm = _get_vmerge(tc)
            text = _get_cell_text_from_xml(tc)

            cell = CellInfo(
                row=ri,
                col=col_idx,
                grid_span=gs,
                v_merge=vm,
                text=text,
            )
            cells.append(cell)

            # 标记 gridSpan 占据的后续列
            for offset in range(1, gs):
                occupied_by_vmerge.add(col_idx + offset)

            col_idx += gs

    # 计算 vMerge 区域的 rowSpan
    cell_map = {(c.row, c.col): c for c in cells}

    for c in cells:
        if c.v_merge == 'restart':
            row_end = c.row
            for rr in range(c.row + 1, num_rows):
                below = cell_map.get((rr, c.col))
                if below and below.v_merge == 'continue':
                    row_end = rr
                else:
                    break
            c.row_span = row_end - c.row + 1

    # 构建合并区域列表
    merge_regions: List[MergeRegion] = []

    # 纵向合并
    for c in cells:
        if c.v_merge == 'restart' and c.row_span > 1:
            merge_regions.append(MergeRegion(
                type='vertical',
                origin_row=c.row,
                origin_col=c.col,
                row_span=c.row_span,
                col_span=1,
                text=c.text,
            ))

    # 横向合并
    for c in cells:
        if c.grid_span > 1:
            merge_regions.append(MergeRegion(
                type='horizontal',
                origin_row=c.row,
                origin_col=c.col,
                row_span=1,
                col_span=c.grid_span,
                text=c.text,
            ))

    return TableMetadata(
        total_rows=num_rows,
        total_cols=num_cols,
        cells=cells,
        merge_regions=merge_regions,
    )


# ============================================================
# 阶段2：网格填充
# ============================================================

def fill_grid(meta: TableMetadata) -> List[List[str]]:
    """
    根据元数据构建完整的二维网格。

    规则：
    - vMerge restart: 文本填入当前行，注册纵向合并追踪器
    - vMerge continue: 用 origin 值填充（Excel 风格）
    - gridSpan > 1: 文本填在起始列，后续列留空
    - 普通单元格: 直接填入文本
    """
    total_rows = meta.total_rows
    total_cols = meta.total_cols

    # 创建空网格
    grid = [['' for _ in range(total_cols)] for _ in range(total_rows)]

    # 建立 vMerge origin 映射: (origin_row, origin_col) → MergeRegion
    vmerge_origins = {}
    for mr in meta.merge_regions:
        if mr.type == 'vertical':
            vmerge_origins[(mr.origin_row, mr.origin_col)] = mr

    # 纵向合并追踪器: col → {text, remaining}
    vmerge_tracker: Dict[int, dict] = {}

    for ri in range(total_rows):
        # 1. 先处理当前行被上方 vMerge 占据的列
        for col in list(vmerge_tracker.keys()):
            info = vmerge_tracker[col]
            if info['remaining'] > 0:
                grid[ri][col] = info['text']
                info['remaining'] -= 1
                if info['remaining'] <= 0:
                    del vmerge_tracker[col]

        # 2. 获取当前行的所有 cell，按 col 排序
        row_cells = sorted([c for c in meta.cells if c.row == ri], key=lambda c: c.col)

        for cell in row_cells:
            col = cell.col
            text = cell.text

            if cell.v_merge == 'restart':
                # 纵向合并起点：填入文本，注册 tracker
                grid[ri][col] = text
                origin = vmerge_origins.get((ri, col))
                if origin:
                    vmerge_tracker[col] = {
                        'text': text,
                        'remaining': origin.row_span - 1,
                    }
            elif cell.v_merge == 'continue':
                # 已由 vmerge_tracker 填充，跳过
                pass
            else:
                # 普通单元格或横向合并起点
                grid[ri][col] = text

    return grid


# ============================================================
# 阶段3：表头检测
# ============================================================

def _row_has_h_merge(ri: int, meta: TableMetadata) -> bool:
    """判断某行是否有横向合并（gridSpan > 1）"""
    for c in meta.cells:
        if c.row == ri and c.grid_span > 1:
            return True
    return False




# ============================================================
# 阶段3+4：行分类与结构检测
# ============================================================

def is_empty_row(row_data: list) -> bool:
    return all(v.strip() == "" for v in row_data)


def _is_title_row(row_data: list, total_cols: int, meta: TableMetadata, ri: int) -> bool:
    """检测标题行：只有一个非空值且跨越 >= 50% 列"""
    non_empty = [v for v in row_data if v.strip()]
    if len(non_empty) != 1:
        return False
    # 检查是否有大范围横向合并
    for c in meta.cells:
        if c.row == ri and c.grid_span >= total_cols * 0.5:
            return True
    return False


def _is_form_row(ri: int, meta: TableMetadata) -> bool:
    """表单字段行：合并起点含 key：value"""
    for c in meta.cells:
        if c.row == ri and c.grid_span > 1:
            val = c.text
            if "：" in val or ":" in val:
                return True
    return False


def _is_signing_row(row_data: list) -> bool:
    """检测签章行"""
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


def _extract_form_fields(ri: int, meta: TableMetadata) -> List[str]:
    """
    从表单行提取 key:value 字段，组合标签和值。

    例: A10:B10="申请人签字：" + C10:D10="张三" → "申请人签字：张三"
    """
    merges = []
    for c in meta.cells:
        if c.row == ri and c.grid_span > 1:
            merges.append((c.col, c.col + c.grid_span - 1, c.text.strip()))

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
            # 查找紧邻的普通单元格
            for c in meta.cells:
                if c.row == ri and c.col > end and c.grid_span == 1 and c.text.strip():
                    next_val = c.text.strip().replace("<br>", "")
                    break
        if next_val:
            fields.append(f"{val}{next_val}")
        else:
            fields.append(val)
    return fields


def _extract_signing_info(row_data: list) -> List[str]:
    """从签章行提取信息"""
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


def _is_group_header(ri: int, rows: List[Tuple[int, list]], meta: TableMetadata) -> bool:
    """
    判断是否为分组表头行。

    条件：
    1. 该行有列合并
    2. 合并区域之外没有太多非空数据（排除纯数据行）
    3. 后续行是叶子表头或另一个分组表头（而非直接数据值）
    """
    if not _row_has_h_merge(ri, meta):
        return False

    # 检查该行在合并区域之外是否有太多非空内容
    non_merge_non_empty = 0
    for c in meta.cells:
        if c.row != ri:
            continue
        # 检查此 cell 是否属于合并区域
        in_merge_span = False
        if c.grid_span > 1:
            in_merge_span = True
        elif c.v_merge in ('restart', 'continue'):
            in_merge_span = True
        if not in_merge_span and c.text.strip():
            non_merge_non_empty += 1

    if non_merge_non_empty >= 2:
        # 合并区域之外还有 2+ 个非空格 → 不是纯分组行
        return False

    # 在 rows 中找到当前行的位置
    row_pos = None
    for i, (r_idx, _) in enumerate(rows):
        if r_idx == ri:
            row_pos = i
            break
    if row_pos is None:
        return False

    # 检查后续行
    for j in range(row_pos + 1, min(row_pos + 5, len(rows))):
        _, next_data = rows[j]
        if is_empty_row(next_data):
            continue
        next_ri = rows[j][0]
        next_has_merges = _row_has_h_merge(next_ri, meta)
        if next_has_merges:
            return True
        non_empty = sum(1 for v in next_data if v.strip())
        if non_empty >= max(meta.total_cols * 0.3, 2):
            # 下一行非空格足够多，但需判断它是"列名"还是"数据值"
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


def _flatten_header_sections(group_rows: List[Tuple[int, list]], leaf_data: list,
                              meta: TableMetadata, first_col_value: str = None) -> List[str]:
    """
    将多行表头展平为单行（用于 sections 模式）。

    例: "基础工商信息" + "公司名称" → "基础工商信息/公司名称"
    """
    flattened = list(leaf_data)

    if first_col_value is not None and flattened[0].strip() == "":
        flattened[0] = first_col_value

    if not group_rows:
        return flattened

    h_merges = [m for m in meta.merge_regions if m.type == 'horizontal']

    for col_idx in range(meta.total_cols):
        col_num = col_idx
        leaf_name = flattened[col_idx].strip() if col_idx < len(flattened) else ""

        parts = []
        if leaf_name:
            parts.append(leaf_name)

        for g_row_idx, _ in reversed(group_rows):
            group_name = ""
            for gm in h_merges:
                if gm.origin_row == g_row_idx:
                    if gm.origin_col <= col_idx < gm.origin_col + gm.col_span:
                        group_name = gm.text.strip()
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

    # 阶段1：解析元数据
    meta = parse_table_metadata(table)

    if meta.total_rows == 0 or meta.total_cols == 0:
        return {
            "markdown": "*（空表格）*\n",
            "title": "", "form_fields": [], "signing_info": [],
            "row_count": 0, "col_count": 0,
        }

    # 阶段2：填充网格
    grid = fill_grid(meta)

    # 构建 rows 列表：[(0-based row_idx, row_data), ...]
    rows = [(ri, grid[ri]) for ri in range(meta.total_rows)]

    # 找首列的行合并值（用于展平表头时填充空的首列）
    first_col_merge_value = None
    for mr in meta.merge_regions:
        if mr.type == 'vertical' and mr.origin_col == 0 and mr.row_span > 1:
            first_col_merge_value = mr.text
            break

    # ---- 检测是否有表头行 ----
    has_header_row = True
    for pos, (ri, row_data) in enumerate(rows):
        if is_empty_row(row_data):
            continue

        # 条件1：第一个非空行有垂直合并 → 无表头
        for c in meta.cells:
            if c.row == ri and c.v_merge == 'restart' and c.row_span > 1:
                has_header_row = False
                break
        if not has_header_row:
            break

        # 条件2：第一个非空行是纯水平合并行 + 下一行含数据值 → 无表头
        all_non_empty_in_merge = True
        has_any_merge = False
        for c in meta.cells:
            if c.row != ri:
                continue
            if c.text.strip():
                if c.grid_span > 1:
                    has_any_merge = True
                elif c.v_merge in ('restart', 'continue'):
                    has_any_merge = True
                else:
                    all_non_empty_in_merge = False
                    break
        if has_any_merge and all_non_empty_in_merge:
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

    title_texts = []
    form_metadata = []
    signing_metadata = []

    # ============================================================
    # 无表头模式
    # ============================================================
    if not has_header_row:
        data_positions = []
        for pos, (ri, row_data) in enumerate(rows):
            if is_empty_row(row_data):
                continue
            if _is_title_row(row_data, meta.total_cols, meta, ri):
                non_empty = [v for v in row_data if v.strip()]
                if non_empty:
                    title_texts.append(non_empty[0])
                continue
            if ENABLE_SIGNING_DETECTION and _is_signing_row(row_data):
                for s in _extract_signing_info(row_data):
                    if s and s not in signing_metadata:
                        signing_metadata.append(s)
                continue
            if _is_form_row(ri, meta):
                for f in _extract_form_fields(ri, meta):
                    form_metadata.append(f)
                continue
            data_positions.append(pos)

        header = [_col_letter(c + 1) for c in range(meta.total_cols)]
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
            "row_count": meta.total_rows,
            "col_count": meta.total_cols,
        }

    # ============================================================
    # 有表头模式：sections 分组
    # ============================================================
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

    for pos, (ri, row_data) in enumerate(rows):
        if is_empty_row(row_data):
            continue

        # 1. 标题行
        if _is_title_row(row_data, meta.total_cols, meta, ri):
            flush()
            cur_groups, cur_leaf, cur_data = [], None, []
            # 提取标题文本
            for c in meta.cells:
                if c.row == ri and c.grid_span >= meta.total_cols * 0.5:
                    title_texts.append(c.text)
                    break
            else:
                title_texts.append(next((v for v in row_data if v.strip()), ""))
            continue

        # 2. 签章行
        if ENABLE_SIGNING_DETECTION and _is_signing_row(row_data):
            flush()
            cur_groups, cur_leaf, cur_data = [], None, []
            for s in _extract_signing_info(row_data):
                if s and s not in signing_metadata:
                    signing_metadata.append(s)
            continue

        # 3. 表单字段行
        if _is_form_row(ri, meta):
            flush()
            cur_groups, cur_leaf, cur_data = [], None, []
            for f in _extract_form_fields(ri, meta):
                form_metadata.append(f)
            continue

        # 4. 分组表头行
        if _is_group_header(ri, rows, meta):
            if cur_leaf is not None and cur_data:
                flush()
                cur_groups = []
                cur_data = []
            cur_groups.append(pos)
            cur_leaf = None
            continue

        # 5. 叶子表头行 vs 数据行
        has_merges = _row_has_h_merge(ri, meta)
        non_empty = sum(1 for v in row_data if v.strip())
        numeric_count = sum(
            1 for v in row_data
            if v.strip() and re.match(r"^[\d,.¥￥$€%]+$", v.strip())
        )

        if not has_merges and non_empty >= max(meta.total_cols * 0.3, 2):
            if cur_leaf is None:
                cur_leaf = pos
            else:
                cur_data.append(pos)
        else:
            if cur_leaf is not None:
                cur_data.append(pos)
            elif non_empty >= 2:
                cur_data.append(pos)

    flush()

    # ---- 输出 ----
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

        header = _flatten_header_sections(
            group_rows, leaf_data, meta,
            first_col_value=first_col_merge_value,
        )
        actual_cols = max(len(header), max((len(r) for r in data_rows), default=0))
        _output_table(lines, header, data_rows, actual_cols)

    # 无 sections 时（简单表格）
    if not sections:
        data_positions = []
        leaf_pos = None
        for pos, (ri, row_data) in enumerate(rows):
            if is_empty_row(row_data):
                continue
            rt = None
            if _is_title_row(row_data, meta.total_cols, meta, ri):
                rt = "title"
            elif ENABLE_SIGNING_DETECTION and _is_signing_row(row_data):
                rt = "signing"
            elif _is_form_row(ri, meta):
                rt = "form"

            if rt == "title":
                for c in meta.cells:
                    if c.row == ri and c.grid_span >= meta.total_cols * 0.5:
                        title_texts.append(c.text)
                        break
                continue
            if rt == "form":
                for f in _extract_form_fields(ri, meta):
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
            header = _flatten_header_sections([], leaf_data, meta,
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
        "row_count": meta.total_rows,
        "col_count": meta.total_cols,
    }
