"""
表格布局重组
===========
将 OCR 文字块与表格识别结果按空间位置重组，
保持原始阅读顺序，表格区域替换为结构化 Markdown。

核心能力：
1. 基于位置的表格检测 — 不依赖 PaddleOCR 结构识别，纯靠 OCR 块坐标聚类
2. 行聚类 / 列边界检测 / 表格区域识别
3. 段落与表格分离，表格输出为 Markdown 格式
"""

import logging
from typing import Optional

from ..ocr import OCRTextBlock
from .models import (
    TableCell,
    TableRecognitionResult,
)
from .formatter import table_to_markdown

logger = logging.getLogger(__name__)


# ============================================================
# 配置常量（可从 table/config.py 覆盖）
# ============================================================

# 行聚类：cy 差值 < median_height * ROW_CLUSTER_RATIO 视为同行
ROW_CLUSTER_RATIO = 0.5

# 列检测：左边缘 x 坐标差值 < COL_TOLERANCE_PX 视为同列
COL_TOLERANCE_PX = 15

# 表格判定：至少 TABLE_MIN_ROWS 行且每行至少 TABLE_MIN_COLS 列
TABLE_MIN_ROWS = 3
TABLE_MIN_COLS = 2

# 表格区域连续性：允许的最大空行间隔
TABLE_MAX_ROW_GAP = 1


# ============================================================
# 行聚类
# ============================================================

def cluster_into_rows(
    blocks: list[OCRTextBlock],
    threshold: Optional[float] = None,
) -> list[list[OCRTextBlock]]:
    """将 OCR 文字块按 y 坐标聚类为行

    算法：
    1. 计算所有块高度的中位数，乘以系数得到行聚类阈值
    2. 按 cy 排序后，相邻块 cy 差值 < 阈值 → 归入同一行
    3. 同行内按 cx 排序

    Args:
        blocks: OCR 文字块列表
        threshold: 自定义聚类阈值，None 时自动计算

    Returns:
        行列表，每行是按 cx 排序的文字块列表
    """
    if not blocks:
        return []

    if threshold is None:
        heights = sorted(b.y1 - b.y0 for b in blocks if b.y1 > b.y0)
        median_h = heights[len(heights) // 2] if heights else 20
        threshold = max(median_h * ROW_CLUSTER_RATIO, 5)

    sorted_blocks = sorted(blocks, key=lambda b: b.cy)

    rows: list[list[OCRTextBlock]] = []
    current_row = [sorted_blocks[0]]

    for b in sorted_blocks[1:]:
        # 与当前行中所有块的 cy 平均值比较，避免单块偏差累积
        row_cy_avg = sum(blk.cy for blk in current_row) / len(current_row)
        if abs(b.cy - row_cy_avg) <= threshold:
            current_row.append(b)
        else:
            current_row.sort(key=lambda x: x.cx)
            rows.append(current_row)
            current_row = [b]

    if current_row:
        current_row.sort(key=lambda x: x.cx)
        rows.append(current_row)

    return rows


# ============================================================
# 列边界检测
# ============================================================

def _cluster_values(values: list[float], tolerance: float) -> list[float]:
    """将一组数值按容差聚类，返回聚类中心（升序）"""
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters: list[list[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def detect_column_boundaries(
    rows: list[list[OCRTextBlock]],
    tolerance: float = COL_TOLERANCE_PX,
) -> list[float]:
    """从多行文字块中检测列边界

    算法：
    1. 收集所有块的左边缘 x 坐标
    2. 按容差聚类（相近的左边缘视为同一列的起始位置）
    3. 只在足够多行中出现的列起始位置才被保留（过滤段落首行缩进等噪音）

    Args:
        rows: 行列表（每行是按 cx 排序的块列表）
        tolerance: 聚类容差（像素）

    Returns:
        列起始 x 坐标列表（升序），长度 = 检测到的列数
    """
    if not rows:
        return []

    # 收集每行中每个块的左边缘
    all_left_edges: list[float] = []
    for row in rows:
        for blk in row:
            all_left_edges.append(float(blk.x0))

    if not all_left_edges:
        return []

    # 聚类左边缘
    col_starts = _cluster_values(all_left_edges, tolerance)

    # 过滤：只保留在 >= 30% 行中出现的列起始位置
    min_row_count = max(2, len(rows) * 0.3)
    filtered: list[float] = []
    for cs in col_starts:
        count = 0
        for row in rows:
            if any(abs(blk.x0 - cs) <= tolerance for blk in row):
                count += 1
        if count >= min_row_count:
            filtered.append(cs)

    return filtered


def _assign_block_to_column(
    blk: OCRTextBlock,
    col_starts: list[float],
    tolerance: float = COL_TOLERANCE_PX,
) -> int:
    """将文字块分配到最近的列

    Returns:
        列索引（0-based）
    """
    best_col = 0
    best_dist = float("inf")
    for i, cs in enumerate(col_starts):
        dist = abs(blk.x0 - cs)
        if dist < best_dist:
            best_dist = dist
            best_col = i
    return best_col


# ============================================================
# 表格区域检测
# ============================================================

def _row_gap_signature(
    row: list[OCRTextBlock],
    tolerance: float = COL_TOLERANCE_PX,
) -> list[float]:
    """计算一行内相邻块之间的水平间隙位置"""
    if len(row) < 2:
        return []
    sorted_row = sorted(row, key=lambda b: b.cx)
    gaps = []
    for i in range(len(sorted_row) - 1):
        gap_center = (sorted_row[i].x1 + sorted_row[i + 1].x0) / 2
        gaps.append(gap_center)
    return gaps


def _signatures_match(
    sig_a: list[float],
    sig_b: list[float],
    tolerance: float = COL_TOLERANCE_PX * 2,
) -> bool:
    """判断两行的间隙特征是否匹配（列数相同且位置接近）"""
    if len(sig_a) != len(sig_b):
        return False
    if not sig_a:
        return False
    matched = sum(
        1 for a in sig_a
        if any(abs(a - b) <= tolerance for b in sig_b)
    )
    return matched >= len(sig_a) * 0.6


def detect_table_regions(
    rows: list[list[OCRTextBlock]],
    min_cols: int = TABLE_MIN_COLS,
    min_rows: int = TABLE_MIN_ROWS,
) -> list[tuple[int, int]]:
    """识别哪些行序列构成表格

    算法：
    1. 计算每行的间隙特征（相邻块之间的水平间隙中心位置）
    2. 连续行如果间隙特征匹配（列数相同、位置接近），归入同一表格区域
    3. 过滤掉列数不足或行数不足的区域

    Args:
        rows: 所有行（从上到下）
        min_cols: 最少列数才算表格
        min_rows: 最少行数才算表格

    Returns:
        表格区域列表 [(start_row_idx, end_row_idx), ...]，索引为 rows 中的位置
    """
    if len(rows) < min_rows:
        return []

    # 计算每行的间隙特征
    signatures = [_row_gap_signature(row) for row in rows]

    # 连续匹配的行归入同一组
    groups: list[tuple[int, int]] = []
    start = None

    for i in range(len(rows)):
        is_table_row = len(signatures[i]) >= min_cols - 1  # N 列有 N-1 个间隙

        if is_table_row and start is None:
            start = i
        elif is_table_row and start is not None:
            # 检查当前行与前一行的间隙特征是否匹配
            if not _signatures_match(signatures[i], signatures[i - 1]):
                # 特征不匹配，结束当前组，开始新组
                groups.append((start, i - 1))
                start = i
        elif not is_table_row and start is not None:
            groups.append((start, i - 1))
            start = None

    if start is not None:
        groups.append((start, len(rows) - 1))

    # 过滤：行数不足的区域
    result = []
    for s, e in groups:
        row_count = e - s + 1
        if row_count >= min_rows:
            result.append((s, e))

    return result


# ============================================================
# 从位置构建表格
# ============================================================

def build_table_from_positions(
    table_rows: list[list[OCRTextBlock]],
    col_starts: list[float],
    row_offset: int = 0,
    tolerance: float = COL_TOLERANCE_PX,
) -> TableRecognitionResult:
    """从行+列边界构建 TableRecognitionResult

    Args:
        table_rows: 表格区域的行列表
        col_starts: 列起始 x 坐标
        row_offset: 行号偏移量（用于多表格场景）
        tolerance: 列分配容差

    Returns:
        TableRecognitionResult
    """
    n_cols = len(col_starts)
    n_rows = len(table_rows)
    cells: list[TableCell] = []

    for ri, row in enumerate(table_rows):
        # 按列分组（同一列可能有多个块，需合并文本）
        col_texts: dict[int, list[str]] = {}
        for blk in row:
            col = _assign_block_to_column(blk, col_starts, tolerance)
            col_texts.setdefault(col, []).append(blk.text)

        for ci in range(n_cols):
            text_parts = col_texts.get(ci, [])
            text = " ".join(text_parts).strip()
            if text:
                cells.append(TableCell(
                    row=ri,
                    col=ci,
                    text=text,
                    row_span=1,
                    col_span=1,
                ))

    # 计算 bbox
    all_blocks = [blk for row in table_rows for blk in row]
    if all_blocks:
        bbox = (
            min(b.x0 for b in all_blocks),
            min(b.y0 for b in all_blocks),
            max(b.x1 for b in all_blocks),
            max(b.y1 for b in all_blocks),
        )
    else:
        bbox = ()

    result = TableRecognitionResult(
        source="position_clustering",
        cells=cells,
        total_rows=n_rows,
        total_cols=n_cols,
        confidence=_estimate_confidence(cells, n_rows, n_cols),
        bbox=bbox,
    )
    result.raw_grid = result.grid

    logger.info(
        f"位置聚类表格: {n_rows}行 × {n_cols}列, "
        f"{len(cells)} 个非空单元格, "
        f"bbox={bbox}"
    )
    for ri, row in enumerate(result.grid):
        logger.info(f"  行{ri}: {row}")

    return result


def _estimate_confidence(cells: list[TableCell], n_rows: int, n_cols: int) -> float:
    """根据非空单元格比例估算置信度"""
    total = n_rows * n_cols
    if total == 0:
        return 0.0
    non_empty = sum(1 for c in cells if c.text.strip())
    return non_empty / total


# ============================================================
# 端到端：从 OCR 块自动检测表格
# ============================================================

def detect_tables_from_blocks(
    blocks: list[OCRTextBlock],
    min_cols: int = TABLE_MIN_COLS,
    min_rows: int = TABLE_MIN_ROWS,
    col_tolerance: float = COL_TOLERANCE_PX,
) -> tuple[list[TableRecognitionResult], list[tuple[int, int]]]:
    """从 OCR 文字块中自动检测并构建表格

    完整流程：
    1. 行聚类
    2. 表格区域检测
    3. 每个表格区域独立做列检测 + 构建表格

    Args:
        blocks: 页面上所有 OCR 文字块
        min_cols: 最少列数
        min_rows: 最少行数
        col_tolerance: 列聚类容差

    Returns:
        (tables, table_row_ranges):
        - tables: TableRecognitionResult 列表
        - table_row_ranges: 对应的行索引范围 [(start, end), ...]
    """
    rows = cluster_into_rows(blocks)
    if not rows:
        return [], []

    table_regions = detect_table_regions(rows, min_cols, min_rows)
    if not table_regions:
        return [], []

    tables: list[TableRecognitionResult] = []
    for start, end in table_regions:
        table_rows = rows[start:end + 1]

        # 对每个表格区域独立做列检测
        col_starts = detect_column_boundaries(table_rows, col_tolerance)
        if len(col_starts) < min_cols:
            logger.info(
                f"表格区域 [{start}:{end}] 列数不足 "
                f"({len(col_starts)} < {min_cols})，跳过"
            )
            continue

        table = build_table_from_positions(
            table_rows, col_starts, row_offset=start, tolerance=col_tolerance,
        )
        tables.append(table)

    return tables, table_regions


# ============================================================
# 排序与重组（保持向后兼容）
# ============================================================

def sort_blocks_by_reading_order(blocks: list[OCRTextBlock]) -> list[OCRTextBlock]:
    """行聚类排序：cy 接近的归为同一行，同行内按 cx 排序"""
    if not blocks:
        return []
    rows = cluster_into_rows(blocks)
    result: list[OCRTextBlock] = []
    for row in rows:
        result.extend(row)
    return result


def reconstruct_structured_text(
    blocks: list[OCRTextBlock],
    tables: Optional[list[TableRecognitionResult]] = None,
    auto_detect_tables: bool = True,
) -> str:
    """用位置信息重组 OCR 文本，保持原始布局

    算法：
    1. 按阅读顺序排列所有文字块
    2. 如果有表格（传入或自动检测），将表格区域内的块替换为 Markdown 表格
    3. 不在表格内的块按原序保留为段落

    Args:
        blocks: OCR 文字块列表
        tables: 外部传入的表格识别结果。为空且 auto_detect_tables=True 时
                会自动从块位置检测表格
        auto_detect_tables: 当 tables 为空时是否自动检测

    Returns:
        重组后的文本（段落 + Markdown 表格混合）
    """
    if not blocks:
        return ""

    sorted_blocks = sort_blocks_by_reading_order(blocks)
    logger.info(
        f"[布局重组] 排序后 {len(sorted_blocks)} 个文字块"
    )

    # 如果没有传入表格，尝试自动检测
    table_row_ranges: list[tuple[int, int]] = []
    if (not tables or not any(
        t.bbox and len(t.bbox) >= 4 and t.grid for t in tables
    )) and auto_detect_tables:
        tables, table_row_ranges = detect_tables_from_blocks(sorted_blocks)
        if tables:
            logger.info(
                f"[布局重组] 自动检测到 {len(tables)} 个表格"
            )

    if not tables:
        logger.info("[布局重组] 无表格，全部按段落输出")
        return "\n".join(b.text for b in sorted_blocks)

    # 构建有效表格列表
    valid_tables = [
        (t, t.bbox) for t in tables
        if t.bbox and len(t.bbox) >= 4 and t.grid
    ]
    valid_tables.sort(key=lambda pair: pair[1][1])

    for ti, t in enumerate(tables):
        if (t, t.bbox) not in valid_tables:
            logger.info(
                f"[布局重组] 表格[{ti}] 被过滤: bbox={t.bbox}, grid 为空={not t.grid}"
            )

    # 将每个块分配到表格或段落
    block_table_map: list[int] = []
    for blk in sorted_blocks:
        assigned = -1
        for t_idx, (_, tbl_bbox) in enumerate(valid_tables):
            if blk.inside_bbox(tbl_bbox):
                assigned = t_idx
                break
        block_table_map.append(assigned)

    in_table = sum(1 for a in block_table_map if a >= 0)
    out_table = sum(1 for a in block_table_map if a == -1)
    logger.info(
        f"[布局重组] 文字块归属: 表格 {in_table} 个, 段落 {out_table} 个"
    )

    # 按阅读顺序输出，表格区域替换为 Markdown
    parts: list[str] = []
    emitted_tables: set[int] = set()

    for i, blk in enumerate(sorted_blocks):
        t_idx = block_table_map[i]

        if t_idx == -1:
            parts.append(blk.text)
        else:
            if t_idx not in emitted_tables:
                tbl, _ = valid_tables[t_idx]
                md = table_to_markdown(tbl)
                if md:
                    parts.append(md)
                    logger.info(
                        f"[布局重组] 插入表格[{t_idx}] "
                        f"({tbl.total_rows}行×{tbl.total_cols}列)"
                    )
                emitted_tables.add(t_idx)

    # 补充未被任何块触发的表格
    for ti in range(len(valid_tables)):
        if ti not in emitted_tables:
            t, bb = valid_tables[ti]
            md = table_to_markdown(t)
            if md:
                parts.append(md)
                logger.info(
                    f"[布局重组] 补充表格[{ti}] (无块触发)"
                )

    result = "\n".join(parts)
    logger.info(f"[布局重组] 最终输出 {len(parts)} 段, {len(result)} 字符")
    return result
