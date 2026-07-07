"""
PDF 表格处理器
==============
支持两种表格提取方式：
1. 视觉检测（默认）：使用 PyMuPDF 的 page.find_tables() 检测表格
2. 结构树提取（标签PDF优先）：从 PDF 的 StructTreeRoot 解析 /Table /TR /TD 节点

PyMuPDF 表格 API（视觉模式）：
- page.find_tables() → TableFinder
- TableFinder.tables → list[Table]
- Table.extract() → list[list[str]] 提取的文本数据（二维网格，合并处为 None/空）
- Table.bbox → (x0, y0, x1, y1) 表格边界框
- Table.cells → list[(x0, y0, x1, y1)] 每个单元格的页面坐标（注意：不是网格索引！）

结构树 API（Tagged PDF 模式）：
- doc.pdf_catalog() → 获取 Catalog xref
- doc.xref_object(xref) → 读取 PDF 对象字典
- StructTreeRoot → /Table → /TBody → /TR → /TD → 文本内容

合并检测策略（基于 extract() 网格分析，视觉模式）：
- 水平合并：某行非空单元格后紧跟连续空/None 单元格
- 垂直合并：某列非空单元格下方连续行同列为空
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from ai_sag.doc_parser.pdf.v2.config import ENABLE_SIGNING_DETECTION, SIGNING_KEYWORDS, TABLE_MIN_ROWS, TABLE_MIN_COLS

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构（与 Word v2 保持一致）
# ============================================================

@dataclass
class CellInfo:
    """表格中单个单元格的信息"""
    row: int = 0          # 0-based 行号
    col: int = 0          # 0-based 列号（起始列）
    grid_span: int = 1    # 横向合并列数
    row_span: int = 1     # 纵向合并行数
    text: str = ""        # 单元格文本


@dataclass
class MergeRegion:
    """一个合并区域的描述"""
    type: str = ""          # 'horizontal', 'vertical' 或 'block'
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
# 阶段0：Tagged PDF 结构树表格提取（优先路径）
# ============================================================

@dataclass
class StructTableResult:
    """
    从 PDF StructTreeRoot 提取的表格结果。
    模拟 PyMuPDF Table 对象的接口（duck-typing），
    以便复用 table_to_markdown() 流水线。
    """
    _grid: List[List[str]]       # 二维文本网格 [row][col]
    _bbox: tuple                  # (x0, y0, x1, y1) 页面坐标

    def extract(self) -> List[List[str]]:
        """模拟 Table.extract()"""
        return self._grid

    @property
    def bbox(self) -> tuple:
        """模拟 Table.bbox"""
        return self._bbox


def is_tagged_pdf(doc) -> bool:
    """检测 PDF 是否为标签 PDF（包含 StructTreeRoot）"""
    try:
        cat_xref = doc.pdf_catalog()
        cat_obj = doc.xref_object(cat_xref)
        return 'StructTreeRoot' in (cat_obj or '')
    except Exception:
        return False


def _read_struct_node(doc, xref: int) -> Optional[dict]:
    """
    读取一个结构树节点的 PDF 对象字典。

    返回解析后的关键字段：
    - tag (/S): 标签名（Table, TR, TD, Span, P 等）
    - kids (/K): 子节点 xref 列表
    - page_xref (/Pg): 所属页面对象 xref
    - actual_text (/ActualText): 直接存储的文本内容
    - mcid (/K 纯整数): 标记内容 ID（引用页面内容流）
    """
    try:
        raw = doc.xref_object(xref)
        if not raw:
            return None
    except Exception:
        return None

    result = {'_xref': xref}

    # 提取 /S (tag type)
    s_match = re.search(r'/S\s*/(\w+)', raw)
    if s_match:
        result['tag'] = s_match.group(1)

    # 提取 /Pg (page reference)
    pg_match = re.search(r'/Pg\s*(\d+)\s+\d+\s*R', raw)
    if pg_match:
        result['page_xref'] = int(pg_match.group(1))

    # 提取 /ActualText — 结构树中最常用的文本存储方式
    at_match = re.search(r'/ActualText\s*\(([^)]*)\)', raw)
    if at_match:
        text = at_match.group(1)
        text = text.replace('\\n', ' ').replace('\\t', ' ')
        text = text.replace('\\(', '(').replace('\\)', ')')
        result['actual_text'] = text.strip()

    # 提取 /K 为纯整数的情况（MCID — Marked Content ID）
    # 格式: /K 29  （不是 /K 29 0 R 这种 xref 引用）
    k_mcid = re.search(r'^\s*/K\s+(\d+)\s*$', raw, re.MULTILINE)
    if k_mcid:
        result['mcid'] = int(k_mcid.group(1))

    # 提取 /K 为子节点数组 [xref1 xref2 ...]
    kids = []
    k_arr = re.search(r'/K\s*\[([^\]]*)\]', raw)
    if k_arr:
        kids_str = k_arr.group(1).strip()
        for ref_match in re.finditer(r'(\d+)\s+\d+\s*R', kids_str):
            kids.append(int(ref_match.group(1)))
    else:
        k_single = re.search(r'/K\s+(\d+)\s+\d+\s*R', raw)
        if k_single:
            kids = [int(k_single.group(1))]

    if kids:
        result['kids'] = kids

    return result


def _get_page_num(doc, page_xref: int) -> int:
    """根据页面对象 xref 获取页码（1-based）"""
    for i in range(len(doc)):
        try:
            if doc.page_xref(i) == page_xref:
                return i + 1
        except Exception:
            continue
    return -1


def _build_mcid_text_map(page) -> Dict[int, str]:
    """
    从页面内容流解析 BDC(/MCID N)...Tj(text)...EMC 序列，
    建立 {mcid: 提取到的文本} 映射。

    这是连接「结构树 MCID 引用」和「实际文本内容」的关键桥梁。
    类似于 Word XML 中通过 rId 关联外部资源的过程。

    Args:
        page: PyMuPDF Page 对象

    Returns:
        {mcid_int: text_string} 字典
    """
    try:
        contents = page.get_contents()
    except Exception:
        return {}

    mcid_map: Dict[int, str] = {}

    for cxref in contents:
        try:
            stream_data = page.parent.xref_stream(cxref)
            raw = stream_data.decode('latin-1', errors='replace')
        except Exception:
            continue

        # 找到每个 /MCID N 出现的位置
        for m in re.finditer(r'/MCID\s+(\d+)', raw):
            mcid = int(m.group(1))
            bdc_search = re.search(r'BDC', raw[m.start():m.start() + 30])
            if not bdc_search:
                continue
            region_start = m.start() + bdc_search.end()

            emc_match = re.search(r'\bEMC\b', raw[region_start:])
            if not emc_match:
                continue

            region_end = region_start + emc_match.start()
            region = raw[region_start:region_end]

            # 在区域内提取文本操作符: (...)Tj 和 [...]TJ
            texts_found: List[str] = []
            for tj in re.finditer(r'\(([^)]{1,200})\)\s*Tj', region):
                t = tj.group(1).replace('\\(', '(').replace('\\)', ')')
                if t.strip():
                    texts_found.append(t.strip())
            for tj_arr in re.finditer(r'\[(.*?)\]\s*TJ', region, re.DOTALL):
                for txt in re.findall(r'\(([^)]{1,200})\)', tj_arr.group(1)):
                    t = txt.replace('\\(', '(').replace('\\)', ')')
                    if t.strip():
                        texts_found.append(t.strip())

            if texts_found:
                full = ' '.join(texts_found)
                if mcid not in mcid_map:
                    mcid_map[mcid] = full
                else:
                    mcid_map[mcid] += ' ' + full

    return mcid_map


def _collect_td_content(doc, td_xref: int,
                         mcid_text_map: Dict[int, str],
                         depth: int = 0) -> Tuple[List[int], List[str]]:
    """
    从 TD 节点递归收集所有的 MCID 引用和 ActualText。

    这是核心递归函数——就像 Word XML 中遍历 <w:tc> 下所有
    <w:p><w:r><w:t> 一样，完整收集单元格内的所有文本片段。

    Args:
        doc: PyMuPDF Document
        td_xref: TD 节点的 xref
        mcid_text_map: 页面的 MCID→文本映射（从内容流构建）
        depth: 当前递归深度（防止无限循环）

    Returns:
        (mcids_list, texts_list): 收集到的 MCID 列表和 ActualText 文本列表
    """
    if depth > 10:
        return [], []

    node = _read_struct_node(doc, td_xref)
    if not node:
        return [], []

    mcids: List[int] = []
    texts: List[str] = []

    # 1. 自身的 ActualText
    if node.get('actual_text'):
        texts.append(node['actual_text'])

    # 2. 自身的 MCID
    if node.get('mcid') is not None:
        mcids.append(node['mcid'])

    # 3. 递归子节点（按类型分派处理策略）
    for kid_xref in node.get('kids', []):
        child = _read_struct_node(doc, kid_xref)
        if not child:
            continue

        child_tag = child.get('tag', '')

        if child_tag == 'Span':
            # Span 叶子节点 — 最常见的文本载体
            if child.get('actual_text'):
                texts.append(child['actual_text'])
            if child.get('mcid') is not None:
                mcids.append(child['mcid'])
            elif child.get('kids'):
                sub_mcids, sub_texts = _collect_td_content(
                    doc, kid_xref, mcid_text_map, depth + 1)
                mcids.extend(sub_mcids)
                texts.extend(sub_texts)

        elif child_tag == 'P':
            # 段落容器 — 递归进入
            sub_mcids, sub_texts = _collect_td_content(
                doc, kid_xref, mcid_text_map, depth + 1)
            mcids.extend(sub_mcids)
            texts.extend(sub_texts)

        elif child_tag == 'TD':
            # 嵌套 TD（极少见），只取文本不递归结构
            sub_mcids, sub_texts = _collect_td_content(
                doc, kid_xref, mcid_text_map, depth + 1)
            texts.extend(sub_texts)

        elif child.get('kids'):
            # NonStruct, Artifact 等其他容器，继续深入
            sub_mcids, sub_texts = _collect_td_content(
                doc, kid_xref, mcid_text_map, depth + 1)
            mcids.extend(sub_mcids)
            texts.extend(sub_texts)

    return mcids, texts


def _resolve_cell_text(mcids: List[int], texts: List[str],
                        mcid_text_map: Dict[int, str]) -> str:
    """
    将收集到的 MCID 和 ActualText 解析为最终单元格文本。

    去重策略：MCID 映射的文本优先，ActualText 补充且去重。
    """
    parts: List[str] = []

    # MCID 解析（优先）
    for mcid in mcids:
        if mcid in mcid_text_map and mcid_text_map[mcid]:
            parts.append(mcid_text_map[mcid])

    # ActualText 补充（大小写不敏感去重）
    seen_lower = {p.lower().strip() for p in parts}
    for t in texts:
        t_clean = t.strip()
        if t_clean and t_clean.lower() not in seen_lower:
            parts.append(t_clean)
            seen_lower.add(t_clean.lower())

    return ' '.join(parts)


# ============================================================
# 阶段0b：表格文本质量工具
# ============================================================

def _clean_cell_text(text: str) -> str:
    """
    清理单元格文本，消除 PDF 提取产生的冗余字符。

    处理的问题：
    1. 内部换行 → 空格或直接合并（"应用\\n账号" → "应用账号"）
    2. 多余空白 → 单空格（"V   1 . 7" → "V1.7"）
    3. CJK 字符间多余空格（"张 路 路" → "张路路"）
    4. 尾部标点残留（"clientkey\\n." → "clientkey"）
    5. 页码泄漏（纯数字/罗马数字如 "II", "3", "5"）
    6. OCR 散列模式（"V 1 . 7", "2 019 - 02 25", "S H A"）

    Args:
        text: 原始单元格文本

    Returns:
        清理后的文本
    """
    if not text:
        return ""

    t = text.strip()

    # 1. 内部换行处理
    t = re.sub(r'[\r\n\t]+', ' ', t)

    # 2. 压缩多个连续空白为单个空格
    t = re.sub(r'[ ]{2,}', ' ', t)

    # 3. 去掉 CJK 字符之间的空格
    t = re.sub(r'([\u4e00-\u9fff\u3400-\u4dbf]) ([\u4e00-\u9fff\u3400-\u4dbf])', r'\1\2', t)

    # 4. 去掉字母数字和 CJK 字符之间不必要的空格
    t = re.sub(r'([a-zA-Z0-9]) ([\u4e00-\u9fff])', r'\1\2', t)
    t = re.sub(r'([\u4e00-\u9fff]) ([a-zA-Z0-9])', r'\1\2', t)

    # 5. 清理首尾无意义字符
    t = re.sub(r'^[\s\.,;:!?\-/_]+', '', t)
    t = re.sub(r'[\s\.,;:!?\-/_]+$', '', t)

    # 6. OCR 散列检测与修复
    # 模式A: 全大写+空格散列 ("S H A", "R S A")
    upper_match = re.match(r'^([A-Z][\sA-Z]{2,})$', t)
    if upper_match:
        cleaned_upper = ''.join(upper_match.group(1).split())
        if len(cleaned_upper) >= 2:
            t = cleaned_upper

    # 模式B: 混合字母数字散列 ("V 1 . 7", "2 019 - 02 25")
    # 条件: 空格数量 >= 非空格数量的一半（说明是被散列的）
    space_count = t.count(' ')
    non_space = len(t.replace(' ', ''))
    if non_space > 0 and space_count >= non_space * 0.3 and len(t) >= 3:
        # 尝试去掉所有内部空格
        candidate = ''.join(t.split())
        # 确保结果不是乱拼：至少包含2个字母数字字符（允许被标点分隔）
        if re.search(r'[a-zA-Z\d]', candidate) and len(candidate) >= 2:
            t = candidate

    # 模式C: 去掉 ASCII 字母/数字之间的孤立空格 ("a pp" → "app", "1 .4" → "1.4")
    # 条件: 单个空格两侧都是 ASCII 字母或数字，且去掉后能形成有意义的词
    t = re.sub(r'([a-zA-Z]) ([a-zA-Z])', r'\1\2', t)
    t = re.sub(r'(\d) ([\.\d])', r'\1\2', t)       # "1 .4" → "1.4"
    t = re.sub(r'([\.\d]) (\d)', r'\1\2', t)       # "2019 - 02" → "2019-02"

    # 7. 去重：检测并修复重复拼接（如 SHA256256 → SHA256）
    # 常见于结构树已有值 + find_tables 融合产生的不完整重复
    if len(t) >= 4:
        deduped = t
        # 尝试从后向前找最长重复后缀：SHA256256 → SHA256 + 256 → SHA256
        for cut in range(len(t) - 2, len(t) // 2, -1):
            suffix = t[cut:]
            prefix = t[:cut]
            # 后缀是前缀的尾部子串 (SHA256 + 256, 其中 256 是前缀末尾的一部分)
            if suffix and prefix.endswith(suffix):
                deduped = prefix
                break
            # 或前后完全相同 (ABAB → AB)
            if prefix == suffix:
                deduped = prefix
                break
        if len(deduped) < len(t) and len(deduped) >= 2:
            t = deduped

    return t.strip()


def _is_valid_fusion_candidate(text: str) -> bool:
    """
    判断 find_tables() 规则检测的文本是否适合用于融合填充。

    拒绝以下类型的文本（find_tables 常见噪声）：
    1. 单个 CJK 字符（"文"、"作"、"者"）— 通常是被拆散的竖排文字
    2. 纯页码数字（"II", "III", "3", "5", "12"） — 页码泄漏
    3. 过短的无意义片段（≤1 个非空格字符）
    4. 全是点号/横线的内容（"............", "---"）
    """
    if not text:
        return False

    t = text.strip()
    if not t:
        return False

    # 规则1: 单个 CJK 字符 → 拒绝
    if re.match(r'^[\u4e00-\u9fff\u3400-\u4dbf]$', t):
        return False

    # 规则2: 罗马数字页码 (I, II, III, IV, V...XII 等)
    roman_pattern = r'^[IVXLCDM]+$'
    if re.fullmatch(roman_pattern, t) and len(t) <= 8:
        return False

    # 规则3: 纯数字且很短 (可能是行号/页码)
    if re.fullmatch(r'^\d{1,3}$', t):
        return False

    # 规则4: 全是点号、横线等无意义填充
    stripped = re.sub(r'[\.\-\s_]+', '', t)
    if not stripped:
        return False

    return True


def _looks_like_toc(grid: list) -> bool:
    """
    检测一个网格是否像目录(TOC)而不是真正的表格。

    目录特征：
    1. 第一列通常是章节编号（"1.", "1.1", "2."）
    2. 第二列是标题文本
    3. 第三列是页码（纯数字）
    4. 标题列中有大量点号填充线（"概述 .............. 4"）

    Args:
        grid: 二维文本网格 [[str, ...], ...]

    Returns:
        True 表示像是目录，应过滤掉
    """
    if not grid or len(grid) < 3:
        return False

    rows = len(grid)
    cols = max(len(r) for r in grid)

    if cols < 2:
        return False

    toc_score = 0

    for row in grid:
        if not row or len(row) < 2:
            continue

        last_cell = str(row[-1]).strip()

        # 特征1: 最后一列是短纯数字（页码）
        if re.fullmatch(r'^\d{1,4}$', last_cell):
            toc_score += 2

        # 特征2: 某一列包含大量点号填充线（目录的点引导线）
        for cell in row:
            cell_str = str(cell).strip()
            dot_count = cell_str.count('.')
            non_dot_len = len(cell_str.replace('.', '').replace(' ', ''))
            # 点号占字符串长度一半以上，且有足够多的点
            if dot_count > 10 and non_dot_len > 0 and dot_count / len(cell_str) > 0.5:
                toc_score += 3
                break

        # 特征3: 第一列看起来像章节编号
        first = str(row[0]).strip()
        if re.match(r'^[\d\.]+\.?$', first) and len(first) <= 10:
            toc_score += 1

    # 如果大部分行都符合目录特征
    threshold = rows * 1.0  # 平均每行至少贡献1分
    return toc_score >= threshold


def _fuse_with_visual(struct_results: list, page) -> list:
    """
    融合策略：用 find_tables() 规则检测结果填充结构树表格中的空缺单元格。

    核心思路：
    1. 对同页运行 page.find_tables() 获取基于线条+文本位置的规则检测结果
    2. 按行数×列数相似度匹配 结构树表格 ↔ 规则检测表格
    3. 用规则检测网格中对应位置的文本填充结构树网格的空字符串单元格

    Args:
        struct_results: [StructTableResult, ...] 从结构树提取的表格列表
        page: PyMuPDF Page 对象（用于运行 find_tables）

    Returns:
        填充后的 [StructTableResult, ...] 列表（原对象会被就地修改）
    """
    if not struct_results:
        return struct_results

    # 运行 find_tables() 规则检测获取补充数据源
    try:
        finder = page.find_tables()
        visual_tables = finder.tables
    except Exception:
        logger.debug("融合: find_tables() 规则检测失败，跳过")
        return struct_results

    if not visual_tables:
        return struct_results

    total_fused = 0

    for st in struct_results:
        s_grid = st.extract()
        if not s_grid:
            continue
        s_rows = len(s_grid)
        s_cols = max(len(r) for r in s_grid) if s_grid else 0
        if s_rows == 0 or s_cols == 0:
            continue

        # 统计结构树中有多少空单元格需要填充
        empty_count = sum(1 for r in s_grid for c in r if not c.strip())
        if empty_count == 0:
            continue

        # 找最佳匹配的规则检测表格（按维度相似度）
        best_vt = None
        best_score = -1

        for vt in visual_tables:
            try:
                v_data = vt.extract()
            except Exception:
                continue
            if not v_data:
                continue
            v_rows = len(v_data)
            v_cols = max(len(r) for r in v_data) if v_data else 0

            # 相似度评分：行和列都接近则得分高
            row_diff = abs(s_rows - v_rows)
            col_diff = abs(s_cols - v_cols)

            # 要求至少行数和列数都不超过2倍差异
            if v_rows > 0 and s_rows / v_rows > 2.0:
                continue
            if v_cols > 0 and s_cols / v_cols > 2.0:
                continue

            score = -(row_diff * row_diff + col_diff * col_diff)
            if score > best_score:
                best_score = score
                best_vt = vt

        if best_vt is None:
            continue

        try:
            v_grid = best_vt.extract()
        except Exception:
            continue
        if not v_grid:
            continue

        v_rows = len(v_grid)
        v_cols = max(len(r) for r in v_grid) if v_grid else 0

        # 标准化视觉网格（None → ""）+ 文本清理 + 质量门控
        norm_visual = []
        for r in v_grid:
            nr = []
            for c in r:
                raw = str(c).strip() if c is not None else ""
                nr.append(_clean_cell_text(raw))
            while len(nr) < v_cols:
                nr.append("")
            norm_visual.append(nr)

        # 执行填充（带质量门控 + 去重检查）
        fused_cells = 0
        rejected_count = 0
        for ri in range(min(s_rows, v_rows)):
            for ci in range(min(s_cols, v_cols)):
                if ri < len(s_grid) and ci < len(s_grid[ri]):
                    existing = s_grid[ri][ci].strip()
                    if not existing:  # 只填充空单元格
                        val = (norm_visual[ri][ci] if ri < len(norm_visual)
                               and ci < len(norm_visual[ri]) else "").strip()
                        if val and _is_valid_fusion_candidate(val):
                            s_grid[ri][ci] = val
                            fused_cells += 1
                        elif val:
                            rejected_count += 1

        if fused_cells > 0 or rejected_count > 0:
            total_fused += fused_cells
            logger.info(
                f"表格融合: {s_rows}×{s_cols} 结构树 + {v_rows}×{v_cols} 规则检测"
                f" → 接受 {fused_cells}, 拒绝 {rejected_count} (噪声过滤)"
            )

    if total_fused > 0:
        logger.info(f"融合完成: 共补充 {total_fused} 个空缺单元格")

    return struct_results


def extract_tables_from_struct_tree(doc, fuse_with_visual_flag: bool = True) -> Dict[int, list]:
    """
    从 PDF 的 StructTreeRoot 提取所有表格（仅适用于 Tagged PDF）。

    完整解析路径（类似 Word OOXML）：
        Catalog → StructTreeRoot → Part → Table → TBody → TR → TD → [Span/P]

    对于每个 TD 单元格:
        递归收集子树中的 ActualText + MCID 引用
        MCID 通过内容流解析映射到实际文本
        最终得到完整的单元格文本

    Args:
        doc: PyMuPDF Document 对象

    Returns:
        {页码(1-based): [StructTableResult, ...]} 映射
    """
    results: Dict[int, list] = {}

    if not is_tagged_pdf(doc):
        logger.debug("PDF 不是 Tagged PDF，跳过结构树表格提取")
        return results

    # Step 1: 定位 StructTreeRoot
    try:
        cat_obj = doc.xref_object(doc.pdf_catalog())
        str_match = re.search(r'/StructTreeRoot\s*(\d+)\s+\d+\s*R', cat_obj)
        if not str_match:
            logger.debug("Catalog 中未找到 StructTreeRoot")
            return results
        root_xref = int(str_match.group(1))
    except Exception as e:
        logger.warning(f"读取 StructTreeRoot 失败: {e}")
        return results

    root_node = _read_struct_node(doc, root_xref)
    if not root_node:
        return results

    # Step 2: 广度优先搜索所有 /Table 节点
    from collections import deque
    queue = deque(root_node.get('kids', []))
    visited: set = set()
    total_tables = 0

    while queue:
        xref = queue.popleft()
        if xref in visited or not isinstance(xref, int):
            continue
        visited.add(xref)

        node = _read_struct_node(doc, xref)
        if not node:
            continue

        # 非 Table 节点：将子节点加入搜索队列
        if node.get('tag') != 'Table':
            for kid in node.get('kids', []):
                if isinstance(kid, int):
                    queue.append(kid)
            continue

        # ====== 发现了一个 Table! ======
        total_tables += 1

        # Step 3: 找到 TBody/THead 容器 + 页面引用
        tbody_xref = None
        page_xref = None
        for kid in node.get('kids', []):
            child = _read_struct_node(doc, kid)
            if not child:
                continue
            if child.get('tag') in ('TBody', 'THead') and not tbody_xref:
                tbody_xref = kid
            if child.get('page_xref') and not page_xref:
                page_xref = child['page_xref']

        if not tbody_xref:
            logger.debug(f"Table xref={xref}: 无 TBody/THead，跳过")
            continue

        tbody = _read_struct_node(doc, tbody_xref)
        if not tbody or not tbody.get('tag'):
            continue
        if tbody.get('page_xref'):
            page_xref = tbody['page_xref']

        # Step 4: 构建该页的 MCID→文本映射
        page_num = _get_page_num(doc, page_xref) if page_xref else -1
        mcid_map: Dict[int, str] = {}
        if page_num > 0:
            try:
                mcid_map = _build_mcid_text_map(doc[page_num - 1])
            except Exception as e:
                logger.warning(f"第{page_num}页 MCID 映射构建失败: {e}")

        # Step 5: 遍历 TR → TD 构建文本网格
        grid_rows: List[List[str]] = []
        max_cols = 0

        for tr_kid in tbody.get('kids', []):
            tr = _read_struct_node(doc, tr_kid)
            if not tr or tr.get('tag') != 'TR':
                continue

            row_texts: List[str] = []
            for td_kid in tr.get('kids', []):
                td = _read_struct_node(doc, td_kid)
                if not td or td.get('tag') != 'TD':
                    continue

                cell_mcids, cell_actual_texts = _collect_td_content(
                    doc, td_kid, mcid_map)
                cell_text = _resolve_cell_text(cell_mcids, cell_actual_texts, mcid_map)
                row_texts.append(cell_text)

            if row_texts:
                grid_rows.append(row_texts)
                max_cols = max(max_cols, len(row_texts))

        # Step 6: 尺寸校验与标准化
        if len(grid_rows) < TABLE_MIN_ROWS or max_cols < TABLE_MIN_COLS:
            logger.debug(
                f"Table xref={xref}: {len(grid_rows)}×{max_cols}"
                f" < 最小要求 {TABLE_MIN_ROWS}×{TABLE_MIN_COLS}")
            continue

        for row in grid_rows:
            while len(row) < max_cols:
                row.append("")

        # 对结构树提取的网格做文本清理（去除 MCID 拼接产生的冗余空白）
        for row in grid_rows:
            for ci in range(len(row)):
                if row[ci]:
                    row[ci] = _clean_cell_text(row[ci])

        tbl_result = StructTableResult(_grid=grid_rows, _bbox=(0, 0, 0, 0))
        results.setdefault(page_num, []).append(tbl_result)

        logger.info(
            f"结构树表格 #{total_tables}: 第{page_num}页, "
            f"{len(grid_rows)}行 × {max_cols}列"
        )

    logger.info(
        f"结构树提取完成: {total_tables} 个 Table 节点, "
        f"{sum(len(t) for t in results.values())} 个满足条件"
    )

    # 融合：用视觉检测结果填充结构树中的空缺单元格
    if fuse_with_visual_flag and results:
        for page_num, tbl_list in results.items():
            if page_num > 0 and tbl_list:
                try:
                    _fuse_with_visual(tbl_list, doc[page_num - 1])
                except Exception as e:
                    logger.warning(f"第{page_num}页表格融合失败: {e}")

    return results


# ============================================================
# 阶段1：从 PyMuPDF 表格结果构建元数据
# ============================================================

def build_table_metadata(pymupdf_table, extracted_data: List[List[str]]) -> TableMetadata:
    """
    从 PyMuPDF 的 Table 对象构建结构化元数据。

    PyMuPDF API:
    - table.cells → list[(x0, y0, x1, y1)]，每个单元格的页面边界框（坐标点）
    - table.extract() → list[list[str]]，二维网格 [row][col]
      合并单元格的文本在 origin 位置，其余位置为 None 或空字符串

    合并检测策略（基于 extract() 网格分析）：
    - 水平合并：某行中非空单元格后紧跟连续空单元格
    - 垂直合并：某列中非空单元格下方连续行同列为空
    - 块合并：同时满足水平和垂直合并条件

    Args:
        pymupdf_table: PyMuPDF Table 对象
        extracted_data: Table.extract() 返回的二维文本数组

    Returns:
        TableMetadata 对象
    """
    if not extracted_data:
        return TableMetadata()

    # 从 extract() 获取网格维度（唯一可靠的数据源）
    num_rows = len(extracted_data)
    num_cols = max((len(row) for row in extracted_data), default=0) if extracted_data else 0

    if num_rows == 0 or num_cols == 0:
        return TableMetadata()

    # 标准化提取数据：None → ""，统一每行长度到 num_cols，清理冗余字符
    normalized = []
    for row in extracted_data:
        norm_row = []
        for v in row:
            raw = str(v).strip() if v is not None else ""
            norm_row.append(_clean_cell_text(raw))
        # 补齐尾部空列
        while len(norm_row) < num_cols:
            norm_row.append("")
        normalized.append(norm_row)

    def _is_empty(val: str) -> bool:
        """判断单元格值是否为空"""
        return val.strip() == ""

    # 构建 CellInfo 列表（每个逻辑位置一个 CellInfo）
    cell_infos: List[CellInfo] = []
    for ri in range(num_rows):
        for ci in range(num_cols):
            cell_infos.append(CellInfo(
                row=ri,
                col=ci,
                grid_span=1,
                row_span=1,
                text=normalized[ri][ci],
            ))

    # ---- 检测合并区域 ----
    merge_regions: List[MergeRegion] = []

    # 用 visited 集合避免重复检测已处理的合并区域
    visited: set = set()

    for ri in range(num_rows):
        ci = 0
        while ci < num_cols:
            if (ri, ci) in visited or _is_empty(normalized[ri][ci]):
                ci += 1
                continue

            val = normalized[ri][ci]

            # 计算水平跨度：向右找连续的空单元格
            h_span = 1
            for nc in range(ci + 1, num_cols):
                if (ri, nc) not in visited and _is_empty(normalized[ri][nc]):
                    h_span += 1
                else:
                    break

            # 计算垂直跨度：向下找连续的同列空单元格
            v_span = 1
            for nr in range(ri + 1, num_rows):
                if (nr, ci) not in visited and _is_empty(normalized[nr][ci]):
                    # 检查整行对应范围是否也为空（块合并验证）
                    all_empty_in_range = True
                    for cc in range(ci, ci + h_span):
                        if cc >= num_cols or not _is_empty(normalized[nr][cc]):
                            all_empty_in_range = False
                            break
                    if all_empty_in_range:
                        v_span += 1
                    else:
                        break
                else:
                    break

            # 只有实际有跨越时才记录合并
            if h_span > 1 or v_span > 1:
                mtype = 'block' if h_span > 1 and v_span > 1 else ('horizontal' if h_span > 1 else 'vertical')
                merge_regions.append(MergeRegion(
                    type=mtype,
                    origin_row=ri,
                    origin_col=ci,
                    row_span=v_span,
                    col_span=h_span,
                    text=val,
                ))
                # 更新对应 CellInfo 的 span 信息
                for info in cell_infos:
                    if info.row == ri and info.col == ci:
                        info.grid_span = h_span
                        info.row_span = v_span
                        break
                # 标记所有被合并覆盖的位置
                for dr in range(v_span):
                    for dc in range(h_span):
                        visited.add((ri + dr, ci + dc))

                ci += h_span
            else:
                ci += 1

    return TableMetadata(
        total_rows=num_rows,
        total_cols=num_cols,
        cells=cell_infos,
        merge_regions=merge_regions,
    )


# ============================================================
# 阶段2：网格填充
# ============================================================

def fill_grid(meta: TableMetadata) -> List[List[str]]:
    """
    根据元数据构建完整的二维网格。

    规则：
    - 合并起点：填入文本
    - 合并延续：行方向填 origin 值，列方向留空
    """
    grid = [['' for _ in range(meta.total_cols)] for _ in range(meta.total_rows)]

    # 建立 cell 映射
    cell_map = {(c.row, c.col): c for c in meta.cells}

    # 纵向合并追踪器: col → {text, remaining}
    vmerge_tracker: Dict[int, dict] = {}

    for ri in range(meta.total_rows):
        # 1. 先处理当前行被上方 vMerge 占据的列
        for col in list(vmerge_tracker.keys()):
            info = vmerge_tracker[col]
            if info['remaining'] > 0:
                if col < meta.total_cols:
                    grid[ri][col] = info['text']
                info['remaining'] -= 1
                if info['remaining'] <= 0:
                    del vmerge_tracker[col]

        # 2. 获取当前行的所有 cell
        row_cells = sorted(
            [c for c in meta.cells if c.row == ri],
            key=lambda c: c.col
        )

        for cell in row_cells:
            col = cell.col
            text = cell.text

            # 安全检查：col 超出网格宽度则跳过（不应发生，但防御性编程）
            if col >= meta.total_cols:
                continue

            if cell.row_span > 1:
                # 纵向合并起点：填入文本，注册 tracker
                grid[ri][col] = text
                vmerge_tracker[col] = {
                    'text': text,
                    'remaining': cell.row_span - 1,
                }
            elif cell.row_span == 1:
                # 普通单元格或横向合并起点
                grid[ri][col] = text

    return grid


# ============================================================
# 阶段3+4：行分类与结构检测
# ============================================================

def is_empty_row(row_data: list) -> bool:
    return all(v.strip() == "" for v in row_data)


def _row_has_h_merge(ri: int, meta: TableMetadata) -> bool:
    """判断某行是否有横向合并（gridSpan > 1）"""
    for c in meta.cells:
        if c.row == ri and c.grid_span > 1:
            return True
    return False


def _is_title_row(row_data: list, total_cols: int, meta: TableMetadata, ri: int) -> bool:
    """检测标题行：只有一个非空值且跨越 >= 50% 列"""
    non_empty = [v for v in row_data if v.strip()]
    if len(non_empty) != 1:
        return False
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
    2. 合并区域之外没有太多非空数据
    3. 后续行是叶子表头或另一个分组表头
    """
    if not _row_has_h_merge(ri, meta):
        return False

    non_merge_non_empty = 0
    for c in meta.cells:
        if c.row != ri:
            continue
        in_merge_span = False
        if c.grid_span > 1:
            in_merge_span = True
        elif c.row_span > 1:
            in_merge_span = True
        if not in_merge_span and c.text.strip():
            non_merge_non_empty += 1

    if non_merge_non_empty >= 2:
        return False

    row_pos = None
    for i, (r_idx, _) in enumerate(rows):
        if r_idx == ri:
            row_pos = i
            break
    if row_pos is None:
        return False

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
    将多行表头展平为单行。

    例: "基础工商信息" + "公司名称" → "基础工商信息/公司名称"
    """
    flattened = list(leaf_data)

    if first_col_value is not None and flattened[0].strip() == "":
        flattened[0] = first_col_value

    if not group_rows:
        return flattened

    h_merges = [m for m in meta.merge_regions if m.type == 'horizontal']

    for col_idx in range(meta.total_cols):
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

def table_to_markdown(pymupdf_table, table_index: int = 0) -> dict:
    """
    将 PyMuPDF Table 对象转换为 Markdown。

    简化版表头策略：
    - 直接以第一个非空、非标题行作为表头行
    - 如果该行存在横向合并（gridSpan > 1），视为"合并混乱"，使用字母列名 (列A, 列B...)，
      原始内容落入数据区首行
    - 否则直接使用该行内容作为表头

    Args:
        pymupdf_table: PyMuPDF Table 对象
        table_index: 表格在文档中的顺序（0-based）

    Returns:
        {
            "markdown": str,
            "title": str,
            "form_fields": list,
            "signing_info": list,
            "row_count": int,
            "col_count": int,
            "bbox": tuple,
        }
    """
    lines = []

    # 阶段1：构建元数据
    extracted_data = pymupdf_table.extract()
    meta = build_table_metadata(pymupdf_table, extracted_data)

    if meta.total_rows == 0 or meta.total_cols == 0:
        return {
            "markdown": "*（空表格）*\n",
            "title": "", "form_fields": [], "signing_info": [],
            "row_count": 0, "col_count": 0,
            "bbox": tuple(pymupdf_table.bbox) if hasattr(pymupdf_table, 'bbox') else (),
        }

    # 阶段2：填充网格
    grid = fill_grid(meta)
    rows = [(ri, grid[ri]) for ri in range(meta.total_rows)]

    title_texts = []
    signing_metadata = []

    # ---- 简化策略：直接取第一非空非标题行作为表头 ----
    header_row_pos = None

    for pos, (ri, row_data) in enumerate(rows):
        if is_empty_row(row_data):
            continue
        if _is_title_row(row_data, meta.total_cols, meta, ri):
            non_empty = [v for v in row_data if v.strip()]
            if non_empty:
                title_texts.append(non_empty[0])
            continue
        # 第一个非空、非标题行 → 作为表头行
        header_row_pos = pos
        break

    # 没有找到合适的表头行（可能全是标题/空行）
    if header_row_pos is None:
        header = [_col_letter(c + 1) for c in range(meta.total_cols)]
        data_rows = []
        for t in title_texts:
            lines.append(f"**{t.strip()}**")
            lines.append("")
        _output_table(lines, header, data_rows, meta.total_cols)
        return {
            "markdown": "\n".join(lines),
            "title": title_texts[0] if title_texts else "",
            "form_fields": [],
            "signing_info": signing_metadata,
            "row_count": meta.total_rows,
            "col_count": meta.total_cols,
            "bbox": tuple(pymupdf_table.bbox) if hasattr(pymupdf_table, 'bbox') else (),
        }

    # 获取表头行的原始数据和行索引
    header_ri = rows[header_row_pos][0]
    header_raw_data = list(rows[header_row_pos][1])

    # 判断表头行是否有横向合并（合并混乱检测）
    has_complex_merge = _row_has_h_merge(header_ri, meta)

    if has_complex_merge:
        # 有横向合并 → 字母表头，原始内容落入数据区首行
        header = [_col_letter(c + 1) for c in range(meta.total_cols)]
    else:
        # 无横向合并 → 直接用第一行内容作表头
        header = list(header_raw_data)

    # ---- 收集数据行：表头行之后的所有非空行（跳过标题和签章） ----
    data_positions = []
    for pos in range(header_row_pos + 1, len(rows)):
        ri, row_data = rows[pos]
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
        data_positions.append(pos)

    # 合并混乱时，把原始表头行内容插入到数据行最前面
    data_rows = [rows[p][1] for p in data_positions]
    if has_complex_merge and header_raw_data:
        data_rows = [header_raw_data] + data_rows

    # ---- 输出 ----
    for t in title_texts:
        lines.append(f"**{t.strip()}**")
        lines.append("")

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
        "form_fields": [],
        "signing_info": signing_metadata,
        "row_count": meta.total_rows,
        "col_count": meta.total_cols,
        "bbox": tuple(pymupdf_table.bbox) if hasattr(pymupdf_table, 'bbox') else (),
    }
