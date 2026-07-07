"""
表格格式化输出
=============
将 TableRecognitionResult 转为 Markdown 等格式。
"""

from ai_sag.doc_parser.image.table.models import TableRecognitionResult


def table_to_markdown(tbl: TableRecognitionResult) -> str:
    """将 TableRecognitionResult 转为 Markdown 表格字符串

    特性：
    - 自动清理单元格中的管道符和换行
    - 空单元格用空格占位，保持表格结构完整
    - 首行作为表头
    """
    grid = tbl.grid
    if not grid:
        return ""

    # 清理单元格文本
    cleaned_grid = []
    for row in grid:
        cleaned_row = [_clean_md_cell(cell) for cell in row]
        cleaned_grid.append(cleaned_row)

    # 计算每列最大宽度（用于对齐，可选）
    n_cols = max(len(row) for row in cleaned_grid) if cleaned_grid else 0
    if n_cols == 0:
        return ""

    # 补齐列数不足的行
    for row in cleaned_grid:
        while len(row) < n_cols:
            row.append("")

    lines: list[str] = []

    # 表头
    header = cleaned_grid[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")

    # 数据行
    for row in cleaned_grid[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _clean_md_cell(text: str) -> str:
    """清理单元格文本，使其适合 Markdown 表格"""
    if not text:
        return ""
    t = text.strip()
    # 转义管道符
    t = t.replace("|", "\\|")
    # 换行替换为空格
    t = t.replace("\n", " ").replace("\r", "")
    # 压缩多余空格
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def tables_to_text(tbl: TableRecognitionResult) -> str:
    """将表格转为纯文本格式（每行用制表符分隔列）"""
    grid = tbl.grid
    if not grid:
        return ""
    lines = []
    for row in grid:
        lines.append("\t".join(cell.strip() for cell in row))
    return "\n".join(lines)
