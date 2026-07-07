"""
表格识别数据模型
===============
TableCell / TableRecognitionResult
"""

from dataclasses import dataclass, field


@dataclass
class TableCell:
    """单个单元格"""
    row: int = 0
    col: int = 0
    text: str = ""
    row_span: int = 1
    col_span: int = 1
    confidence: float = 0.0


@dataclass
class TableRecognitionResult:
    """单张图片的表格识别结果"""
    source: str = ""
    cells: list[TableCell] = field(default_factory=list)
    total_rows: int = 0
    total_cols: int = 0
    html_table: str = ""
    raw_grid: list = field(default_factory=list)
    confidence: float = 0.0
    bbox: tuple = ()

    @property
    def grid(self) -> list[list[str]]:
        """将 cells 转为二维字符串网格"""
        if not self.cells or self.total_rows == 0 or self.total_cols == 0:
            return []
        grid = [["" for _ in range(self.total_cols)] for _ in range(self.total_rows)]
        for cell in self.cells:
            r, c = cell.row, cell.col
            if 0 <= r < self.total_rows and 0 <= c < self.total_cols:
                grid[r][c] = cell.text
        return grid


__all__ = ["TableCell", "TableRecognitionResult"]
