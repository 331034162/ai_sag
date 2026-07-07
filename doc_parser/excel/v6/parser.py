"""
Excel V6 CSV 格式解析器
========================

提供 ExcelParser 类和便捷函数，将 Excel 文件解析为 CSV 文本。

用法：
    from ai_sag.doc_parser.excel.v6 import parse_excel

    result = parse_excel("/path/to/file.xlsx")
    print(result.to_csv_text())
"""

import logging
from pathlib import Path
from typing import Optional

import openpyxl

from ai_sag.doc_parser.excel.v6.config import INCLUDE_HIDDEN, INCLUDE_EMPTY_CELLS
from ai_sag.doc_parser.excel.v6.models import SheetCSV, ExcelCSV
from ai_sag.doc_parser.excel.v6.converter import sheet_to_csv

logger = logging.getLogger(__name__)


class ExcelParser:
    """
    Excel CSV 格式解析器。

    将 Excel 文件解析为 CSV 文本，每个 Sheet 输出为逗号分隔的纯文本。

    Args:
        output_dir: 输出目录，传值则保存到磁盘
        include_hidden: 是否包含隐藏行列
        include_empty: 是否包含空行
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        include_hidden: bool = INCLUDE_HIDDEN,
        include_empty: bool = INCLUDE_EMPTY_CELLS,
    ):
        self.output_dir = output_dir
        self.include_hidden = include_hidden
        self.include_empty = include_empty

    def parse(self, file_path: str) -> ExcelCSV:
        """解析单个 Excel 文件为 CSV 文本"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if path.suffix.lower() not in (".xlsx", ".xls"):
            raise ValueError(f"不支持的文件格式: {path.suffix}，仅支持 .xlsx / .xls")

        # 动态设置配置
        from ai_sag.doc_parser.excel.v6 import config as cfg
        original_hidden = cfg.INCLUDE_HIDDEN
        original_empty = cfg.INCLUDE_EMPTY_CELLS
        cfg.INCLUDE_HIDDEN = self.include_hidden
        cfg.INCLUDE_EMPTY_CELLS = self.include_empty

        try:
            # data_only=True 获取公式计算值
            wb = openpyxl.load_workbook(file_path, data_only=True)

            sheets = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                data = sheet_to_csv(
                    ws, sheet_title=sheet_name,
                    include_hidden=self.include_hidden,
                    include_empty=self.include_empty,
                )
                sc = SheetCSV(
                    sheet_name=data["sheet_name"],
                    csv_text=data["csv_text"],
                    row_count=data["row_count"],
                    col_count=data["col_count"],
                )
                sheets.append(sc)

            result = ExcelCSV(
                file_path=str(path),
                file_name=path.name,
                total_sheets=len(sheets),
                sheets=sheets,
                metadata={
                    "include_hidden": self.include_hidden,
                    "include_empty": self.include_empty,
                    "sheet_names": wb.sheetnames,
                },
            )

            # 保存到磁盘
            if self.output_dir:
                from ai_sag.doc_parser.excel.v6.formatter import save_csv_text
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                save_csv_text(result, str(out_dir / f"{path.stem}.csv"))

            return result

        finally:
            cfg.INCLUDE_HIDDEN = original_hidden
            cfg.INCLUDE_EMPTY_CELLS = original_empty

    def parse_directory(self, dir_path: str, recursive: bool = False) -> list[ExcelCSV]:
        """批量解析目录下的所有 Excel 文件"""
        dp = Path(dir_path)
        if not dp.is_dir():
            raise NotADirectoryError(f"不是目录: {dir_path}")

        pattern = "**/*.xlsx" if recursive else "*.xlsx"
        files = list(dp.glob(pattern))
        if not recursive:
            files += list(dp.glob("*.xls"))
        else:
            files += list(dp.glob("**/*.xls"))

        files = [f for f in files if not f.name.startswith("~$")]

        results = []
        for f in sorted(files):
            try:
                logger.info(f"解析: {f.name}")
                result = self.parse(str(f))
                results.append(result)
            except Exception as e:
                logger.error(f"解析失败 [{f.name}]: {e}")

        return results


# ============================================================
# 便捷函数
# ============================================================

def parse_excel(
    file_path: str,
    output_dir: Optional[str] = None,
    include_hidden: bool = INCLUDE_HIDDEN,
    include_empty: bool = INCLUDE_EMPTY_CELLS,
) -> ExcelCSV:
    """一键解析 Excel 文档为 CSV 文本"""
    parser = ExcelParser(
        output_dir=output_dir,
        include_hidden=include_hidden,
        include_empty=include_empty,
    )
    return parser.parse(file_path)


def parse_directory(
    dir_path: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    include_hidden: bool = INCLUDE_HIDDEN,
    include_empty: bool = INCLUDE_EMPTY_CELLS,
) -> list[ExcelCSV]:
    """一键批量解析目录下的所有 Excel 文件"""
    parser = ExcelParser(
        output_dir=output_dir,
        include_hidden=include_hidden,
        include_empty=include_empty,
    )
    return parser.parse_directory(dir_path, recursive=recursive)
