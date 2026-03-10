"""Constants for OECD IO parameter extraction."""

from pathlib import Path

DEFAULT_DATA_ROOT = Path("中美投入产出表数据（附表格阅读说明）")

COUNTRY_DIR = {
    "USA": Path("美国/OECD美国投入产出数据"),
    "CHN": Path("中国/OECD中国投入产出数据"),
}

TOTAL_SUBDIR = Path("投入产出，行业-行业，total口径")
DOMESTIC_SUBDIR = Path("投入产出（含进口行），行业-行业，Domestic口径")

TOTAL_PREFIX = "TTL_"
DOMESTIC_PREFIX = "DOM_"
OUTPUT_ROW_LABEL = "OUTPUT"

DEFAULT_CONSUMPTION_COLUMNS = ("HFCE", "NPISH", "GGFC")

