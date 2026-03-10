"""Common utilities shared across parameter categories."""

from .oecd_io_table import CountryYearTables, IoTableData, load_country_year_tables

__all__ = [
    "IoTableData",
    "CountryYearTables",
    "load_country_year_tables",
]

