"""Category-organized IO parameter calculators.

Layout:
- common: shared table loaders/constants
- category_a: A-class parameters directly identified from IO tables
- category_b: B-class parameters requiring IO + external inputs
- category_c: C-class exogenous/scenario path parameters
- category_d: placeholders for future work
"""

from .common import IoTableData, load_country_year_tables

__all__ = [
    "IoTableData",
    "load_country_year_tables",
]
