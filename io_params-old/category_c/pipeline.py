"""C 类参数计算编排层。"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..common.oecd_io_table import IoTableData
from .a_i import compute_a_i
from .export_t import compute_export_t
from .external_inputs import ExternalCInputs
from .gamma_cj import compute_gamma_cj
from .gamma_ij import compute_gamma_ij
from .p_t_o import compute_p_t_o
from .rho_cj import compute_rho_cj
from .shared import build_sector_series


def _build_export_base_j(total_table: IoTableData, external: ExternalCInputs) -> pd.Series:
    """构造 Export_t 的基线向量。"""

    return build_sector_series(
        total_table.sector_codes,
        default_value=external.export_base_default,
        overrides=external.export_base_by_sector,
        non_negative=True,
        name="Export_base_j",
        index_name="sector_j",
    )


def _build_p_t_o_base_j(total_table: IoTableData, external: ExternalCInputs) -> pd.Series:
    """构造 p_t^O 的基线向量。"""

    return build_sector_series(
        total_table.sector_codes,
        default_value=external.p_t_o_base_default,
        overrides=external.p_t_o_base_by_sector,
        positive=True,
        name="P_t_O_base_j",
        index_name="sector_j",
    )


def compute_c_parameters(
    total_table: IoTableData,
    domestic_table: IoTableData,
    external: ExternalCInputs,
    consumption_columns: list[str],
    *,
    export_base_j: pd.Series | None = None,
    p_t_o_base_j: pd.Series | None = None,
    periods: int | None = None,
) -> Dict[str, Any]:
    """计算 C 类参数（按文件拆分后的统一编排入口）。"""

    a_i = compute_a_i(total_table, external)
    gamma_ij = compute_gamma_ij(total_table, domestic_table, external)
    gamma_cj = compute_gamma_cj(total_table, domestic_table, external, consumption_columns)
    rho_cj = compute_rho_cj(total_table, external)

    export_base = export_base_j if export_base_j is not None else _build_export_base_j(total_table, external)
    p_t_o_base = p_t_o_base_j if p_t_o_base_j is not None else _build_p_t_o_base_j(total_table, external)

    export_t = compute_export_t(export_base, external, periods=periods)
    p_t_o = compute_p_t_o(p_t_o_base, external, periods=periods)

    diagnostics: Dict[str, Any] = {
        "horizon_periods": int(periods if periods is not None else external.horizon_periods),
        "export_base_source": "argument" if export_base_j is not None else "external_defaults",
        "p_t_o_base_source": "argument" if p_t_o_base_j is not None else "external_defaults",
    }

    return {
        "A_i": a_i,
        "gamma_ij": gamma_ij,
        "gamma_cj": gamma_cj,
        "rho_cj": rho_cj,
        "Export_base_j": export_base,
        "P_t_O_base_j": p_t_o_base,
        "Export_t": export_t,
        "P_t_O": p_t_o,
        "diagnostics": diagnostics,
    }

