"""B 类参数计算编排层。"""

from __future__ import annotations

from typing import Any, Dict

from ..common.oecd_io_table import IoTableData
from .e_j import compute_e_j
from .export_value_j import compute_export_value_j
from .external_inputs import ExternalBInputs
from .factor_params import compute_factor_params
from .p_j import compute_p_j_exogenous
from .p_j_o import compute_p_j_o_exogenous
from .price_system import solve_price_system_endogenous


def compute_b_parameters(
    total_table: IoTableData,
    domestic_table: IoTableData,
    external: ExternalBInputs,
    consumption_columns: list[str],
    price_mode: str = "endogenous",
    solver_max_iterations: int = 400,
    solver_tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """计算 B 类参数（分文件函数编排）。"""

    factor_params = compute_factor_params(external)
    diagnostics: Dict[str, Any] = {"requested_price_mode": price_mode}

    if price_mode == "endogenous":
        try:
            p_j, p_j_o, export_value_j, solver_info = solve_price_system_endogenous(
                total_table=total_table,
                domestic_table=domestic_table,
                external=external,
                consumption_columns=consumption_columns,
                solver_max_iterations=solver_max_iterations,
                solver_tolerance=solver_tolerance,
            )
            diagnostics["price_mode_used"] = "endogenous"
            diagnostics["solver_info"] = solver_info
            diagnostics["fallback_reason"] = None
        except Exception as exc:  # pragma: no cover - 环境相关
            if float(external.fallback_to_exogenous_if_solver_fails) < 0.5:
                raise
            p_j = compute_p_j_exogenous(total_table, external)
            p_j_o = compute_p_j_o_exogenous(total_table, external, p_j)
            export_value_j = compute_export_value_j(total_table, external)
            diagnostics["price_mode_used"] = "exogenous_fallback"
            diagnostics["solver_info"] = None
            diagnostics["fallback_reason"] = str(exc)
    else:
        p_j = compute_p_j_exogenous(total_table, external)
        p_j_o = compute_p_j_o_exogenous(total_table, external, p_j)
        export_value_j = compute_export_value_j(total_table, external)
        diagnostics["price_mode_used"] = "exogenous"
        diagnostics["solver_info"] = None
        diagnostics["fallback_reason"] = None

    e_j = compute_e_j(export_value_j=export_value_j, p_j=p_j)
    return {
        "factor_params": factor_params,
        "P_j": p_j,
        "P_j_O": p_j_o,
        "E_j": e_j,
        "ExportValue_j": export_value_j,
        "diagnostics": diagnostics,
    }

