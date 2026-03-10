"""B 类价格系统联立求解工具。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..category_a.alpha_ij import compute_alpha_ij
from ..category_a.beta_j import compute_beta_j
from ..category_a.theta_cj import compute_theta_cj
from ..category_a.theta_ij import compute_theta_ij
from ..common.oecd_io_table import IoTableData
from .export_value_j import compute_export_value_j
from .external_inputs import ExternalBInputs
from .p_j import compute_p_j_from_endogenous_solution
from .p_j_o import compute_p_j_o_from_endogenous_solution
from .shared import safe_positive, sanitize_rho


def build_equilibrium_raw_params(
    total_table: IoTableData,
    domestic_table: IoTableData,
    external: ExternalBInputs,
    consumption_columns: list[str],
) -> tuple[Dict[str, Any], pd.Series]:
    """构造 `solve_initial_equilibrium` 所需输入。"""

    alpha_df = compute_alpha_ij(total_table)
    theta_df = compute_theta_ij(total_table, domestic_table, fill_on_zero_total=1.0)
    beta_s = compute_beta_j(total_table, consumption_columns)
    theta_c_s = compute_theta_cj(total_table, domestic_table, consumption_columns, fill_on_zero_total=1.0)
    export_value_j = compute_export_value_j(total_table, external)

    sectors = list(total_table.sector_codes)
    n = len(sectors)

    alpha = alpha_df.reindex(index=sectors, columns=sectors).fillna(0.0).to_numpy(dtype=float)
    theta = theta_df.reindex(index=sectors, columns=sectors).fillna(1.0).to_numpy(dtype=float)
    beta = beta_s.reindex(sectors).fillna(0.0).to_numpy(dtype=float)
    theta_c = theta_c_s.reindex(sectors).fillna(1.0).to_numpy(dtype=float)

    # 数值稳定：保持增加值份额为正
    for i in range(n):
        row_sum = float(alpha[i].sum())
        if row_sum >= 0.98 and row_sum > 0.0:
            alpha[i] *= 0.98 / row_sum

    gamma = np.clip(theta, 1e-6, 1 - 1e-6)
    gamma_c = np.clip(theta_c, 1e-6, 1 - 1e-6)

    rho_val = sanitize_rho(external.rho_default)
    rho = np.full((n, n), rho_val, dtype=float)
    rho_c = np.full(n, rho_val, dtype=float)

    a_default = safe_positive(external.tfp_A_default, 1.0)
    a_i = np.full(n, a_default, dtype=float)

    import_cost_default = safe_positive(external.import_cost_default, 1.0)
    import_cost = np.array(
        [safe_positive(external.import_prices_by_sector.get(s, import_cost_default), import_cost_default) for s in sectors],
        dtype=float,
    )

    export_i = np.maximum(export_value_j.to_numpy(dtype=float), 0.0)

    if float(external.use_all_sectors_tradable) >= 0.5:
        tradable = list(range(n))
    else:
        import_share = 1.0 - theta.mean(axis=0)
        tradable = [j for j in range(n) if import_share[j] > 1e-8]
        if not tradable:
            tradable = list(range(n))

    block = {
        "alpha_ij": alpha,
        "gamma_ij": gamma,
        "rho_ij": rho,
        "beta_j": beta,
        "A_i": a_i,
        "Export_i": export_i,
        "gamma_cj": gamma_c,
        "rho_cj": rho_c,
        "import_cost": import_cost,
    }
    raw_params = {"H": block, "F": block, "tradable_sectors": tradable}
    return raw_params, export_value_j


def solve_price_system_endogenous(
    total_table: IoTableData,
    domestic_table: IoTableData,
    external: ExternalBInputs,
    consumption_columns: list[str],
    solver_max_iterations: int,
    solver_tolerance: float,
) -> tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """联立求解 `P_j` 与 `P_j^O`。"""

    try:
        # 延迟导入，避免环境缺失 torch 时导入即失败
        from eco_simu.model import solve_initial_equilibrium
    except Exception as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(f"导入 eco_simu 求解器失败: {exc}") from exc

    raw_params, export_value_j = build_equilibrium_raw_params(
        total_table=total_table,
        domestic_table=domestic_table,
        external=external,
        consumption_columns=consumption_columns,
    )

    result = solve_initial_equilibrium(
        raw_params,
        max_iterations=int(max(10, solver_max_iterations)),
        tolerance=float(max(1e-12, solver_tolerance)),
    )
    info = result.get("convergence_info", {})
    if not bool(info.get("converged", False)):
        raise RuntimeError(
            f"联立求解未收敛: iterations={info.get('iterations')}, "
            f"residual={info.get('final_residual')}, msg={info.get('solver_message')}"
        )

    p_j = compute_p_j_from_endogenous_solution(result, total_table)
    p_j_o = compute_p_j_o_from_endogenous_solution(result, total_table)
    return p_j, p_j_o, export_value_j, info

