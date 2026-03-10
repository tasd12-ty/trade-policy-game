"""兼容适配层。

提供从 io_params 管道 CSV 输出转换为 eco_model_v2 格式的桥接函数：
- from_io_params: 单国（中美投入产出数据）
- from_io_params_two_country: 两国版本

依赖：types.py, model_defaults.py
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .types import CountryParams, TwoCountryParams
from . import model_defaults


# ---- io_params CSV bridge ----

def _read_csv_matrix(path: str) -> np.ndarray:
    """读取 io_params 输出的 CSV 矩阵（含行/列索引）。"""
    import pandas as pd
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(float)


def _read_csv_vector(path: str) -> np.ndarray:
    """读取 io_params 输出的 CSV 向量（Series → DataFrame 格式）。"""
    import pandas as pd
    df = pd.read_csv(path, index_col=0)
    return df.iloc[:, 0].values.astype(float)


def from_io_params(
    cat_a_dir: str,
    cat_b_dir: str,
    cat_c_dir: str,
    Ml: Optional[int] = None,
    rho_ij: Optional[float] = None,
    M_factors: int = 2,
) -> CountryParams:
    """从 io_params 管道输出构造 CountryParams。

    读取 OECD 投入产出数据经 io_params 三类管道处理后的 CSV 文件，
    转换为 eco_model_v2 的 CountryParams 格式。

    参数映射:
        cat_a/alpha_ij.csv   → alpha[:, :Nl]  中间投入弹性
        cat_a/beta_j.csv     → beta            消费预算权重
        cat_a/theta_cj.csv   → gamma_cons      消费端国内份额
        cat_b/factor_params  → L               要素禀赋 [L, K]
        cat_b/P_j_O.csv      → import_cost     进口品到岸价
        cat_b/E_j.csv        → exports[:Nl]    基期出口量
        cat_c/A_i.csv        → A               全要素生产率
        cat_c/gamma_ij.csv   → gamma           Armington 生产端权重
        cat_c/rho_cj.csv     → rho_cons        消费端 CES 弹性
        model_defaults       → rho             生产端 CES（默认 Cobb-Douglas）

    参数：
        cat_a_dir: Category A 输出目录路径
        cat_b_dir: Category B 输出目录路径
        cat_c_dir: Category C 输出目录路径
        Ml:        非贸易部门数（默认从 model_defaults.ML_DEFAULT）
        rho_ij:    生产端 Armington ρ（默认从 model_defaults.RHO_IJ_DEFAULT）
        M_factors: 要素种类数（默认 2: 劳动+资本）

    返回：
        CountryParams
    """
    if Ml is None:
        Ml = model_defaults.ML_DEFAULT
    if rho_ij is None:
        rho_ij = model_defaults.RHO_IJ_DEFAULT

    # ---- Category A ----
    alpha_ij = _read_csv_matrix(os.path.join(cat_a_dir, "alpha_ij.csv"))
    beta = _read_csv_vector(os.path.join(cat_a_dir, "beta_j.csv"))
    gamma_cons = _read_csv_vector(os.path.join(cat_a_dir, "theta_cj.csv"))

    Nl = alpha_ij.shape[0]

    # ---- Category B ----
    import_cost = _read_csv_vector(os.path.join(cat_b_dir, "P_j_O.csv"))
    exports_product = _read_csv_vector(os.path.join(cat_b_dir, "E_j.csv"))

    # 要素禀赋
    factor_df_path = os.path.join(cat_b_dir, "factor_params.csv")
    import pandas as pd
    factor_df = pd.read_csv(factor_df_path, index_col=0)
    factor_vals = factor_df.iloc[:, 0]

    # 提取 L, K（按 io_params 格式：index = [w, r, K, L]）
    L_val = float(factor_vals.get("L", factor_vals.iloc[-1]))
    K_val = float(factor_vals.get("K", factor_vals.iloc[-2]))
    if M_factors == 2:
        L = np.array([L_val, K_val])
    elif M_factors == 1:
        L = np.array([L_val])
    else:
        L = np.array([L_val, K_val][:M_factors])

    # ---- Category C ----
    A = _read_csv_vector(os.path.join(cat_c_dir, "A_i.csv"))
    gamma = _read_csv_matrix(os.path.join(cat_c_dir, "gamma_ij.csv"))
    rho_cons = _read_csv_vector(os.path.join(cat_c_dir, "rho_cj.csv"))

    # ---- 构造完整 alpha (Nl, Nl+M) ----
    # 要素列：按 residual 规则 (1 − Σ_j α_{ij}) / M
    row_sum = alpha_ij.sum(axis=1)
    remaining = np.maximum(1.0 - row_sum, 0.01)
    if model_defaults.FACTOR_ALPHA_RULE == "residual":
        alpha_factor = np.tile(
            (remaining / max(M_factors, 1))[:, np.newaxis],
            (1, M_factors),
        )
    else:
        alpha_factor = np.full((Nl, M_factors), 1.0 / max(M_factors, 1))
    alpha = np.concatenate([alpha_ij, alpha_factor], axis=1)

    # ---- rho: 生产端 CES 弹性矩阵 (Nl, Nl) ----
    rho = np.full((Nl, Nl), rho_ij, dtype=float)

    # ---- exports: (Nl+M,) ----
    exports = np.concatenate([exports_product, np.zeros(M_factors)])

    # ---- gamma_cons / rho_cons 截断到 Nl，填充 NaN ----
    gamma_cons = gamma_cons[:Nl]
    gamma_cons = np.where(np.isfinite(gamma_cons), gamma_cons, 0.5)
    rho_cons = rho_cons[:Nl]
    rho_cons = np.where(np.isfinite(rho_cons), rho_cons, 0.0)

    return CountryParams(
        alpha=alpha,
        gamma=gamma[:Nl, :Nl],
        rho=rho,
        beta=beta[:Nl],
        A=A[:Nl],
        exports=exports,
        gamma_cons=gamma_cons,
        rho_cons=rho_cons,
        import_cost=import_cost[:Nl],
        L=L,
        Ml=Ml,
        M_factors=M_factors,
    )


def from_io_params_two_country(
    home_dirs: Tuple[str, str, str],
    foreign_dirs: Tuple[str, str, str],
    Ml: Optional[int] = None,
    rho_ij: Optional[float] = None,
    M_factors: int = 2,
) -> TwoCountryParams:
    """从两国的 io_params 管道输出构造 TwoCountryParams。

    参数：
        home_dirs:    (cat_a_dir, cat_b_dir, cat_c_dir) 本国数据目录
        foreign_dirs: (cat_a_dir, cat_b_dir, cat_c_dir) 外国数据目录
        Ml:           非贸易部门数
        rho_ij:       生产端 Armington ρ
        M_factors:    要素种类数

    返回：
        TwoCountryParams

    示例（中美数据）：
        params = from_io_params_two_country(
            home_dirs=(
                "io_params/outputs/category_a/CHN2022",
                "io_params/outputs/category_b/CHN2022",
                "io_params/outputs/category_c/CHN2022",
            ),
            foreign_dirs=(
                "io_params/outputs/category_a/USA2022",
                "io_params/outputs/category_b/USA2022",
                "io_params/outputs/category_c/USA2022",
            ),
        )
    """
    home = from_io_params(
        home_dirs[0], home_dirs[1], home_dirs[2],
        Ml=Ml, rho_ij=rho_ij, M_factors=M_factors,
    )
    foreign = from_io_params(
        foreign_dirs[0], foreign_dirs[1], foreign_dirs[2],
        Ml=Ml, rho_ij=rho_ij, M_factors=M_factors,
    )
    return TwoCountryParams(home=home, foreign=foreign)
