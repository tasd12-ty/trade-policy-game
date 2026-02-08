"""向后兼容适配层。

将旧版 project_refactor/project_model 的参数格式转换为 eco_model_v2 格式。
也提供从 grad_op 格式的转换。

依赖：types.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .types import CountryParams, TwoCountryParams


def from_project_refactor(
    old_params: Dict[str, Any],
    country: str = "H",
) -> CountryParams:
    """从 project_refactor 格式转换。

    旧格式字段：
    - alpha: (Nl, Nl) — 无要素列
    - theta/gamma: (Nl, Nl)
    - rho: (Nl, Nl)
    - beta: (Nl,)
    - A: (Nl,)
    - exports: (Nl,)
    - imp_cost/import_cost: (Nl,)
    - Ml: int
    - L: float 或 (M,) — 旧版可能只有标量劳动

    新格式扩展：
    - alpha: (Nl, Nl+M) — 追加要素列
    - exports: (Nl+M,) — 追加零
    - L: (M,)
    """
    alpha_old = np.asarray(old_params["alpha"], dtype=float)
    Nl = alpha_old.shape[0]
    Ml = int(old_params.get("Ml", 0))

    # 要素数：默认 1（劳动）
    L_raw = old_params.get("L", 10.0)
    if np.isscalar(L_raw):
        L = np.array([float(L_raw)])
    else:
        L = np.asarray(L_raw, dtype=float)
    M = len(L)

    # 扩展 alpha
    row_sum = alpha_old.sum(axis=1)
    remaining = np.maximum(1.0 - row_sum, 0.01)
    alpha_factor = remaining[:, np.newaxis] / max(M, 1)
    alpha = np.concatenate([alpha_old, alpha_factor], axis=1)

    # gamma, rho
    gamma = np.asarray(old_params.get("gamma", old_params.get("theta", np.ones((Nl, Nl)))), dtype=float)
    rho = np.asarray(old_params.get("rho", np.zeros((Nl, Nl))), dtype=float)

    # beta
    beta = np.asarray(old_params.get("beta", np.ones(Nl) / Nl), dtype=float)

    # A
    A = np.asarray(old_params.get("A", np.ones(Nl)), dtype=float)

    # exports
    exports_old = np.asarray(old_params.get("exports", np.zeros(Nl)), dtype=float)
    exports = np.concatenate([exports_old, np.zeros(M)])

    # import_cost
    import_cost = np.asarray(
        old_params.get("import_cost", old_params.get("imp_cost", np.ones(Nl) * 1.1)),
        dtype=float,
    )

    # gamma_cons, rho_cons
    gamma_cons = np.asarray(old_params.get("gamma_cons", gamma[0, :].copy()), dtype=float)
    rho_cons = np.asarray(old_params.get("rho_cons", rho[0, :].copy()), dtype=float)

    return CountryParams(
        alpha=alpha, gamma=gamma, rho=rho,
        beta=beta, A=A, exports=exports,
        gamma_cons=gamma_cons, rho_cons=rho_cons,
        import_cost=import_cost, L=L,
        Ml=Ml, M_factors=M,
    )


def from_grad_op(
    params_dict: Dict[str, Any],
) -> TwoCountryParams:
    """从 grad_op 格式转换。

    grad_op 使用 PyTorch tensor，字段名有些不同。
    此函数将 tensor → numpy 并重命名字段。
    """
    def _to_np(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    home_dict = params_dict.get("home", params_dict.get("H", {}))
    foreign_dict = params_dict.get("foreign", params_dict.get("F", {}))

    countries = []
    for cd in [home_dict, foreign_dict]:
        alpha = _to_np(cd["alpha"])
        gamma = _to_np(cd.get("gamma", cd.get("theta")))
        rho = _to_np(cd["rho"])
        beta = _to_np(cd["beta"])
        A = _to_np(cd["A"])
        L = _to_np(cd.get("L", [10.0]))
        Ml = int(cd.get("Ml", cd.get("n_nontradable", 0)))
        M = len(L)
        Nl = alpha.shape[0]

        # alpha 可能已经是 (Nl, Nl+M) 或 (Nl, Nl)
        if alpha.shape[1] == Nl:
            row_sum = alpha.sum(axis=1)
            remaining = np.maximum(1.0 - row_sum, 0.01)
            alpha = np.concatenate([alpha, remaining[:, np.newaxis] / M], axis=1)

        exports_raw = _to_np(cd.get("exports", np.zeros(Nl)))
        if len(exports_raw) == Nl:
            exports = np.concatenate([exports_raw, np.zeros(M)])
        else:
            exports = exports_raw

        import_cost = _to_np(cd.get("import_cost", cd.get("imp_cost", np.ones(Nl) * 1.1)))

        gamma_cons = _to_np(cd.get("gamma_cons", gamma[0]))
        rho_cons = _to_np(cd.get("rho_cons", rho[0]))

        countries.append(CountryParams(
            alpha=alpha, gamma=gamma, rho=rho,
            beta=beta, A=A, exports=exports,
            gamma_cons=gamma_cons, rho_cons=rho_cons,
            import_cost=import_cost, L=L,
            Ml=Ml, M_factors=M,
        ))

    return TwoCountryParams(home=countries[0], foreign=countries[1])
