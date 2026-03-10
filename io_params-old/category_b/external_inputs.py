"""B 类参数计算的外部输入配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ExternalBInputs:
    """B 类计算所需的外生输入。

    约定：可外生给定的数值默认均为 1.0。
    """

    labor_income_total: float = 1.0
    labor_population: float = 1.0
    interest_rate_r: float = 1.0
    capital_income_total: float = 1.0
    domestic_price_default: float = 1.0
    import_price_multiplier_default: float = 1.0
    export_value_fallback: float = 1.0
    # 价格联立相关外生参数（默认也为 1）
    rho_default: float = 1.0
    tfp_A_default: float = 1.0
    import_cost_default: float = 1.0
    use_all_sectors_tradable: float = 1.0
    fallback_to_exogenous_if_solver_fails: float = 1.0
    domestic_prices_by_sector: Dict[str, float] = field(default_factory=dict)
    import_prices_by_sector: Dict[str, float] = field(default_factory=dict)


def _coerce_float(data: Dict[str, Any], key: str, default: float = 1.0) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_map(data: Dict[str, Any], key: str) -> Dict[str, float]:
    raw = data.get(key, {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _coerce_mode(data: Dict[str, Any]) -> str:
    mode = str(data.get("price_mode", "endogenous")).strip().lower()
    if mode not in {"endogenous", "exogenous"}:
        return "endogenous"
    return mode


def load_external_b_inputs(path: Path | None) -> ExternalBInputs:
    """从 JSON 读取 B 类外部输入。

    若 path 为 None，返回默认值对象。
    """

    if path is None:
        return ExternalBInputs()
    if not path.exists():
        raise FileNotFoundError(f"External inputs JSON not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("External inputs JSON must be an object")

    return ExternalBInputs(
        labor_income_total=_coerce_float(data, "labor_income_total", 1.0),
        labor_population=_coerce_float(data, "labor_population", 1.0),
        interest_rate_r=_coerce_float(data, "interest_rate_r", 1.0),
        capital_income_total=_coerce_float(data, "capital_income_total", 1.0),
        domestic_price_default=_coerce_float(data, "domestic_price_default", 1.0),
        import_price_multiplier_default=_coerce_float(data, "import_price_multiplier_default", 1.0),
        export_value_fallback=_coerce_float(data, "export_value_fallback", 1.0),
        rho_default=_coerce_float(data, "rho_default", 1.0),
        tfp_A_default=_coerce_float(data, "tfp_A_default", 1.0),
        import_cost_default=_coerce_float(data, "import_cost_default", 1.0),
        use_all_sectors_tradable=_coerce_float(data, "use_all_sectors_tradable", 1.0),
        fallback_to_exogenous_if_solver_fails=_coerce_float(data, "fallback_to_exogenous_if_solver_fails", 1.0),
        domestic_prices_by_sector=_coerce_map(data, "domestic_prices_by_sector"),
        import_prices_by_sector=_coerce_map(data, "import_prices_by_sector"),
    )


def load_price_mode(path: Path | None) -> str:
    """读取价格计算模式（endogenous 或 exogenous）。"""

    if path is None:
        return "endogenous"
    if not path.exists():
        raise FileNotFoundError(f"External inputs JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("External inputs JSON must be an object")
    return _coerce_mode(data)
