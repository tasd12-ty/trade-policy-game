"""C 类参数计算的外生输入配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ExternalCInputs:
    """C 类计算所需外生输入（统一默认 1）。"""

    horizon_periods: int = 1
    a_i_default: float = 1.0
    rho_cj_default: float = 1.0
    gamma_ij_fill_on_zero_total: float = 1.0
    gamma_cj_fill_on_zero_total: float = 1.0
    gamma_ij_clip_min: float = 1e-6
    gamma_ij_clip_max: float = 1.0 - 1e-6
    gamma_cj_clip_min: float = 1e-6
    gamma_cj_clip_max: float = 1.0 - 1e-6

    export_base_default: float = 1.0
    p_t_o_base_default: float = 1.0
    export_multiplier_default: float = 1.0
    p_t_o_multiplier_default: float = 1.0

    a_i_by_sector: Dict[str, float] = field(default_factory=dict)
    rho_cj_by_sector: Dict[str, float] = field(default_factory=dict)
    gamma_cj_by_sector: Dict[str, float] = field(default_factory=dict)
    gamma_ij_by_user_input: Dict[str, Dict[str, float]] = field(default_factory=dict)

    export_base_by_sector: Dict[str, float] = field(default_factory=dict)
    p_t_o_base_by_sector: Dict[str, float] = field(default_factory=dict)

    export_multiplier_scalar_by_period: Dict[int, float] = field(default_factory=dict)
    p_t_o_multiplier_scalar_by_period: Dict[int, float] = field(default_factory=dict)
    export_multiplier_by_period_sector: Dict[int, Dict[str, float]] = field(default_factory=dict)
    p_t_o_multiplier_by_period_sector: Dict[int, Dict[str, float]] = field(default_factory=dict)


def _coerce_float(data: Dict[str, Any], key: str, default: float = 1.0) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(data: Dict[str, Any], key: str, default: int = 1) -> int:
    value = data.get(key, default)
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(iv, 1)


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


def _coerce_nested_map(data: Dict[str, Any], key: str) -> Dict[str, Dict[str, float]]:
    raw = data.get(key, {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for outer_k, inner in raw.items():
        if not isinstance(inner, dict):
            continue
        cleaned: Dict[str, float] = {}
        for inner_k, inner_v in inner.items():
            try:
                cleaned[str(inner_k)] = float(inner_v)
            except (TypeError, ValueError):
                continue
        out[str(outer_k)] = cleaned
    return out


def _coerce_period_scalar_map(data: Dict[str, Any], key: str) -> Dict[int, float]:
    raw = data.get(key, {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            p = int(k)
            out[p] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _coerce_period_sector_map(data: Dict[str, Any], key: str) -> Dict[int, Dict[str, float]]:
    raw = data.get(key, {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for pk, pv in raw.items():
        try:
            period = int(pk)
        except (TypeError, ValueError):
            continue
        if not isinstance(pv, dict):
            continue
        inner: Dict[str, float] = {}
        for sk, sv in pv.items():
            try:
                inner[str(sk)] = float(sv)
            except (TypeError, ValueError):
                continue
        out[period] = inner
    return out


def load_external_c_inputs(path: Path | None) -> ExternalCInputs:
    """从 JSON 读取 C 类外生输入；path=None 时返回默认值。"""

    if path is None:
        return ExternalCInputs()
    if not path.exists():
        raise FileNotFoundError(f"External inputs JSON not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("External inputs JSON must be an object")

    return ExternalCInputs(
        horizon_periods=_coerce_int(data, "horizon_periods", 1),
        a_i_default=_coerce_float(data, "a_i_default", 1.0),
        rho_cj_default=_coerce_float(data, "rho_cj_default", 1.0),
        gamma_ij_fill_on_zero_total=_coerce_float(data, "gamma_ij_fill_on_zero_total", 1.0),
        gamma_cj_fill_on_zero_total=_coerce_float(data, "gamma_cj_fill_on_zero_total", 1.0),
        gamma_ij_clip_min=_coerce_float(data, "gamma_ij_clip_min", 1e-6),
        gamma_ij_clip_max=_coerce_float(data, "gamma_ij_clip_max", 1.0 - 1e-6),
        gamma_cj_clip_min=_coerce_float(data, "gamma_cj_clip_min", 1e-6),
        gamma_cj_clip_max=_coerce_float(data, "gamma_cj_clip_max", 1.0 - 1e-6),
        export_base_default=_coerce_float(data, "export_base_default", 1.0),
        p_t_o_base_default=_coerce_float(data, "p_t_o_base_default", 1.0),
        export_multiplier_default=_coerce_float(data, "export_multiplier_default", 1.0),
        p_t_o_multiplier_default=_coerce_float(data, "p_t_o_multiplier_default", 1.0),
        a_i_by_sector=_coerce_map(data, "a_i_by_sector"),
        rho_cj_by_sector=_coerce_map(data, "rho_cj_by_sector"),
        gamma_cj_by_sector=_coerce_map(data, "gamma_cj_by_sector"),
        gamma_ij_by_user_input=_coerce_nested_map(data, "gamma_ij_by_user_input"),
        export_base_by_sector=_coerce_map(data, "export_base_by_sector"),
        p_t_o_base_by_sector=_coerce_map(data, "p_t_o_base_by_sector"),
        export_multiplier_scalar_by_period=_coerce_period_scalar_map(data, "export_multiplier_scalar_by_period"),
        p_t_o_multiplier_scalar_by_period=_coerce_period_scalar_map(data, "p_t_o_multiplier_scalar_by_period"),
        export_multiplier_by_period_sector=_coerce_period_sector_map(data, "export_multiplier_by_period_sector"),
        p_t_o_multiplier_by_period_sector=_coerce_period_sector_map(data, "p_t_o_multiplier_by_period_sector"),
    )

