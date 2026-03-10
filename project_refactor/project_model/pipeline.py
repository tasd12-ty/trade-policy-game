"""高层启动管线。"""

from __future__ import annotations

from typing import Any, Dict

from .dynamics import TwoCountrySimulator
from .dynamics import bootstrap_dynamic_simulator as _bootstrap_dynamic
from .presets import from_raw_dict
from .types import TwoCountryParams


def bootstrap_simulator(
    params: TwoCountryParams | Dict[str, Dict[str, Any]],
    *,
    tau_price: float = 0.1,
    normalize_gap_by_supply: bool = False,
    max_iterations: int = 400,
    tolerance: float = 1e-8,
) -> TwoCountrySimulator:
    """统一启动入口。

    支持两种输入：
    1. `TwoCountryParams` 结构化参数；
    2. 与旧版 `grad_op` 相同的原始字典。
    """
    if not isinstance(params, TwoCountryParams):
        params = from_raw_dict(params)
    return _bootstrap_dynamic(
        params,
        tau_price=tau_price,
        normalize_gap_by_supply=normalize_gap_by_supply,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
