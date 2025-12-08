from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


"""
观测构造工具：从仿真器状态提取当前期的宏观指标与部门向量。
"""

# Moved from EcoModel.agent_loop to eco_simu.agent_loop.


def build_obs(
    sim: Any,
    country: str,
    prev_action: Optional[Dict[str, Any]] = None,
    last_actions: Optional[Dict[str, Any]] = None,
    obs_full: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    """从仿真器状态构造观测字典。"""
    hist = sim.history[country]
    last = hist[-1]
    summ = sim.summarize_history()[country]

    metrics = {
        "price_mean_last": float(summ["price_mean"][-1]),
        "output_sum_last": float(summ["output_sum"][-1]),
        "income_last": float(summ["income"][-1]),
        "export_value_last": float(summ["export_value"][-1]),
        "import_value_last": float(summ["import_value"][-1]),
        "trade_balance_last": float(summ["trade_balance"][-1]),
        "income_growth_last": float(summ["income_growth"][-1]),
        "output_growth_last": float(summ["output_growth"][-1]),
    }

    if len(summ["income"]) >= 2:
        metrics_delta = {
            "income_d": float(summ["income"][-1] - summ["income"][-2]),
            "price_mean_d": float(summ["price_mean"][-1] - summ["price_mean"][-2]),
            "trade_balance_d": float(summ["trade_balance"][-1] - summ["trade_balance"][-2]),
            "output_sum_d": float(summ["output_sum"][-1] - summ["output_sum"][-2]),
        }
    else:
        metrics_delta = {
            "income_d": 0.0,
            "price_mean_d": 0.0,
            "trade_balance_d": 0.0,
            "output_sum_d": 0.0,
        }

    vectors = {
        "price": last.price.detach().cpu().numpy().astype(float).copy(),
        "output": last.output.detach().cpu().numpy().astype(float).copy(),
        "export_actual": last.export_actual.detach().cpu().numpy().astype(float).copy(),
        "X_imp": last.X_imp.detach().cpu().numpy().astype(float).copy(),
        "C_imp": last.C_imp.detach().cpu().numpy().astype(float).copy(),
    }

    vec_summary: Dict[str, Any] = {}
    if len(hist) >= 2:
        prev = hist[-2]

        def _topk_delta(cur: np.ndarray, pre: np.ndarray, k: int) -> List[Tuple[int, float]]:
            delta = (np.array(cur, float) - np.array(pre, float)).astype(float)
            idx = np.argsort(-np.abs(delta))[: max(k, 0)]
            return [(int(i), float(delta[int(i)])) for i in idx]

        vec_summary["price_topk_d"] = _topk_delta(
            last.price.detach().cpu().numpy(), prev.price.detach().cpu().numpy(), int(top_k)
        )
        vec_summary["output_topk_d"] = _topk_delta(
            last.output.detach().cpu().numpy(), prev.output.detach().cpu().numpy(), int(top_k)
        )
    else:
        vec_summary["price_topk_d"] = []
        vec_summary["output_topk_d"] = []

    obs: Dict[str, Any] = {
        "t": len(hist) - 1,
        "country": country,
        "metrics": metrics,
        "metrics_delta": metrics_delta,
        "vectors": vectors,
        "vectors_summary": vec_summary,
        "prev_action": prev_action or {},
        "last_actions": last_actions or {},
    }
    if not obs_full:
        obs["vectors"] = {}
    return obs


__all__ = ["build_obs"]
