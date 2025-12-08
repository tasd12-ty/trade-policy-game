from __future__ import annotations

from typing import Any, Dict, List, Optional

"""
奖励函数与稳定性判定工具。
"""

# Moved from EcoModel.agent_loop to eco_simu.agent_loop.


def compute_reward(obs: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> float:
    """基于收入增长、价格水平与贸易差额的线性奖励。"""
    if weights is None:
        weights = {"w_income": 1.0, "w_price": 1.0, "w_trade": 1.0}
    metrics = obs["metrics"]
    reward = (
        weights.get("w_income", 1.0) * float(metrics.get("income_growth_last", 0.0))
        - weights.get("w_price", 0.0) * float(metrics.get("price_mean_last", 0.0))
        + weights.get("w_trade", 0.0) * float(metrics.get("trade_balance_last", 0.0))
    )
    return float(reward)


def parse_reward_weights(spec: Optional[str]) -> Optional[Dict[str, float]]:
    """解析形如 'w_income=1.0,w_price=0.2' 的权重字符串。"""
    if not spec:
        return None
    out: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            out[key.strip()] = float(value)
        except Exception:
            continue
    return out or None


def _stable(values: List[float], window: int, eps: float) -> bool:
    """判断最近窗口是否落在 eps 范围内，用于稳定性检测。"""
    if window <= 1 or len(values) < window:
        return False
    recent = values[-window:]
    return (max(recent) - min(recent)) < eps


def _single_rewards_from_log(log: List[Dict[str, Any]]) -> List[float]:
    """从日志记录中提取奖励序列。"""
    rewards: List[float] = []
    for item in log:
        try:
            rewards.append(float(item.get("reward", 0.0)))
        except Exception:
            continue
    return rewards


__all__ = [
    "compute_reward",
    "parse_reward_weights",
    "_stable",
    "_single_rewards_from_log",
]
