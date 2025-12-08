from __future__ import annotations

from typing import Any, Dict, Optional

"""
动作校验与执行工具：确保智能体输出的政策映射在安全范围内，并提供明确的错误反馈。
"""

# Moved from EcoModel.agent_loop to eco_simu.agent_loop.


def _clip_mapping(mapping: Optional[Dict[int, float]], low: float, high: float) -> Dict[int, float]:
    """校验映射数值是否落在合法区间；若超界则抛出错误供调用端重试。"""
    if not mapping:
        return {}
    out: Dict[int, float] = {}
    for key, value in mapping.items():
        try:
            idx = int(key)
        except Exception as exc:
            raise ValueError(f"sector key must be int-like: {key}") from exc
        val = float(value)
        if not (low <= val <= high):
            raise ValueError(f"value out of range for sector {idx}: {val} not in [{low}, {high}]")
        out[idx] = val
    return out


def _validate_action_strict(action: Dict[str, Any], n_sectors: int, max_per_type: Optional[int] = None) -> Dict[str, Any]:
    """严格校验动作结构与取值范围。"""
    if not isinstance(action, dict):
        raise ValueError("action must be dict")
    action = dict(action)
    action.setdefault("type", "policy_bundle")
    if action.get("type") != "policy_bundle":
        raise ValueError("action.type must be 'policy_bundle'")

    def _check_map(mapping: Any, low: float, high: float, name: str) -> Dict[int, float]:
        if mapping is None:
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(f"{name} must be object map")
        out_map: Dict[int, float] = {}
        for key, value in mapping.items():
            idx = int(key)
            if idx < 0 or idx >= int(n_sectors):
                raise ValueError(f"{name} sector out of range: {idx}")
            val = float(value)
            if not (low <= val <= high):
                raise ValueError(f"{name} value out of range: {val}")
            out_map[idx] = val
        if max_per_type is not None and len(out_map) > int(max_per_type):
            raise ValueError(f"{name} too many sectors: {len(out_map)} > {max_per_type}")
        return out_map

    action["import_tariff"] = _check_map(action.get("import_tariff"), -1.0, 1.0, "import_tariff")
    action["export_quota"] = _check_map(action.get("export_quota"), 0.0, 1.0, "export_quota")
    action["import_multiplier"] = {}
    return action


def zeros_like_action() -> Dict[str, Any]:
    """返回零动作，占位用。"""
    return {"type": "policy_bundle", "actor": "H", "import_tariff": {}, "export_quota": {}, "import_multiplier": {}}


def apply_actions_to_sim(sim: Any, actor: str, action: Dict[str, Any]) -> None:
    """将智能体动作施加到双国仿真器上。"""
    actor = (actor or action.get("actor") or "H").upper()
    tariff = action.get("import_tariff") or {}
    quota = action.get("export_quota") or {}
    multiplier = action.get("import_multiplier") or {}

    if quota:
        sim.apply_export_control(actor, quota)
    if tariff:
        sim.apply_import_tariff(actor, tariff)
    if multiplier:
        sim.set_import_multiplier(actor, multiplier, relative_to_baseline=True)


__all__ = [
    "apply_actions_to_sim",
    "zeros_like_action",
    "_clip_mapping",
    "_validate_action_strict",
]
