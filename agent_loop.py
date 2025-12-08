from __future__ import annotations

"""
兼容层：保持历史导入路径可用，统一从 eco_simu.agent_loop 包导出公共接口。
原本的单文件实现已拆分到子模块，这里仅负责再导出。
"""

from eco_simu.agent_loop import (
    LLMPolicyAdapter,
    MultiCountryLoopState,
    PolicyAdapter,
    PolicyFn,
    SearchPolicyAdapter,
    apply_actions_to_sim,
    build_multilateral_workflow,
    build_obs,
    build_policy_from_spec,
    compute_reward,
    parse_reward_weights,
    run_bilateral_loop,
    run_multilateral_with_graph,
    zeros_like_action,
    _clip_mapping,
    _single_rewards_from_log,
    _stable,
    _validate_action_strict,
)

__all__ = [
    "LLMPolicyAdapter",
    "SearchPolicyAdapter",
    "MultiCountryLoopState",
    "PolicyAdapter",
    "PolicyFn",
    "apply_actions_to_sim",
    "build_multilateral_workflow",
    "build_obs",
    "build_policy_from_spec",
    "compute_reward",
    "parse_reward_weights",
    "run_bilateral_loop",
    "run_multilateral_with_graph",
    "zeros_like_action",
    "_clip_mapping",
    "_single_rewards_from_log",
    "_stable",
    "_validate_action_strict",
]
