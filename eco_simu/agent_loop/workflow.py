from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph  # type: ignore

from .actions import apply_actions_to_sim, _validate_action_strict
from .observations import build_obs
from .policies import PolicyFn
from .reward import _stable, compute_reward

"""
多国博弈调度：通过 LangGraph 编排“观测→决策→执行→仿真→评估”的循环。
"""

# Moved from EcoModel.agent_loop to eco_simu.agent_loop.


@dataclass
class MultiCountryLoopState:
    """多国循环的状态容器，供 LangGraph 工作流使用。"""

    t: int
    sim: Any
    countries: List[str]
    obs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    action: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reward: Dict[str, float] = field(default_factory=dict)
    log: List[Dict[str, Any]] = field(default_factory=list)
    k_per_step: int = 5
    max_rounds: int = 50
    policy: Dict[str, Optional[PolicyFn]] = field(default_factory=dict)
    policy_bound: Dict[str, bool] = field(default_factory=dict)
    reward_weights: Dict[str, Optional[Dict[str, float]]] = field(default_factory=dict)
    use_early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_eps: float = 1e-6
    mode: str = "simultaneous"
    order: List[str] = field(default_factory=list)
    pending_new: List[str] = field(default_factory=list)
    expand_hook: Optional[Callable[["MultiCountryLoopState"], List[str]]] = None
    obs_full: bool = True
    obs_topk: int = 5
    max_sectors_per_type: Optional[int] = None
    recursion_limit: Optional[int] = None
    human_log_dir: Optional[str] = None
    human_log_enabled: bool = True
    run_tag: Optional[str] = None
    fixed_import_multiplier: Optional[float] = None


def _multi_node_observe(state: MultiCountryLoopState) -> MultiCountryLoopState:
    if not getattr(state, "run_tag", None):
        state.run_tag = time.strftime("%Y%m%d-%H%M%S")
        try:
            setattr(state, "run_start_ts", float(time.time()))
        except Exception:
            pass
    for country in state.countries:
        state.obs[country] = build_obs(
            state.sim,
            country,
            prev_action=state.action.get(country),
            last_actions=state.action,
            obs_full=bool(getattr(state, "obs_full", False)),
            top_k=int(getattr(state, "obs_topk", 5)),
        )
        try:
            weights = (state.reward_weights or {}).get(country)
        except Exception:
            weights = None
        if weights is None:
            eff = {"w_income": 1.0, "w_price": 1.0, "w_trade": 1.0}
        else:
            eff = {
                "w_income": float((weights or {}).get("w_income", 1.0)),
                "w_price": float((weights or {}).get("w_price", 0.0)),
                "w_trade": float((weights or {}).get("w_trade", 0.0)),
            }
        state.obs[country]["reward_weights"] = eff
    return state


def _multi_node_maybe_expand(state: MultiCountryLoopState) -> MultiCountryLoopState:
    new_codes: List[str] = []
    if state.pending_new:
        new_codes.extend(state.pending_new)
        state.pending_new = []
    if state.expand_hook is not None:
        try:
            hook_new = state.expand_hook(state) or []
            new_codes.extend(hook_new)
        except Exception:
            pass
    for code in [str(c).upper() for c in new_codes]:
        if code in state.countries:
            continue
        state.countries.append(code)
        try:
            n_sec = int(getattr(state.sim.params.home.alpha, "shape")[0])  # type: ignore[attr-defined]
        except Exception:
            n_sec = 5
        if state.policy is None:
            state.policy = {}
        if state.reward_weights is None:
            state.reward_weights = {}
        state.reward_weights.setdefault(code, None)
        if getattr(state, "policy_bound", None) is None:
            state.policy_bound = {}
        state.policy_bound.setdefault(code, False)
        state.action[code] = {}
        state.obs[code] = build_obs(state.sim, code, state.action.get(code))
        state.reward[code] = 0.0
    return state


def _multi_node_agent_policy(state: MultiCountryLoopState) -> MultiCountryLoopState:
    try:
        n_sec = int(getattr(state.sim.params.home.alpha, "shape")[0])  # type: ignore[attr-defined]
    except Exception:
        n_sec = 5

    def _policy(country: str) -> PolicyFn:
        _ensure_bound(country)
        policy = (state.policy or {}).get(country)
        if policy is None:
            raise RuntimeError(f"no policy provided for country '{country}' (production mode requires explicit policy)")
        return policy

    mode = (state.mode or "simultaneous").lower()
    order = [c.upper() for c in (state.order or state.countries)]
    limit = getattr(state, "max_sectors_per_type", None)
    if getattr(state, "policy_bound", None) is None:
        state.policy_bound = {}

    def _ensure_bound(country: str) -> None:
        if state.policy_bound.get(country):
            return
        policy = (state.policy or {}).get(country)
        if policy is None:
            state.policy_bound[country] = True
            return
        binder = getattr(policy, "bind_runtime", None)
        if callable(binder):
            try:
                weights = (state.reward_weights or {}).get(country)
            except Exception:
                weights = None
            try:
                binder(sim=state.sim, actor=country, reward_weights=weights, k_per_step=int(getattr(state, "k_per_step", 1)))
            except Exception:
                pass
        state.policy_bound[country] = True

    def _human_log_decision(country: str) -> None:
        if not getattr(state, "human_log_enabled", True):
            return
        base = Path(state.human_log_dir or os.getenv("AGENT_LOG_DIR", "output/agent_logs"))
        directory = base / str(state.run_tag or time.strftime("%Y%m%d-%H%M%S")) / "human"
        directory.mkdir(parents=True, exist_ok=True)
        t_val = int(getattr(state, "t", 0))
        action = state.action.get(country, {}) or {}
        obs = state.obs.get(country, {}) or {}

        csv_path = directory / f"decisions_{country}.csv"
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["t", "country", "kind", "sector", "value"])
            for kind in ("import_tariff", "export_quota", "import_multiplier"):
                mapping = action.get(kind) or {}
                if isinstance(mapping, dict):
                    for key, value in mapping.items():
                        writer.writerow([t_val, country, kind, key, value])

        md_path = directory / f"rounds_{country}.md"
        metrics = obs.get("metrics", {})
        md_lines = [
            f"## t={t_val} country={country}",
            f"metrics: income={metrics.get('income_last')}, price_mean={metrics.get('price_mean_last')}, trade_balance={metrics.get('trade_balance_last')}",
            f"import_tariff: {action.get('import_tariff', {})}",
            f"export_quota: {action.get('export_quota', {})}",
            f"import_multiplier: {action.get('import_multiplier', {})}",
        ]
        if "rationale" in action:
            md_lines.append(f"rationale: {action.get('rationale')}")
        if "plan" in action:
            md_lines.append(f"plan: {action.get('plan')}")
        md_lines.append("")
        with md_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")

    if mode == "alternating":
        current = order[state.t % max(len(order), 1)]
        state.action[current] = _policy(current)(state.obs[current])
        state.action[current] = _validate_action_strict(state.action[current], n_sectors=n_sec, max_per_type=limit)
        if getattr(state, "fixed_import_multiplier", None) is not None:
            state.action[current]["import_multiplier"] = {}
        _human_log_decision(current)
    else:
        for country in state.countries:
            state.action[country] = _policy(country)(state.obs[country])
            state.action[country] = _validate_action_strict(state.action[country], n_sectors=n_sec, max_per_type=limit)
            if getattr(state, "fixed_import_multiplier", None) is not None:
                state.action[country]["import_multiplier"] = {}
            _human_log_decision(country)
    return state


def _multi_node_apply_action(state: MultiCountryLoopState) -> MultiCountryLoopState:
    for country in state.countries:
        action = state.action.get(country) or {}
        fixed_multiplier = getattr(state, "fixed_import_multiplier", None)
        if fixed_multiplier is not None:
            try:
                n_sec = int(getattr(state.sim.params.home.alpha, "shape")[0])  # type: ignore[attr-defined]
            except Exception:
                n_sec = 5
            fixed_map = {i: float(fixed_multiplier) for i in range(n_sec)}
            action = dict(action)
            action["import_multiplier"] = fixed_map
        if action:
            apply_actions_to_sim(state.sim, actor=country, action=action)
    return state


def _multi_node_simulate_step(state: MultiCountryLoopState) -> MultiCountryLoopState:
    for _ in range(int(max(state.k_per_step, 1))):
        state.sim.step()
    state.t += 1
    return state


def _multi_node_evaluate(state: MultiCountryLoopState) -> MultiCountryLoopState:
    for country in state.countries:
        state.obs[country] = build_obs(
            state.sim,
            country,
            prev_action=state.action.get(country),
            last_actions=state.action,
            obs_full=bool(getattr(state, "obs_full", False)),
            top_k=int(getattr(state, "obs_topk", 5)),
        )
        try:
            weights = (state.reward_weights or {}).get(country)
        except Exception:
            weights = None
        if weights is None:
            eff = {"w_income": 1.0, "w_price": 1.0, "w_trade": 1.0}
        else:
            eff = {
                "w_income": float((weights or {}).get("w_income", 1.0)),
                "w_price": float((weights or {}).get("w_price", 0.0)),
                "w_trade": float((weights or {}).get("w_trade", 0.0)),
            }
        state.obs[country]["reward_weights"] = eff
        state.reward[country] = compute_reward(state.obs[country], weights=weights)
    state.log.append(
        {
            "t": state.t,
            "obs": {c: state.obs[c] for c in state.countries},
            "action": {c: state.action.get(c, {}) for c in state.countries},
            "reward": dict(state.reward),
        }
    )

    try:
        every_raw = os.getenv("AGENT_PROGRESS_EVERY", "")
        every = int(every_raw) if (every_raw is not None and every_raw.strip() != "") else 0
    except Exception:
        every = 0
    if every and (state.t == 1 or state.t % max(every, 1) == 0 or state.t >= int(getattr(state, "max_rounds", 0))):
        snapshot: Dict[str, Any] = {"t": int(state.t), "max_rounds": int(getattr(state, "max_rounds", 0)), "countries": {}}
        for country in state.countries:
            metrics = state.obs.get(country, {}).get("metrics", {})
            snapshot["countries"][country] = {
                "reward": float(state.reward.get(country, 0.0)),
                "income_growth": float(metrics.get("income_growth_last", 0.0)),
                "price_mean": float(metrics.get("price_mean_last", 0.0)),
                "trade_balance": float(metrics.get("trade_balance_last", 0.0)),
            }
        as_json = str(os.getenv("AGENT_PROGRESS_JSON", "")).strip().lower() in ("1", "true", "yes", "y")
        if as_json:
            try:
                print(json.dumps({"progress": snapshot}, ensure_ascii=False), flush=True)
            except Exception:
                pass
        else:
            parts: List[str] = [f"[progress] t={snapshot['t']}/{snapshot['max_rounds']}"]
            for country in state.countries:
                entry = snapshot["countries"][country]
                parts.append(f"{country}: reward={entry['reward']:.4f} ig={entry['income_growth']:.2f}% pm={entry['price_mean']:.3f}")
            try:
                print(" | ".join(parts), flush=True)
            except Exception:
                pass
    return state


def _multi_decide_next(state: MultiCountryLoopState) -> str:
    if state.t >= int(state.max_rounds):
        return END
    if getattr(state, "use_early_stop", False):
        all_stable = True
        for country in state.countries:
            sequence: List[float] = []
            for item in state.log:
                reward_map = item.get("reward", {})
                if isinstance(reward_map, dict) and country in reward_map:
                    sequence.append(float(reward_map[country]))
            if not _stable(sequence, int(state.early_stop_patience), float(state.early_stop_eps)):
                all_stable = False
                break
        if all_stable:
            return END
    return "observe"


def build_multilateral_workflow() -> Any:
    graph = StateGraph(MultiCountryLoopState)
    graph.add_node("observe", _multi_node_observe)
    graph.add_node("maybe_expand", _multi_node_maybe_expand)
    graph.add_node("agent_policy", _multi_node_agent_policy)
    graph.add_node("apply_action", _multi_node_apply_action)
    graph.add_node("simulate_step", _multi_node_simulate_step)
    graph.add_node("evaluate", _multi_node_evaluate)
    graph.set_entry_point("observe")
    graph.add_edge("observe", "maybe_expand")
    graph.add_edge("maybe_expand", "agent_policy")
    graph.add_edge("agent_policy", "apply_action")
    graph.add_edge("apply_action", "simulate_step")
    graph.add_edge("simulate_step", "evaluate")
    graph.add_conditional_edges("evaluate", _multi_decide_next, {"observe": "observe", END: END})
    return graph.compile()


def run_multilateral_with_graph(
    state: MultiCountryLoopState,
    mode: str = "simultaneous",
    order: Optional[List[str]] = None,
) -> MultiCountryLoopState:
    state.countries = [c.upper() for c in (state.countries or ["H", "F"])]
    state.mode = str(mode or state.mode or "simultaneous").lower()
    state.order = [c.upper() for c in (order or state.order or state.countries)]

    app = build_multilateral_workflow()
    recursion_limit = getattr(state, "recursion_limit", None)
    if recursion_limit is None:
        return app.invoke(state)
    return app.invoke(state, config={"recursion_limit": int(recursion_limit)})


def run_bilateral_loop(
    state: "TwoCountryLoopState",
    mode: str = "simultaneous",
    start_actor: str = "H",
) -> "TwoCountryLoopState":  # type: ignore[override]
    countries = ["H", "F"]
    order = [start_actor.upper(), ("F" if start_actor.upper() == "H" else "H")]
    multi = MultiCountryLoopState(
        t=state.t,
        sim=state.sim,
        countries=countries,
        obs=state.obs.copy(),
        action=state.action.copy(),
        reward=state.reward.copy(),
        log=state.log.copy(),
        k_per_step=state.k_per_step,
        max_rounds=state.max_rounds,
        policy=state.policy,
        reward_weights=state.reward_weights,
        use_early_stop=state.use_early_stop,
        early_stop_patience=state.early_stop_patience,
        early_stop_eps=state.early_stop_eps,
    )
    multi = run_multilateral_with_graph(multi, mode=("simultaneous" if mode == "simultaneous" else "alternating"), order=order)
    state.t = multi.t
    state.obs = {"H": multi.obs.get("H", {}), "F": multi.obs.get("F", {})}
    state.action = {"H": multi.action.get("H", {}), "F": multi.action.get("F", {})}
    state.reward = {"H": float(multi.reward.get("H", 0.0)), "F": float(multi.reward.get("F", 0.0))}
    state.log = multi.log
    return state


__all__ = [
    "MultiCountryLoopState",
    "build_multilateral_workflow",
    "run_multilateral_with_graph",
    "run_bilateral_loop",
]
