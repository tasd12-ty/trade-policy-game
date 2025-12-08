from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from eco_simu import SimulationConfig, create_symmetric_parameters, simulate
from eco_simu.agent_loop import (
    MultiCountryLoopState,
    run_multilateral_with_graph,
    build_policy_from_spec,
    parse_reward_weights,
)


class RewardWeights(BaseModel):
    w_income: float = 1.0
    w_price: float = 0.2
    w_trade: float = 0.001


class RunMode(str):
    SIMULTANEOUS = "simultaneous"
    ALTERNATING = "alternating"


class ExpandSchedule(BaseModel):
    schedule: Dict[int, List[str]] = Field(default_factory=dict)

    def get_new(self, t: int) -> List[str]:
        return [c.upper() for c in self.schedule.get(int(t), [])]


class GraphRunRequest(BaseModel):
    rounds: int = 50
    k_per_step: int = 5
    countries: List[str] = Field(default_factory=lambda: ["H", "F"])
    mode: str = RunMode.SIMULTANEOUS
    order: Optional[List[str]] = None
    policy: Optional[Dict[str, str]] = None
    default_policy: str = "llm"
    reward_weights: Optional[Dict[str, RewardWeights]] = None
    early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_eps: float = 1e-6
    expand: Optional[ExpandSchedule] = None

    @validator("countries", pre=True)
    def _upper(cls, v):  # type: ignore[override]
        return [str(c).upper() for c in (v or [])]

    @validator("order", always=True)
    def _order(cls, v, values):  # type: ignore[override]
        if v is None:
            return list(values.get("countries", []))
        return [str(c).upper() for c in v]


class TickLog(BaseModel):
    t: int
    reward: Dict[str, float]
    metrics: Dict[str, Dict[str, float]]


class GraphRunResult(BaseModel):
    rounds: int
    countries: List[str]
    log: List[TickLog]


def _build_expand_hook(schedule: Optional[ExpandSchedule]):
    if schedule is None:
        return None

    def _hook(state: MultiCountryLoopState) -> List[str]:
        return schedule.get_new(state.t)

    return _hook


def run_graph(req: GraphRunRequest) -> GraphRunResult:
    cfg = SimulationConfig(total_periods=max(req.rounds * max(req.k_per_step, 1), 1), conflict_start=200)
    sim = simulate(cfg, params_raw=create_symmetric_parameters())

    try:
        n_sec = int(sim.params.home.alpha.shape[0])  # type: ignore[attr-defined]
    except Exception:
        n_sec = 5

    pol: Dict[str, Any] = {}
    if req.policy:
        for c, spec in req.policy.items():
            pol[str(c).upper()] = build_policy_from_spec(str(spec), n_sec)
    for c in req.countries:
        pol.setdefault(c, build_policy_from_spec(req.default_policy, n_sec))

    rw: Dict[str, Optional[Dict[str, float]]] = {}
    if req.reward_weights:
        for c, w in req.reward_weights.items():
            rw[str(c).upper()] = {"w_income": w.w_income, "w_price": w.w_price, "w_trade": w.w_trade}
    for c in req.countries:
        rw.setdefault(c, None)

    state = MultiCountryLoopState(
        t=0,
        sim=sim,
        countries=list(req.countries),
        k_per_step=int(req.k_per_step),
        max_rounds=int(req.rounds),
        policy=pol,
        reward_weights=rw,
        use_early_stop=bool(req.early_stop),
        early_stop_patience=int(req.early_stop_patience),
        early_stop_eps=float(req.early_stop_eps),
        mode=req.mode,
        order=list(req.order or req.countries),
        expand_hook=_build_expand_hook(req.expand),
    )

    state = run_multilateral_with_graph(state, mode=req.mode, order=state.order)

    out_log: List[TickLog] = []
    for item in state.log:
        metrics: Dict[str, Dict[str, float]] = {}
        obs = item.get("obs", {})
        for c in state.countries:
            m = (obs.get(c) or {}).get("metrics") or {}
            metrics[c] = {
                "income_growth_last": float(m.get("income_growth_last", 0.0)),
                "price_mean_last": float(m.get("price_mean_last", 0.0)),
                "trade_balance_last": float(m.get("trade_balance_last", 0.0)),
            }
        out_log.append(TickLog(t=int(item.get("t", 0)), reward={k: float(v) for k, v in (item.get("reward") or {}).items()}, metrics=metrics))

    return GraphRunResult(rounds=int(state.t), countries=list(state.countries), log=out_log)
