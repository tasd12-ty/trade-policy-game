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
    """奖励权重配置模型。
    
    定义了计算国家奖励时各项指标的权重。
    """
    w_income: float = 1.0   # 收入增长的权重
    w_price: float = 0.2    # 价格控制的权重
    w_trade: float = 0.001  # 贸易平衡的权重


class RunMode(str):
    """运行模式枚举。
    
    定义多国博弈的执行顺序。
    """
    SIMULTANEOUS = "simultaneous"  # 同时行动
    ALTERNATING = "alternating"    # 轮流行动


class ExpandSchedule(BaseModel):
    """动态扩展调度配置。
    
    定义在特定时间步加入新国家的计划。
    """
    schedule: Dict[int, List[str]] = Field(default_factory=dict)

    def get_new(self, t: int) -> List[str]:
        """获取在时间步 t 需要加入的新国家列表。"""
        return [c.upper() for c in self.schedule.get(int(t), [])]


class GraphRunRequest(BaseModel):
    """图结构运行请求模型。
    
    封装了启动多国经济仿真图运行所需的所有参数。
    """
    rounds: int = 50           # 总回合数
    k_per_step: int = 5        # 每回合仿真的步数
    countries: List[str] = Field(default_factory=lambda: ["H", "F"])  # 参与国家列表
    mode: str = RunMode.SIMULTANEOUS  # 运行模式
    order: Optional[List[str]] = None  # 行动顺序（用于轮流模式）
    policy: Optional[Dict[str, str]] = None  # 各国策略配置 spec
    default_policy: str = "llm"        # 默认策略 spec
    reward_weights: Optional[Dict[str, RewardWeights]] = None  # 各国奖励权重
    early_stop: bool = False       # 是否启用早停
    early_stop_patience: int = 5   # 早停耐心值
    early_stop_eps: float = 1e-6   # 早停阈值
    expand: Optional[ExpandSchedule] = None  # 动态扩展计划

    @validator("countries", pre=True)
    def _upper(cls, v):  # type: ignore[override]
        """确保国家代码大写。"""
        return [str(c).upper() for c in (v or [])]

    @validator("order", always=True)
    def _order(cls, v, values):  # type: ignore[override]
        """验证或生成默认顺序。"""
        if v is None:
            return list(values.get("countries", []))
        return [str(c).upper() for c in v]


class TickLog(BaseModel):
    """单步运行日志模型。"""
    t: int  # 时间步
    reward: Dict[str, float]  # 各国奖励值
    metrics: Dict[str, Dict[str, float]]  # 各国关键指标


class GraphRunResult(BaseModel):
    """图运行结果模型。"""
    rounds: int  # 实际运行回合数
    countries: List[str]  # 最终国家列表
    log: List[TickLog]  # 运行日志序列


def _build_expand_hook(schedule: Optional[ExpandSchedule]):
    """构建动态扩展的回调钩子函数。"""
    if schedule is None:
        return None

    def _hook(state: MultiCountryLoopState) -> List[str]:
        return schedule.get_new(state.t)

    return _hook


def run_graph(req: GraphRunRequest) -> GraphRunResult:
    """执行图结构的多国经济仿真。
    
    1. 初始化仿真器和各国参数。
    2. 构建各国策略和奖励配置。
    3. 初始化循环状态。
    4. 执行多国博弈循环。
    5. 收集并格式化运行日志。
    
    Args:
        req: 运行请求配置对象。
        
    Returns:
        包含运行摘要和日志的结果对象。
    """
    # 1. 初始化仿真配置
    cfg = SimulationConfig(total_periods=max(req.rounds * max(req.k_per_step, 1), 1), conflict_start=200)
    sim = simulate(cfg, params_raw=create_symmetric_parameters())

    try:
        n_sec = int(sim.params.home.alpha.shape[0])  # type: ignore[attr-defined]
    except Exception:
        n_sec = 5

    # 2. 构建策略
    pol: Dict[str, Any] = {}
    if req.policy:
        for c, spec in req.policy.items():
            pol[str(c).upper()] = build_policy_from_spec(str(spec), n_sec)
    for c in req.countries:
        pol.setdefault(c, build_policy_from_spec(req.default_policy, n_sec))

    # 3. 设置奖励权重
    rw: Dict[str, Optional[Dict[str, float]]] = {}
    if req.reward_weights:
        for c, w in req.reward_weights.items():
            rw[str(c).upper()] = {"w_income": w.w_income, "w_price": w.w_price, "w_trade": w.w_trade}
    for c in req.countries:
        rw.setdefault(c, None)

    # 4. 初始化状态
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

    # 5. 运行循环
    state = run_multilateral_with_graph(state, mode=req.mode, order=state.order)

    # 6. 处理输出日志
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

