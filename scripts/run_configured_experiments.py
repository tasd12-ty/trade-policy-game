from __future__ import annotations

"""
配置式入口（无 CLI 参数）：编辑下方 EXPERIMENTS 列表即可复用本项目。
- 运行：python scripts/run_configured_experiments.py
- 策略：
  - 搜索：policy="search"（默认），自动读取 search_* 参数生成 search:... 规格。
  - LLM：policy="llm:model=Qwen/...,base=http://localhost:8001/v1,temp=0.2"；或通过环境变量设置 OPENAI_* 后用 policy="llm"。
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Union
import time

import numpy as np

# 确保能直接执行本文件：将仓库根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from eco_simu.model import create_symmetric_parameters
from eco_simu.sim import bootstrap_simulator
from eco_simu.plotting import plot_history, plot_diagnostics, plot_history_agent_view, plot_sector_paths
from agent_loop import (
    MultiCountryLoopState,
    run_multilateral_with_graph,
    build_policy_from_spec,
    parse_reward_weights,
)


@dataclass
class InitialPolicies:
    """初始冲突/政策：按国家代码配置，键为部门索引。"""

    import_tariff: Dict[str, Dict[int, float]] = field(default_factory=dict)  # {"H": {2: 0.1}, "F": {0: -0.05}}
    export_quota: Dict[str, Dict[int, float]] = field(default_factory=dict)  # {"H": {1: 0.8}} 0..1 乘数
    import_multiplier: Dict[str, Dict[int, float]] = field(default_factory=dict)  # {"H": {3: 1.2}} 相对基线乘数


@dataclass
class ExperimentConfig:
    """单个实验的配置：编辑字段即可运行不同情景。"""

    # --- 基础运行参数 ---
    name: str = "bayes_search_1000x100" # 实验名称，用于日志/输出前缀
    countries: Sequence[str] = ("H", "F")  # 参与博弈的国家代码列表
    mode: str = "alternating"  # 同步 "simultaneous" 或交替 "alternating"
    rounds: int = 1000  # 回合数（智能体决策轮数）
    k_per_step: int = 100  # 每轮内部仿真步长 k
    # policy: Union[str, Dict[str, str]] = "llm:model=Qwen/Qwen2.5-7B-Instruct,base=http://localhost:8001/v1,temp=0.2,step=0.05"  # LLM 或 search spec，可按国家映射
    # policy 填 "search" 或不含 ":" 时，使用下方 search_* 参数自动生成 search:... 规格；若填 "llm" 则使用 OPENAI_* 环境变量
    policy: Union[str, Dict[str, str]] = "search"  # 默认走搜索策略，按 search_* 配置
    reward_weights: Optional[str] = "w_income=1.0,w_price=0.2,w_trade=0.1"  # 奖励权重字符串，如 "w_income=1.0,w_price=0.5,w_trade=0.5"
    warmup_window: int = 100  # 预热稳定窗口长度
    warmup_eps: float = 0.001  # 预热判定阈值（变化绝对值）
    warmup_max: int = 2000  # 预热最大步数
    require_stable_warmup: bool = True  # 预热不稳定时是否中止
    theta_price: float = 1  # 价格调整系数 τ
    max_sectors_per_type: Optional[int] = None  # 每类政策最多作用的部门数限制
    recursion_limit: Optional[int] = None  # LangGraph 递归限制，None 自动估算
    fixed_import_multiplier: Optional[float] = None  # 若设定则锁定 import_multiplier（忽略智能体）

    # --- 输出与绘图 ---
    plot: bool = True  # 是否生成图表
    plot_dir: str = "results-config/bayes_search_1000x100"  # 图表输出目录
    plot_sectors: Sequence[int] = (0, 1, 2)  # 绘图的部门索引
    results_dir: str = "results-config/bayes_search_1000x100"  # CSV 输出目录
    agent_log_dir: str = "output/agent_logs"  # 智能体日志目录
    output_tag: Optional[str] = "bayes_search_1000x100"  # 自定义文件名前缀，None 用 name

    # --- 初始政策 ---
    initial: InitialPolicies = field(default_factory=InitialPolicies)  # 初始冲突/政策

    # --- 搜索策略参数（当 policy=search 或值不含冒号时生效）---
    search_method: str = "grid"
    search_lookahead_rounds: int = 200  # 按回合推演步数；与 k_per_step 相乘得到仿真步
    search_lookahead_steps: Optional[int] = 200  # 直接指定仿真步数（覆盖 lookahead_rounds*k_per_step），None 表示使用回合推演
    search_tariff_grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)  # 关税候选值
    search_quota_grid: Sequence[float] = (0.9, 0.5,0)  # 出口配额候选值 (0..1, 1 为不限制)
    search_sectors: Sequence[int] = (0, 1)  # 必须指定要搜索的部门（示例默认 0,1，可自行修改）
    search_objective: str = "reward"  # 可选 reward|income|trade|price


def _stable_window(arr: np.ndarray, window: int, eps: float) -> bool:
    if window <= 1 or arr.shape[0] < window + 1:
        return False
    diffs = np.diff(arr[-window - 1 :])
    return bool(np.all(np.abs(diffs) < eps))


def _warmup_until_stable(sim, window: int, eps: float, max_steps: int, require: bool) -> Dict[str, Union[int, bool, float]]:
    steps = 0
    stable = False
    t0 = time.time()
    while steps < max_steps:
        summary = sim.summarize_history()
        ok = True
        for c in ("H", "F"):
            m = summary.get(c, {})
            ig = np.array(m.get("income_growth", []), float)
            og = np.array(m.get("output_growth", []), float)
            if not (_stable_window(ig, window, eps) and _stable_window(og, window, eps)):
                ok = False
                break
        if ok:
            stable = True
            break
        sim.step()
        steps += 1
    if require and (not stable):
        raise RuntimeError(f"[warmup] 未在 {max_steps} 步内达到稳定 (window={window}, eps={eps})")
    return {"steps": steps, "stable": stable, "seconds": time.time() - t0}


def _apply_initial_policies(sim, init: InitialPolicies) -> None:
    for actor, mapping in (init.import_tariff or {}).items():
        sim.apply_import_tariff(actor, mapping, note="initial import tariff")
    for actor, mapping in (init.export_quota or {}).items():
        sim.apply_export_control(actor, mapping, note="initial export quota")
    for actor, mapping in (init.import_multiplier or {}).items():
        sim.set_import_multiplier(actor, mapping, relative_to_baseline=True, note="initial import multiplier")


def _make_search_spec(cfg: ExperimentConfig) -> str:
    """根据 ExperimentConfig 构造 search:... 规格，避免在字符串里手写网格/部门。"""
    parts: List[str] = []

    def _add(key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            if not value:
                return
            val_str = ";".join(str(v) for v in value)
        else:
            val_str = str(value)
        parts.append(f"{key}={val_str}")

    if not cfg.search_sectors:
        raise ValueError("search_sectors must be set (e.g., (0,1) or (0,2,4))")
    _add("method", "grid")
    _add("lookahead", cfg.search_lookahead_rounds)
    _add("steps", cfg.search_lookahead_steps)
    _add("tariffs", cfg.search_tariff_grid)
    _add("quotas", cfg.search_quota_grid)
    _add("sectors", cfg.search_sectors)
    _add("objective", cfg.search_objective)
    return "search:" + ",".join(parts)


def _build_policy_map(policy_spec: Union[str, Dict[str, str]], countries: Sequence[str], n_sec: int, cfg: ExperimentConfig) -> Dict[str, Any]:
    """支持 policy=search 时自动拼装 search:... 规格，减少手写字符串。"""
    def _resolve(spec: str) -> str:
        spec = (spec or "").strip()
        if spec.lower().startswith("search") and ":" not in spec:
            return _make_search_spec(cfg)
        return spec

    pol: Dict[str, Any] = {}
    if isinstance(policy_spec, dict):
        for c, spec in policy_spec.items():
            pol[c.upper()] = build_policy_from_spec(_resolve(str(spec)), int(n_sec))
    else:
        resolved = _resolve(str(policy_spec))
        for c in countries:
            pol[c.upper()] = build_policy_from_spec(resolved, int(n_sec))
    return pol


def _iter_log(state: Union[MultiCountryLoopState, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """兼容 LangGraph 返回 dict 或对象的日志迭代器。"""
    if isinstance(state, dict):
        return list(state.get("log", []))
    return list(getattr(state, "log", []))


def _export_csv(state: MultiCountryLoopState, countries: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        import csv

        writer = csv.writer(fh)
        header = ["t"]
        for c in countries:
            header += [f"reward_{c}", f"income_growth_{c}", f"price_mean_{c}", f"trade_balance_{c}"]
        writer.writerow(header)
        for item in _iter_log(state):
            row = [item.get("t")]
            for c in countries:
                obs_c = item["obs"][c]
                metrics = obs_c["metrics"]
                row += [
                    item["reward"].get(c),
                    metrics.get("income_growth_last"),
                    metrics.get("price_mean_last"),
                    metrics.get("trade_balance_last"),
                ]
            writer.writerow(row)


def _maybe_plot(sim, state: MultiCountryLoopState, cfg: ExperimentConfig, tag: str, countries: Sequence[str], warmup_steps: int = 0) -> None:
    try:
        plot_history(sim, save_path=f"{cfg.plot_dir}/{tag}_summary.png", show=False, warmup=warmup_steps)
        plot_diagnostics(sim, save_path=f"{cfg.plot_dir}/{tag}_diagnostics.png", show=False, warmup=warmup_steps)
        try:
            agent_log = _iter_log(state)
        except Exception:
            agent_log = []
        plot_history_agent_view(
            sim,
            agent_log=agent_log,
            k_per_step=int(getattr(state, "k_per_step", cfg.k_per_step)),
            save_path=f"{cfg.plot_dir}/{tag}_summary_agent.png",
            show=False,
            annotate_decisions=True,
            warmup=warmup_steps,
        )
        secs = [int(s) for s in cfg.plot_sectors]
        for c in countries:
            plot_sector_paths(
                sim,
                c,
                "output",
                sectors=secs,
                save_path=f"{cfg.plot_dir}/{tag}_sector_output_{c}.png",
                show=False,
                relative=True,
                warmup=warmup_steps,
            )
            plot_sector_paths(
                sim,
                c,
                "price",
                sectors=secs,
                save_path=f"{cfg.plot_dir}/{tag}_sector_price_{c}.png",
                show=False,
                relative=True,
                warmup=warmup_steps,
            )
    except Exception as exc:
        print(f"[plot] failed: {exc}")


def run_experiment(cfg: ExperimentConfig) -> Path:
    countries = [c.upper() for c in cfg.countries]
    out_tag = cfg.output_tag or cfg.name
    plot_dir = Path(cfg.plot_dir)
    results_dir = Path(cfg.results_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.agent_log_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AGENT_LOG_DIR", str(Path(cfg.agent_log_dir).absolute()))

    sim = bootstrap_simulator(params_raw=create_symmetric_parameters(), theta_price=cfg.theta_price)
    n_sec = int(getattr(sim.params.home.alpha, "shape")[0])  # type: ignore[attr-defined]
    _apply_initial_policies(sim, cfg.initial)

    warm_info = _warmup_until_stable(sim, cfg.warmup_window, cfg.warmup_eps, cfg.warmup_max, cfg.require_stable_warmup)
    warmup_steps = int(warm_info.get("steps", 0) or 0)
    print(f"[{cfg.name}] warmup steps={warm_info['steps']} stable={warm_info['stable']} time={warm_info['seconds']:.2f}s")

    policy_map = _build_policy_map(cfg.policy, countries, n_sec, cfg)
    rw_common = parse_reward_weights(cfg.reward_weights)
    reward_weights = {c: rw_common for c in countries}
    rec_limit = cfg.recursion_limit
    if rec_limit is None:
        rec_limit = cfg.rounds * 100 + 200

    state = MultiCountryLoopState(
        t=0,
        sim=sim,
        countries=countries,
        k_per_step=cfg.k_per_step,
        max_rounds=cfg.rounds,
        policy=policy_map,
        reward_weights=reward_weights,
        use_early_stop=False,
        obs_full=True,
        obs_topk=5,
        max_sectors_per_type=cfg.max_sectors_per_type,
        recursion_limit=rec_limit,
        fixed_import_multiplier=cfg.fixed_import_multiplier,
    )
    mode = (cfg.mode or "simultaneous").lower()
    state = run_multilateral_with_graph(state, mode=("alternating" if "alt" in mode else "simultaneous"), order=countries)

    out_csv = results_dir / f"{out_tag}.csv"
    _export_csv(state, countries, out_csv)
    print(f"[{cfg.name}] csv -> {out_csv}")

    if cfg.plot:
        _maybe_plot(sim, state, cfg, out_tag, countries, warmup_steps=warmup_steps)
        print(f"[{cfg.name}] plots -> {cfg.plot_dir}/{out_tag}_*.png")
    return out_csv


# ---------------------------------------
# 在此处配置多个实验；编辑即可重复试验
# ---------------------------------------

# 说明：将 policy 改为 "llm" 或 "llm:model=..." 可切换为大模型决策；按国区分时用字典 {"H": "...", "F": "..."}。
# 默认示例仍使用搜索策略，方便快速复现。
EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="search_game_100x100",
        mode="alternating",
        rounds=100,  # 总共博弈 100 次
        k_per_step=15,  # 决策间隔 100 期
        policy={
            "H": "llm:model=/home/u20249114/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct,base=http://localhost:8001/v1,temp=0.7",
            "F": "search:method=grid,steps=10,tariffs=0.1;0.2;0.3;0.4;0.5;0.6,quotas=0.9;0.5;0,sectors=1;2",
        },
        reward_weights="w_income=1.0,w_price=0.2,w_trade=0.1",
        # 初始冲击：H 对部门 2 征收 10% 关税
        initial=InitialPolicies(import_tariff={"H": {2: 0.1}}),
        # 搜索配置（仅 grid）
        search_method="grid",
        search_lookahead_steps=100,  # 前瞻 100 期 (覆盖 lookahead_rounds)
        search_tariff_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        search_quota_grid=(0.9, 0.5),
        search_sectors=(0,),  # 搜索部门
        warmup_window=100,
        warmup_eps=0.001,
        warmup_max=2000,
        require_stable_warmup=True,
        plot=True,
        results_dir="results-llm_vs_search-config3",
        plot_dir="results-llm_vs_search-config3",
        output_tag="llm_vs_search_100x100-3",
    ),
]


if __name__ == "__main__":
    for cfg in EXPERIMENTS:
        run_experiment(cfg)
