#!/usr/bin/env python3
"""两国策略博弈统一入口。

支持多种策略组合：gradient vs gradient, LLM vs gradient, fixed vs tit-for-tat 等。

用法：
    # 直接修改底部 __main__ 配置块运行
    python3 eco_model_v2/examples/run_game.py

    # 或 CLI 覆盖部分参数
    python3 eco_model_v2/examples/run_game.py --rounds 5 --tag test
    python3 eco_model_v2/examples/run_game.py --h-strategy llm --f-strategy gradient

输出目录: eco_model_v2/results/<run_name>/
  - game_analysis.png   可视化面板
  - game_log.txt        完整运行日志

策略类型:
  - "gradient"     梯度优化 (Adam + torch, 需安装 torch)
  - "llm"          LLM 智能体 (需设置 API key)
  - "fixed"        固定策略 (每轮施加相同的关税/配额)
  - "tit_for_tat"  以牙还牙 (模仿对手上轮的关税)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eco_model_v2.presets import make_symmetric_params
from eco_model_v2.sandbox import EconomicSandbox, GameConfig
from eco_model_v2.agent_interface import (
    PolicyAgent, FixedPolicyAgent, TitForTatAgent, LLMPolicyAgent,
)
from eco_model_v2.plotting import summarize_history, plot_game_analysis


# ============================================================
# 配置数据结构
# ============================================================

@dataclass
class LLMConfig:
    """LLM 提供商配置。"""
    preset: str = "deepseek"            # "deepseek" | "openai" | "qwen"
    model: str = "deepseek-chat"
    api_key: Optional[str] = None       # None = 从环境变量读取
    base_url: Optional[str] = None      # None = 使用 preset 默认
    temperature: Optional[float] = None
    max_tokens: int = 2048


@dataclass
class OptConfig:
    """梯度优化配置。"""
    lr: float = 0.01
    iterations: int = 200
    multi_start: int = 8
    start_strategy: str = "noisy_current"


@dataclass
class ExperimentConfig:
    """博弈实验完整配置。

    等价于 grad_op 的 LLMGameConfig / GameConfig 的统一版本。
    """
    # ---- 实验标识 ----
    name: str = "game"
    tag: str = ""                       # 自定义后缀标签

    # ---- 模型参数 ----
    tau: float = 0.1                    # 价格调整速度 (theta_price)
    normalize_gap: bool = True          # 供需缺口按产出归一化

    # ---- 博弈参数 ----
    rounds: int = 10
    decision_interval: int = 10
    warmup_periods: int = 1000
    lookahead_periods: int = 12

    # ---- 触发事件 ----
    trigger_country: Optional[str] = "F"
    trigger_tariff: Dict[int, float] = field(default_factory=lambda: {4: 0.5})
    trigger_settle_periods: int = 0

    # ---- 策略约束 ----
    active_sectors: List[int] = field(default_factory=lambda: [2, 3])
    max_tariff: float = 1.0
    min_quota: float = 0.0

    # ---- 目标函数权重 ----
    income_weight: float = 1.0
    trade_weight: float = 1.0
    price_stability_weight: float = 1.0

    # ---- 策略分配 ----
    h_strategy: str = "gradient"        # "gradient" | "llm" | "fixed" | "tit_for_tat"
    f_strategy: str = "gradient"        # "gradient" | "llm" | "fixed" | "tit_for_tat"

    # ---- 固定策略参数 (仅 strategy="fixed" 时使用) ----
    h_fixed_tariff: Dict[int, float] = field(default_factory=dict)
    h_fixed_quota: Dict[int, float] = field(default_factory=dict)
    f_fixed_tariff: Dict[int, float] = field(default_factory=dict)
    f_fixed_quota: Dict[int, float] = field(default_factory=dict)

    # ---- 梯度优化参数 ----
    opt: OptConfig = field(default_factory=OptConfig)

    # ---- LLM 参数 ----
    llm: LLMConfig = field(default_factory=LLMConfig)

    # ---- 输出 ----
    plot: bool = True


# ============================================================
# 输出命名
# ============================================================

def make_run_name(cfg: ExperimentConfig) -> str:
    """从配置生成可读的运行目录名。

    格式: {name}_{h_strat}V{f_strat}_R{rounds}_W{warmup}_tau{tau}[_trig...]_act{sectors}[_tag]
    示例: game_gradVgrad_R10_W1000_tau0.1_trigF_s4t50_act23
          game_llmVgrad_R10_W1000_tau0.1_trigF_s4t50_act23_deepseek
    """
    parts = [cfg.name]

    # 策略对
    short = {"gradient": "grad", "llm": "llm", "fixed": "fix", "tit_for_tat": "tft"}
    parts.append(f"{short.get(cfg.h_strategy, cfg.h_strategy)}V"
                 f"{short.get(cfg.f_strategy, cfg.f_strategy)}")

    # 博弈参数
    parts.append(f"R{cfg.rounds}")
    parts.append(f"W{cfg.warmup_periods}")
    parts.append(f"tau{cfg.tau}")

    # 触发
    if cfg.trigger_country and cfg.trigger_tariff:
        trig_desc = "".join(
            f"s{s}t{int(r*100)}" for s, r in sorted(cfg.trigger_tariff.items())
        )
        parts.append(f"trig{cfg.trigger_country}_{trig_desc}")

    # 活跃部门
    if cfg.active_sectors:
        parts.append(f"act{''.join(str(s) for s in cfg.active_sectors)}")

    # 标签
    if cfg.tag:
        parts.append(cfg.tag)

    return "_".join(parts)


# ============================================================
# 智能体构建
# ============================================================

def build_agent(
    strategy: str,
    sandbox: EconomicSandbox,
    country: str,
    cfg: ExperimentConfig,
    fixed_tariff: Optional[Dict[int, float]] = None,
    fixed_quota: Optional[Dict[int, float]] = None,
) -> PolicyAgent:
    """根据策略类型构建智能体。"""

    if strategy == "gradient":
        from eco_model_v2.gradient_agent import GradientPolicyAgent
        return GradientPolicyAgent(
            sandbox, country,
            active_sectors=cfg.active_sectors,
            lookahead_periods=cfg.lookahead_periods,
            lr=cfg.opt.lr,
            iterations=cfg.opt.iterations,
            multi_start=cfg.opt.multi_start,
            start_strategy=cfg.opt.start_strategy,
            objective_weights=(
                cfg.income_weight, cfg.trade_weight, cfg.price_stability_weight,
            ),
        )

    elif strategy == "llm":
        client = _build_llm_client(cfg.llm)
        return LLMPolicyAgent(
            llm_client=client,
            active_sectors=cfg.active_sectors,
            max_tariff=cfg.max_tariff,
            min_quota=cfg.min_quota,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )

    elif strategy == "fixed":
        return FixedPolicyAgent(
            tariff=fixed_tariff or {},
            quota=fixed_quota or {},
        )

    elif strategy == "tit_for_tat":
        return TitForTatAgent()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _build_llm_client(llm_cfg: LLMConfig):
    """构建 LLM 客户端。

    支持 deepseek / openai / qwen preset。
    """
    PRESETS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "env_key": "DEEPSEEK_API_KEY",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "env_key": "OPENAI_API_KEY",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "env_key": "DASHSCOPE_API_KEY",
        },
    }

    preset_info = PRESETS.get(llm_cfg.preset, PRESETS["openai"])
    api_key = llm_cfg.api_key or os.environ.get(preset_info["env_key"], "")
    base_url = llm_cfg.base_url or preset_info["base_url"]

    if not api_key:
        raise RuntimeError(
            f"LLM API key not found. Set {preset_info['env_key']} environment variable "
            f"or pass api_key in LLMConfig."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  (required for LLM strategy)")

    raw_client = OpenAI(api_key=api_key, base_url=base_url)

    class _LLMClientWrapper:
        """适配 openai SDK → LLMPolicyAgent 接口。"""
        def __init__(self, client, model):
            self._client = client
            self._model = model

        def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=2048):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            kwargs = {"model": self._model, "messages": messages, "max_tokens": max_tokens}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

    return _LLMClientWrapper(raw_client, llm_cfg.model)


# ============================================================
# Tee writer (同时输出到终端和日志)
# ============================================================

class TeeWriter:
    def __init__(self, file_path: str):
        self._file = open(file_path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ============================================================
# 主运行函数
# ============================================================

def run_experiment(cfg: ExperimentConfig) -> None:
    """运行博弈实验。

    等价于 grad_op 的 run_llm_experiment / run_gradient_game。
    """
    # 输出目录
    run_name = make_run_name(cfg)
    results_base = os.path.join(os.path.dirname(__file__), "..", "results")
    output_dir = os.path.join(results_base, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Tee 日志
    log_path = os.path.join(output_dir, "game_log.txt")
    tee = TeeWriter(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        _run_inner(cfg, output_dir)
    finally:
        sys.stdout = old_stdout
        tee.close()

    print(f"\nResults saved to {output_dir}/")
    print(f"  game_analysis.png  (visualization)")
    print(f"  game_log.txt       (full log)")


def _run_inner(cfg: ExperimentConfig, output_dir: str) -> None:
    """博弈主逻辑。"""
    print("=" * 60)
    print("eco_model_v2 Game Runner")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---- 参数 ----
    params = make_symmetric_params(Nl=5, Ml=2, M_factors=1)

    game_config = GameConfig(
        name=cfg.name,
        rounds=cfg.rounds,
        decision_interval=cfg.decision_interval,
        warmup_periods=cfg.warmup_periods,
        trigger_country=cfg.trigger_country,
        trigger_tariff=cfg.trigger_tariff,
        trigger_settle_periods=cfg.trigger_settle_periods,
        active_sectors=cfg.active_sectors,
        max_tariff=cfg.max_tariff,
        min_quota=cfg.min_quota,
        income_weight=cfg.income_weight,
        trade_weight=cfg.trade_weight,
        price_stability_weight=cfg.price_stability_weight,
        lookahead_periods=cfg.lookahead_periods,
    )

    sandbox = EconomicSandbox(params, game_config, tau=cfg.tau)
    if cfg.normalize_gap:
        sandbox.sim._normalize_gap = True
        sandbox.sim._rebuild_engines()
    sandbox.initialize()

    # ---- 智能体 ----
    agent_H = build_agent(cfg.h_strategy, sandbox, "H", cfg,
                          fixed_tariff=cfg.h_fixed_tariff,
                          fixed_quota=cfg.h_fixed_quota)
    agent_F = build_agent(cfg.f_strategy, sandbox, "F", cfg,
                          fixed_tariff=cfg.f_fixed_tariff,
                          fixed_quota=cfg.f_fixed_quota)

    # ---- 配置摘要 ----
    print(f"\nExperiment: {cfg.name}")
    print(f"  Strategies:  H={cfg.h_strategy}, F={cfg.f_strategy}")
    print(f"  Rounds:      {cfg.rounds}")
    print(f"  Interval:    {cfg.decision_interval}")
    print(f"  Warmup:      {cfg.warmup_periods}")
    print(f"  Lookahead:   {cfg.lookahead_periods}")
    print(f"  tau:         {cfg.tau}")
    print(f"  norm_gap:    {cfg.normalize_gap}")
    if cfg.trigger_country:
        print(f"  Trigger:     {cfg.trigger_country} tariff={cfg.trigger_tariff}")
    print(f"  Active:      {cfg.active_sectors}")
    print(f"  Max tariff:  {cfg.max_tariff}")
    print(f"  Objective:   inc={cfg.income_weight}, "
          f"trade={cfg.trade_weight}, stab={cfg.price_stability_weight}")
    if cfg.h_strategy == "gradient" or cfg.f_strategy == "gradient":
        print(f"  Optimizer:   Adam(lr={cfg.opt.lr}), "
              f"iter={cfg.opt.iterations}, multi_start={cfg.opt.multi_start}")
    if cfg.h_strategy == "llm" or cfg.f_strategy == "llm":
        print(f"  LLM:         {cfg.llm.preset}/{cfg.llm.model}")
    if cfg.h_strategy == "fixed":
        print(f"  H fixed:     tariff={cfg.h_fixed_tariff}, quota={cfg.h_fixed_quota}")
    if cfg.f_strategy == "fixed":
        print(f"  F fixed:     tariff={cfg.f_fixed_tariff}, quota={cfg.f_fixed_quota}")
    print()

    # ---- 运行 ----
    t0 = time.time()
    result = sandbox.run_game({"H": agent_H, "F": agent_F})
    elapsed = time.time() - t0

    # ---- 结果 ----
    print(f"\n{'=' * 60}")
    print(f"Game completed in {elapsed:.1f}s")
    print(f"  Total rounds:   {len(result.rounds)}")
    print(f"  History length: {result.history_length} periods")
    print(f"  Total payoffs:  H={result.total_payoffs['H']:.4f}, "
          f"F={result.total_payoffs['F']:.4f}")

    print(f"\nRound details:")
    for rr in result.rounds:
        print(f"  R{rr.round_num}: H_payoff={rr.payoffs['H']:.4f}, "
              f"F_payoff={rr.payoffs['F']:.4f}")
        print(f"    H decision: {rr.decisions['H']}")
        print(f"    F decision: {rr.decisions['F']}")

    # ---- 可视化 ----
    if cfg.plot:
        summary = summarize_history(sandbox.sim)
        plot_path = os.path.join(output_dir, "game_analysis.png")
        plot_game_analysis(
            summary, sandbox.policy_events, plot_path,
            warmup_periods=cfg.warmup_periods,
        )
        print(f"\nPlot saved to {plot_path}")

    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def parse_tariff(s: str) -> dict:
    """解析 '4:0.5' 或 '4:0.5,3:0.3'。"""
    if not s:
        return {}
    result = {}
    for pair in s.split(","):
        k, v = pair.split(":")
        result[int(k.strip())] = float(v.strip())
    return result


def parse_sectors(s: str) -> list:
    """解析 '2,3' 或 '2 3'。"""
    return [int(x.strip()) for x in s.replace(",", " ").split()]


def apply_cli_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    """将 CLI 参数覆盖到配置中。"""
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.interval is not None:
        cfg.decision_interval = args.interval
    if args.warmup is not None:
        cfg.warmup_periods = args.warmup
    if args.lookahead is not None:
        cfg.lookahead_periods = args.lookahead
    if args.tau is not None:
        cfg.tau = args.tau
    if args.h_strategy is not None:
        cfg.h_strategy = args.h_strategy
    if args.f_strategy is not None:
        cfg.f_strategy = args.f_strategy
    if args.trigger_tariff is not None:
        cfg.trigger_tariff = parse_tariff(args.trigger_tariff)
    if args.active_sectors is not None:
        cfg.active_sectors = parse_sectors(args.active_sectors)
    if args.max_tariff is not None:
        cfg.max_tariff = args.max_tariff
    if args.lr is not None:
        cfg.opt.lr = args.lr
    if args.iterations is not None:
        cfg.opt.iterations = args.iterations
    if args.multi_start is not None:
        cfg.opt.multi_start = args.multi_start
    if args.tag is not None:
        cfg.tag = args.tag
    if args.llm_preset is not None:
        cfg.llm.preset = args.llm_preset
    if args.llm_model is not None:
        cfg.llm.model = args.llm_model
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="两国策略博弈统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认: gradient vs gradient
  python3 eco_model_v2/examples/run_game.py

  # LLM (H) vs gradient (F)
  python3 eco_model_v2/examples/run_game.py --h-strategy llm --f-strategy gradient --llm-preset deepseek

  # 快速测试 (2 轮, 50 期预热)
  python3 eco_model_v2/examples/run_game.py --rounds 2 --warmup 50 --tag quicktest

  # 自定义触发和约束
  python3 eco_model_v2/examples/run_game.py --trigger-tariff "3:0.3,4:0.5" --active-sectors "2,3,4"

  # 以牙还牙 vs 梯度优化
  python3 eco_model_v2/examples/run_game.py --h-strategy tit_for_tat --f-strategy gradient
""",
    )
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--interval", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--lookahead", type=int, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--h-strategy", type=str, default=None,
                   choices=["gradient", "llm", "fixed", "tit_for_tat"])
    p.add_argument("--f-strategy", type=str, default=None,
                   choices=["gradient", "llm", "fixed", "tit_for_tat"])
    p.add_argument("--trigger-tariff", type=str, default=None,
                   help="触发关税, 如 '4:0.5' 或 '3:0.3,4:0.5'")
    p.add_argument("--active-sectors", type=str, default=None,
                   help="活跃部门, 如 '2,3'")
    p.add_argument("--max-tariff", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--multi-start", type=int, default=None)
    p.add_argument("--llm-preset", type=str, default=None,
                   choices=["deepseek", "openai", "qwen"])
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--tag", type=str, default=None, help="自定义运行标签")
    return p


# ============================================================
# 入口: 在此修改实验配置
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 实验配置 — 修改此处即可运行不同实验
    # ========================================================
    #
    # 策略选择:
    #   "gradient"     — 梯度优化 (Adam + torch)
    #   "llm"          — LLM 智能体 (需 API key)
    #   "fixed"        — 固定策略
    #   "tit_for_tat"  — 以牙还牙
    #
    # LLM 使用方法:
    #   export DEEPSEEK_API_KEY="your-key"
    #   然后将 h_strategy 或 f_strategy 设为 "llm"
    #

    cfg = ExperimentConfig(
        name="game",

        # ---- 策略分配 ----
        h_strategy="gradient",          # H 方: gradient | llm | fixed | tit_for_tat
        f_strategy="gradient",          # F 方: gradient | llm | fixed | tit_for_tat

        # ---- 模型参数 ----
        tau=0.1,                        # 价格调整速度
        normalize_gap=True,             # 供需缺口归一化

        # ---- 博弈参数 ----
        rounds=10,
        decision_interval=10,
        warmup_periods=1000,
        lookahead_periods=12,

        # ---- 触发事件 ----
        trigger_country="F",            # F 国首先加关税
        trigger_tariff={4: 0.5},        # sector 4 加 50%
        trigger_settle_periods=0,

        # ---- 策略约束 ----
        active_sectors=[2, 3],          # 可调整的部门
        max_tariff=1.0,
        min_quota=0.0,

        # ---- 目标函数权重 ----
        income_weight=1.0,
        trade_weight=1.0,
        price_stability_weight=1.0,

        # ---- 梯度优化参数 (strategy="gradient") ----
        opt=OptConfig(
            lr=0.01,
            iterations=200,
            multi_start=8,
            start_strategy="noisy_current",
        ),

        # ---- LLM 参数 (strategy="llm") ----
        llm=LLMConfig(
            preset="deepseek",          # "deepseek" | "openai" | "qwen"
            model="deepseek-chat",
        ),

        # ---- 固定策略参数 (strategy="fixed") ----
        # h_fixed_tariff={2: 0.1, 3: 0.1},
        # f_fixed_tariff={},

        plot=True,
    )

    # ---- 快速切换示例配置 ----
    # 取消注释即可使用:

    # # 示例 1: LLM (H) vs 梯度优化 (F) — 对应旧 llm_game.py
    # #   export DEEPSEEK_API_KEY="your-key"
    # cfg.h_strategy = "llm"
    # cfg.f_strategy = "gradient"
    # cfg.llm = LLMConfig(preset="deepseek", model="deepseek-chat")

    # # 示例 2: 梯度 vs 固定（F 不做反应）
    # cfg.h_strategy = "gradient"
    # cfg.f_strategy = "fixed"
    # cfg.f_fixed_tariff = {}

    # # 示例 3: LLM 双方对弈
    # cfg.h_strategy = "llm"
    # cfg.f_strategy = "llm"
    # cfg.llm = LLMConfig(preset="openai", model="gpt-4o")

    # # 示例 4: 以牙还牙 vs 梯度
    # cfg.h_strategy = "tit_for_tat"
    # cfg.f_strategy = "gradient"

    # CLI 参数覆盖（可选）
    parser = build_parser()
    args = parser.parse_args()
    cfg = apply_cli_overrides(cfg, args)

    run_experiment(cfg)
