#!/usr/bin/env python3
"""两国梯度优化策略博弈测试。

用法：
    python3 eco_model_v2/examples/run_game_test.py              # 默认配置
    python3 eco_model_v2/examples/run_game_test.py --tag myrun  # 自定义标签
    python3 eco_model_v2/examples/run_game_test.py --rounds 5 --tau 0.2

输出目录: eco_model_v2/results/<run_name>/
  - game_analysis.png   可视化面板
  - game_log.txt        完整运行日志
"""

import argparse
import io
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eco_model_v2.presets import make_symmetric_params
from eco_model_v2.sandbox import EconomicSandbox, GameConfig
from eco_model_v2.gradient_agent import GradientPolicyAgent
from eco_model_v2.plotting import summarize_history, plot_game_analysis


# ---- 输出目录命名 ----

def make_run_name(config: GameConfig, tau: float, tag: str = "") -> str:
    """从配置生成可读的运行目录名。

    格式: {name}_R{rounds}_W{warmup}_tau{tau}_trig{country}_s{sectors}t{rates}_act{sectors}[_tag]
    示例: grad_R10_W1000_tau0.1_trigF_s4t50_act23
    """
    parts = [config.name]

    # 博弈参数
    parts.append(f"R{config.rounds}")
    parts.append(f"W{config.warmup_periods}")
    parts.append(f"tau{tau}")

    # 触发事件
    if config.trigger_country and config.trigger_tariff:
        trig_desc = "".join(
            f"s{s}t{int(r*100)}" for s, r in sorted(config.trigger_tariff.items())
        )
        parts.append(f"trig{config.trigger_country}_{trig_desc}")

    # 活跃部门
    if config.active_sectors:
        parts.append(f"act{''.join(str(s) for s in config.active_sectors)}")

    # 自定义标签
    if tag:
        parts.append(tag)

    return "_".join(parts)


class TeeWriter:
    """同时写入 stdout 和文件。"""
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


# ---- 主流程 ----

def run_game(
    rounds: int = 10,
    decision_interval: int = 10,
    warmup_periods: int = 1000,
    lookahead_periods: int = 12,
    tau: float = 0.1,
    trigger_country: str = "F",
    trigger_tariff: dict = None,
    active_sectors: list = None,
    max_tariff: float = 1.0,
    lr: float = 0.01,
    iterations: int = 200,
    multi_start: int = 8,
    objective_weights: tuple = (1.0, 1.0, 1.0),
    tag: str = "",
):
    if trigger_tariff is None:
        trigger_tariff = {4: 0.5}
    if active_sectors is None:
        active_sectors = [2, 3]

    # 参数：5 部门 (0,1 非贸易; 2,3,4 可贸易), 1 要素
    params = make_symmetric_params(Nl=5, Ml=2, M_factors=1)

    config = GameConfig(
        name="grad",
        rounds=rounds,
        decision_interval=decision_interval,
        warmup_periods=warmup_periods,
        trigger_country=trigger_country,
        trigger_tariff=trigger_tariff,
        trigger_settle_periods=0,
        active_sectors=active_sectors,
        max_tariff=max_tariff,
        min_quota=0.0,
        income_weight=objective_weights[0],
        trade_weight=objective_weights[1],
        price_stability_weight=objective_weights[2],
        lookahead_periods=lookahead_periods,
    )

    # 输出目录
    run_name = make_run_name(config, tau, tag)
    results_base = os.path.join(os.path.dirname(__file__), "..", "results")
    output_dir = os.path.join(results_base, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Tee: 同时输出到终端和日志文件
    log_path = os.path.join(output_dir, "game_log.txt")
    tee = TeeWriter(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        _run_game_inner(params, config, tau, output_dir,
                        active_sectors, lookahead_periods,
                        lr, iterations, multi_start, objective_weights)
    finally:
        sys.stdout = old_stdout
        tee.close()

    print(f"Results saved to {output_dir}/")
    print(f"  game_analysis.png")
    print(f"  game_log.txt")


def _run_game_inner(params, config, tau, output_dir,
                    active_sectors, lookahead_periods,
                    lr, iterations, multi_start, objective_weights):
    """博弈主逻辑（在 tee 重定向下运行）。"""
    print("=" * 60)
    print("eco_model_v2 Gradient Game Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    sandbox = EconomicSandbox(params, config, tau=tau)
    sandbox.sim._normalize_gap = True
    sandbox.sim._rebuild_engines()
    sandbox.initialize()

    # 梯度优化智能体
    agent_H = GradientPolicyAgent(
        sandbox, "H",
        active_sectors=active_sectors,
        lookahead_periods=lookahead_periods,
        lr=lr,
        iterations=iterations,
        multi_start=multi_start,
        start_strategy="noisy_current",
        objective_weights=objective_weights,
    )
    agent_F = GradientPolicyAgent(
        sandbox, "F",
        active_sectors=active_sectors,
        lookahead_periods=lookahead_periods,
        lr=lr,
        iterations=iterations,
        multi_start=multi_start,
        start_strategy="noisy_current",
        objective_weights=objective_weights,
    )

    # 配置摘要
    print(f"\nConfig: {config.name}")
    print(f"  Rounds:      {config.rounds}")
    print(f"  Interval:    {config.decision_interval}")
    print(f"  Warmup:      {config.warmup_periods}")
    print(f"  Lookahead:   {config.lookahead_periods}")
    print(f"  tau:         {tau}")
    print(f"  Trigger:     {config.trigger_country} tariff {config.trigger_tariff}")
    print(f"  Active:      {active_sectors}")
    print(f"  Max tariff:  {config.max_tariff}")
    print(f"  Optimizer:   Adam(lr={lr}), iter={iterations}, multi_start={multi_start}")
    print(f"  Objective:   w_inc={objective_weights[0]}, "
          f"w_trade={objective_weights[1]}, w_stab={objective_weights[2]}")
    print()

    t0 = time.time()
    result = sandbox.run_game({"H": agent_H, "F": agent_F})
    elapsed = time.time() - t0

    # 结果摘要
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

    # 可视化
    summary = summarize_history(sandbox.sim)
    plot_path = os.path.join(output_dir, "game_analysis.png")
    plot_game_analysis(
        summary, sandbox.policy_events, plot_path,
        warmup_periods=config.warmup_periods,
    )

    print(f"\nGame analysis plot saved to {plot_path}")
    print("=" * 60)


# ---- CLI ----

def parse_tariff(s: str) -> dict:
    """解析关税字符串 '4:0.5' 或 '4:0.5,3:0.3'。"""
    result = {}
    for pair in s.split(","):
        k, v = pair.split(":")
        result[int(k.strip())] = float(v.strip())
    return result


def parse_sectors(s: str) -> list:
    """解析部门列表 '2,3' 或 '2 3'。"""
    return [int(x.strip()) for x in s.replace(",", " ").split()]


def main():
    parser = argparse.ArgumentParser(description="两国梯度博弈测试")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lookahead", type=int, default=12)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--trigger-country", type=str, default="F")
    parser.add_argument("--trigger-tariff", type=str, default="4:0.5",
                        help="触发关税, 如 '4:0.5' 或 '3:0.3,4:0.5'")
    parser.add_argument("--active-sectors", type=str, default="2,3",
                        help="活跃部门, 如 '2,3'")
    parser.add_argument("--max-tariff", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--multi-start", type=int, default=8)
    parser.add_argument("--weights", type=str, default="1,1,1",
                        help="目标权重 income,trade,stability, 如 '1,1,1'")
    parser.add_argument("--tag", type=str, default="",
                        help="自定义运行标签")
    args = parser.parse_args()

    weights = tuple(float(x) for x in args.weights.split(","))

    run_game(
        rounds=args.rounds,
        decision_interval=args.interval,
        warmup_periods=args.warmup,
        lookahead_periods=args.lookahead,
        tau=args.tau,
        trigger_country=args.trigger_country,
        trigger_tariff=parse_tariff(args.trigger_tariff),
        active_sectors=parse_sectors(args.active_sectors),
        max_tariff=args.max_tariff,
        lr=args.lr,
        iterations=args.iterations,
        multi_start=args.multi_start,
        objective_weights=weights,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
