#!/usr/bin/env python3
"""策略博弈演示：H 国加关税 vs F 国固定策略。"""

import sys
sys.path.insert(0, ".")

from eco_model_v2 import (
    run_game, make_symmetric_params,
    GameConfig, FixedPolicyAgent,
)


def main():
    print("=" * 60)
    print("eco_model_v2 策略博弈演示")
    print("=" * 60)

    params = make_symmetric_params(
        Nl=4, Ml=1, M_factors=1,
        gamma_tradable=0.6, rho_tradable=0.3,
    )

    config = GameConfig(
        name="tariff_war_demo",
        rounds=5,
        decision_interval=15,
        warmup_periods=30,
        trigger_country="H",
        trigger_tariff={2: 0.2},
        trigger_settle_periods=10,
        active_sectors=[1, 2, 3],
        max_tariff=0.5,
    )

    # H 国逐步加关税，F 国固定不动
    agents = {
        "H": FixedPolicyAgent(tariff={2: 0.1, 3: 0.05}),
        "F": FixedPolicyAgent(),
    }

    result = run_game(
        params=params, config=config, agents=agents, tau=0.2,
    )

    print(f"\n博弈名称: {result.config.name}")
    print(f"总轮数: {len(result.rounds)}")
    print(f"总仿真期数: {result.history_length}")
    print()

    for rr in result.rounds:
        print(f"第 {rr.round_num} 轮:")
        print(f"  H 决策: {rr.decisions['H']}")
        print(f"  F 决策: {rr.decisions['F']}")
        print(f"  H 收益: {rr.payoffs['H']:.6f}")
        print(f"  F 收益: {rr.payoffs['F']:.6f}")

    print(f"\n累计收益:")
    print(f"  H: {result.total_payoffs['H']:.6f}")
    print(f"  F: {result.total_payoffs['F']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
