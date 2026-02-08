#!/usr/bin/env python3
"""基础演示：构造对称两国参数，运行动态仿真，观察价格演化。"""

import numpy as np
import sys
sys.path.insert(0, ".")

from eco_model_v2 import run_simulation, make_symmetric_params


def main():
    print("=" * 60)
    print("eco_model_v2 基础演示")
    print("=" * 60)

    # 构造参数：5 部门 (2 non-tradable + 3 tradable), 1 factor
    params = make_symmetric_params(
        Nl=5, Ml=2, M_factors=1,
        alpha_diag=0.15, alpha_off=0.05,
        gamma_tradable=0.6, rho_tradable=0.3,
        export_base=0.5, import_cost_base=1.1,
    )
    print(f"\n参数: Nl={params.Nl}, Ml={params.Ml}, M={params.M}")
    print(f"Alpha 行和: {params.home.alpha.sum(axis=1)}")

    # 运行仿真
    sim = run_simulation(params=params, periods=100, tau=0.2)

    # 打印结果
    h = sim.current_state["H"]
    f = sim.current_state["F"]
    Nl = params.Nl

    print(f"\n100 期后:")
    print(f"  H 国产品价格: {h.price[:Nl]}")
    print(f"  F 国产品价格: {f.price[:Nl]}")
    print(f"  H 国要素价格: {h.price[Nl:]}")
    print(f"  H 国产出: {h.output[:Nl]}")
    print(f"  H 国收入: {h.income:.4f}")
    print(f"  贸易差额: {sim._trade_balance('H'):.4f}")

    # 施加关税冲击
    print("\n--- 施加 H 国对部门 3 加 30% 关税 ---")
    sim.apply_tariff("H", {3: 0.3})
    sim.run(50)

    h2 = sim.current_state["H"]
    print(f"\n冲击后 50 期:")
    print(f"  H 国产品价格: {h2.price[:Nl]}")
    print(f"  H 国收入: {h2.income:.4f}")
    print(f"  贸易差额: {sim._trade_balance('H'):.4f}")

    print(f"\n仿真总期数: {sim.t}")
    print("=" * 60)


if __name__ == "__main__":
    main()
