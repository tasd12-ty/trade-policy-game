#!/usr/bin/env python3
"""供应链扩展演示：中断冲击。"""

import sys
sys.path.insert(0, ".")

from eco_model_v2 import run_simulation, make_symmetric_params
from eco_model_v2.supply_chain import (
    SupplyChainNode,
    SupplyChainNetwork,
    disruption_transform,
    bottleneck_transform,
)


def main():
    print("=" * 60)
    print("eco_model_v2 供应链扩展演示")
    print("=" * 60)

    params = make_symmetric_params(Nl=4, Ml=1, M_factors=1)

    # 构建供应链网络
    net = SupplyChainNetwork()
    # H 国部门 2 → F 国部门 2 存在瓶颈
    net.add_edge(
        SupplyChainNode("H", 2),
        SupplyChainNode("F", 2),
        bottleneck_transform(capacity=0.3, price_elasticity=3.0),
    )
    # F 国部门 3 → H 国部门 3 遭遇供应中断
    net.add_edge(
        SupplyChainNode("F", 3),
        SupplyChainNode("H", 3),
        disruption_transform(severity=0.4),
    )

    print(f"\n供应链边数: {len(net.edges)}")
    for e in net.edges:
        print(f"  {e.source.country}[{e.source.sector}] → {e.target.country}[{e.target.sector}]")

    # 运行基线（无供应链）
    sim_base = run_simulation(params=params, periods=50, tau=0.2)
    h_base = sim_base.current_state["H"]

    # 运行带供应链
    sim_sc = run_simulation(params=params, periods=50, tau=0.2, supply_chain=net)
    h_sc = sim_sc.current_state["H"]

    Nl = params.Nl
    print(f"\n50 期后对比:")
    print(f"{'':>16} {'基线':>12} {'供应链':>12}")
    for j in range(Nl):
        print(f"  部门 {j} 价格: {h_base.price[j]:>12.4f} {h_sc.price[j]:>12.4f}")
    print(f"  {'H 收入':>12}: {h_base.income:>12.4f} {h_sc.income:>12.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
