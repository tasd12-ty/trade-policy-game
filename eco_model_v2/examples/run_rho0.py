#!/usr/bin/env python3
"""ρ≡0 解析解演示。"""

import numpy as np
import sys
sys.path.insert(0, ".")

from eco_model_v2.equilibrium_rho0 import solve_equilibrium_rho0


def main():
    print("=" * 60)
    print("eco_model_v2 ρ≡0 解析解演示")
    print("=" * 60)

    Nl, Ml, M = 3, 1, 1
    alpha = np.array([
        [0.2, 0.2, 0.2, 0.4],
        [0.1, 0.3, 0.2, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ])
    gamma = np.array([
        [1.0, 0.6, 0.5],
        [1.0, 0.5, 0.6],
        [1.0, 0.6, 0.6],
    ])
    beta = np.array([0.3, 0.3, 0.4])
    A = np.array([1.0, 1.2, 0.8])
    exports = np.array([0.5, 0.5, 0.5, 0.0])
    imp_price = np.array([1.0, 1.1, 1.2])
    L = np.array([10.0])
    gamma_cons = np.array([1.0, 0.6, 0.5])

    result = solve_equilibrium_rho0(
        alpha, gamma, beta, A, exports, imp_price, L, Ml, M,
        gamma_cons=gamma_cons,
    )

    print(f"\n部门数: {Nl}, 非贸易部门: {Ml}, 要素: {M}")
    print(f"TFP: {A}")
    print()
    print(f"产品价格: {result['price'][:Nl]}")
    print(f"要素价格: {result['factor_prices']}")
    print(f"完整价格: {result['price']}")
    print(f"产出量:   {result['output']}")
    print(f"产出值 PY: {result['PY']}")
    print(f"收入:     {result['income']:.6f}")
    print(f"P[0] (名义锚): {result['price'][0]:.6f}")

    # 验证行和
    print(f"\nalpha 行和: {alpha.sum(axis=1)}")
    print(f"alpha_tilde:")
    print(result['alpha_tilde'])

    print("=" * 60)


if __name__ == "__main__":
    main()
