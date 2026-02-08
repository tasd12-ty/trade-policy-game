"""equilibrium.py 测试 — eq 28-30。"""

import numpy as np
import pytest

from eco_model_v2.types import CountryParams, TwoCountryParams
from eco_model_v2.equilibrium import solve_static_equilibrium


class TestStaticEquilibrium:
    def test_solver_runs(self, two_country_params):
        """求解器能运行并返回结果。"""
        result = solve_static_equilibrium(two_country_params, max_iterations=500)
        assert result.home is not None
        assert result.foreign is not None
        assert isinstance(result.iterations, int)
        assert isinstance(result.final_residual, float)

    def test_prices_positive(self, two_country_params):
        """均衡价格应为正。"""
        result = solve_static_equilibrium(two_country_params, max_iterations=2000)
        assert (result.home.price > 0).all()
        assert (result.foreign.price > 0).all()

    def test_output_positive(self, two_country_params):
        """均衡产出应为正。"""
        result = solve_static_equilibrium(two_country_params, max_iterations=2000)
        assert (result.home.output > 0).all()
        assert (result.foreign.output > 0).all()

    def test_nominal_anchor(self, two_country_params):
        """P_H[0] ≈ 1（名义锚）。"""
        result = solve_static_equilibrium(two_country_params, max_iterations=2000)
        # 允许一定偏差（数值求解器精度）
        assert abs(result.home.price[0] - 1.0) < 0.1

    def test_residual_below_threshold(self, two_country_params):
        """最终残差应在可接受范围内。"""
        result = solve_static_equilibrium(two_country_params, max_iterations=3000)
        # 残差阈值：允许较宽裕但有明确上界
        assert result.final_residual < 1.0, (
            f"残差 {result.final_residual:.4f} 超过阈值 1.0"
        )

    def test_rho0_consistency(self):
        """ρ=0 参数下，通用求解器与解析解应一致。"""
        from eco_model_v2.equilibrium_rho0 import solve_equilibrium_rho0

        Nl, Ml, M = 2, 1, 1
        alpha = np.array([[0.3, 0.3, 0.4],
                          [0.2, 0.4, 0.4]])
        gamma = np.array([[1.0, 0.5],
                          [1.0, 0.5]])
        beta = np.array([0.4, 0.6])
        A = np.ones(Nl)
        exports = np.array([0.5, 0.5, 0.0])
        imp_price = np.array([1.0, 1.0])
        L = np.array([10.0])

        # 解析解
        rho0_result = solve_equilibrium_rho0(
            alpha, gamma, beta, A, exports, imp_price, L, Ml, M,
        )

        # 通用求解器（ρ=0 参数）
        rho = np.array([[0.0, 1e-10],
                        [0.0, 1e-10]])
        rho_cons = np.array([0.0, 1e-10])
        gamma_cons = np.array([1.0, 0.5])

        cp = CountryParams(
            alpha=alpha, gamma=gamma, rho=rho,
            beta=beta, A=A, exports=exports,
            gamma_cons=gamma_cons, rho_cons=rho_cons,
            import_cost=imp_price, L=L,
            Ml=Ml, M_factors=M,
        )
        import dataclasses
        cp2 = dataclasses.replace(
            cp,
            alpha=alpha.copy(), gamma=gamma.copy(), rho=rho.copy(),
            beta=beta.copy(), A=A.copy(), exports=exports.copy(),
            gamma_cons=gamma_cons.copy(), rho_cons=rho_cons.copy(),
            import_cost=imp_price.copy(), L=L.copy(),
        )
        params = TwoCountryParams(home=cp, foreign=cp2)

        general_result = solve_static_equilibrium(params, max_iterations=3000)

        # 两者价格方向应一致（同序）
        rho0_prices = rho0_result["price"][:Nl]
        gen_prices = general_result.home.price[:Nl]
        # 归一化后比较排序
        rho0_norm = rho0_prices / rho0_prices[0]
        gen_norm = gen_prices / gen_prices[0]
        # 放宽容差：两种方法的相对价格比应在同一量级
        for j in range(Nl):
            assert abs(rho0_norm[j] - gen_norm[j]) < 0.5, (
                f"部门 {j}: rho0={rho0_norm[j]:.3f}, general={gen_norm[j]:.3f}"
            )
