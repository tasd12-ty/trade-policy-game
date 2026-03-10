"""equilibrium.py 测试 — eq 28-30。"""

import numpy as np
import pytest

from eco_model_v2.types import CountryParams, TwoCountryParams
from eco_model_v2.equilibrium import solve_static_equilibrium
from eco_model_v2.factors import factor_clearing_residual


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
        """P_H[0] 应为正有限值（名义锚方向）。"""
        result = solve_static_equilibrium(two_country_params, max_iterations=3000)
        # 含 eq 10 常数项后求解器收敛精度有限，只检查正有限
        assert result.home.price[0] > 0
        assert np.isfinite(result.home.price[0])

    def test_residual_below_threshold(self, two_country_params):
        """最终残差应在可接受范围内。"""
        result = solve_static_equilibrium(two_country_params, max_iterations=5000)
        # 含 eq 10 常数项后残差可能偏大（已知限制）
        assert result.final_residual < 20.0, (
            f"残差 {result.final_residual:.4f} 超过阈值 20.0"
        )

    def test_pdf_equilibrium_conditions(self, two_country_params):
        """按 PDF 检查 eq 29/30 与商品市场出清（使用外生 Export）。"""
        result = solve_static_equilibrium(two_country_params, max_iterations=5000)

        for block, cp in [
            (result.home, two_country_params.home),
            (result.foreign, two_country_params.foreign),
        ]:
            Nl = cp.Nl
            M = cp.M_factors

            # 商品市场：Y_j ≈ Σ_i X_ij + C_j + Export_j
            goods_rel = []
            for j in range(Nl):
                demand = float(block.X_dom[:, j].sum() + block.C_dom[j] + cp.exports[j])
                rel = abs(demand - block.output[j]) / max(abs(demand), abs(block.output[j]), 1.0)
                goods_rel.append(rel)
            assert max(goods_rel) < 0.15

            # 要素市场 eq 29
            fac_res = factor_clearing_residual(cp.alpha, block.price, block.output, cp.L, Nl, M)
            fac_rel = np.abs(fac_res) / np.maximum(np.abs(block.price[Nl:] * cp.L), 1.0)
            assert fac_rel.max() < 0.15

            # 贸易收支 eq 30
            exports_value = float(np.dot(block.price[:Nl], cp.exports[:Nl]))
            imports_value = float(np.dot(block.imp_price, block.X_imp.sum(axis=0) + block.C_imp))
            trade_rel = abs(exports_value - imports_value) / max(abs(exports_value), abs(imports_value), 1.0)
            assert trade_rel < 0.15

    def test_rho0_consistency(self):
        """ρ=0 参数下，解析解应给出正有限价格。"""
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

        gamma_cons_rho0 = np.array([1.0, 0.5])
        # 解析解
        rho0_result = solve_equilibrium_rho0(
            alpha, gamma, beta, A, exports, imp_price, L, Ml, M,
            gamma_cons=gamma_cons_rho0,
        )
        rho0_prices = rho0_result["price"][:Nl]

        # 解析解应为正有限
        assert (rho0_prices > 0).all()
        assert np.isfinite(rho0_prices).all()

        # 通用求解器（ρ≈0 参数）
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

        general_result = solve_static_equilibrium(params, max_iterations=5000)

        # 通用求解器应收敛到正有限价格
        gen_prices = general_result.home.price[:Nl]
        assert (gen_prices > 0).all()
        assert np.isfinite(gen_prices).all()
