"""supply_chain.py 测试 — eq 27。"""

import numpy as np
import pytest

from eco_model_v2.supply_chain import (
    SupplyChainNode,
    SupplyChainNetwork,
    identity_transform,
    markup_transform,
    bottleneck_transform,
    disruption_transform,
)


class TestSupplyChainNetwork:
    def test_identity_no_change(self):
        """恒等变换不改变价格。"""
        net = SupplyChainNetwork()
        src = SupplyChainNode("H", 1)
        tgt = SupplyChainNode("F", 1)
        net.add_edge(src, tgt, identity_transform)

        h_p = np.array([1.0, 2.0, 1.0])
        f_p = np.array([1.0, 1.5, 1.0])
        h_e = np.array([0.5, 1.0, 0.0])
        f_e = np.array([0.3, 0.8, 0.0])
        h_imp = np.array([1.1, 1.5])
        f_imp = np.array([1.2, 1.8])

        result = net.apply(h_p, f_p, h_e, f_e, h_imp, f_imp)
        # F 国部门 1 的进口价格应等于 H 国部门 1 的价格
        assert abs(result["foreign_imp_price"][1] - 2.0) < 1e-6

    def test_markup(self):
        """加成变换。"""
        net = SupplyChainNetwork()
        net.add_edge(
            SupplyChainNode("H", 1),
            SupplyChainNode("F", 1),
            markup_transform(0.2),
        )

        h_p = np.array([1.0, 2.0, 1.0])
        f_p = np.array([1.0, 1.5, 1.0])
        h_e = np.array([0.5, 1.0, 0.0])
        f_e = np.array([0.3, 0.8, 0.0])
        h_imp = np.array([1.1, 1.5])
        f_imp = np.array([1.2, 1.8])

        result = net.apply(h_p, f_p, h_e, f_e, h_imp, f_imp)
        expected = 2.0 * 1.2  # P * (1 + markup)
        assert abs(result["foreign_imp_price"][1] - expected) < 1e-6

    def test_disruption_reduces_quantity(self):
        """中断变换应减少出口量。"""
        net = SupplyChainNetwork()
        net.add_edge(
            SupplyChainNode("H", 1),
            SupplyChainNode("F", 1),
            disruption_transform(severity=0.3),
        )

        h_p = np.array([1.0, 2.0, 1.0])
        f_p = np.array([1.0, 1.5, 1.0])
        h_e = np.array([0.5, 1.0, 0.0])
        f_e = np.array([0.3, 0.8, 0.0])
        h_imp = np.array([1.1, 1.5])
        f_imp = np.array([1.2, 1.8])

        result = net.apply(h_p, f_p, h_e, f_e, h_imp, f_imp)
        # H 国部门 1 出口应减少 30%
        expected_export = 1.0 * 0.7
        assert abs(result["home_export"][1] - expected_export) < 1e-6

    def test_bottleneck_caps_quantity(self):
        """瓶颈变换限制数量。"""
        net = SupplyChainNetwork()
        net.add_edge(
            SupplyChainNode("H", 1),
            SupplyChainNode("F", 1),
            bottleneck_transform(capacity=0.5),
        )

        h_p = np.array([1.0, 2.0, 1.0])
        f_p = np.array([1.0, 1.5, 1.0])
        h_e = np.array([0.5, 1.0, 0.0])  # sector 1 export = 1.0 > capacity 0.5
        f_e = np.array([0.3, 0.8, 0.0])
        h_imp = np.array([1.1, 1.5])
        f_imp = np.array([1.2, 1.8])

        result = net.apply(h_p, f_p, h_e, f_e, h_imp, f_imp)
        assert abs(result["home_export"][1] - 0.5) < 1e-6

    def test_apply_does_not_mutate_inputs(self):
        """apply() 不应修改输入数组。"""
        net = SupplyChainNetwork()
        net.add_edge(
            SupplyChainNode("H", 1),
            SupplyChainNode("F", 1),
            disruption_transform(severity=0.5),
        )

        h_p = np.array([1.0, 2.0, 1.0])
        f_p = np.array([1.0, 1.5, 1.0])
        h_e = np.array([0.5, 1.0, 0.0])
        f_e = np.array([0.3, 0.8, 0.0])
        h_imp = np.array([1.1, 1.5])
        f_imp = np.array([1.2, 1.8])

        h_e_orig = h_e.copy()
        f_e_orig = f_e.copy()
        h_imp_orig = h_imp.copy()
        f_imp_orig = f_imp.copy()

        net.apply(h_p, f_p, h_e, f_e, h_imp, f_imp)

        np.testing.assert_array_equal(h_e, h_e_orig, "h_e was mutated")
        np.testing.assert_array_equal(f_e, f_e_orig, "f_e was mutated")
        np.testing.assert_array_equal(h_imp, h_imp_orig, "h_imp was mutated")
        np.testing.assert_array_equal(f_imp, f_imp_orig, "f_imp was mutated")
