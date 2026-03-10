from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_model.armington import armington_price, armington_quantity, armington_share


class TestArmington(unittest.TestCase):
    def test_share_bounds(self) -> None:
        share = armington_share(gamma=0.4, p_dom=1.2, p_for=0.9, rho=0.2)
        self.assertGreaterEqual(float(share), 0.0)
        self.assertLessEqual(float(share), 1.0)

    def test_perfect_substitute_limit(self) -> None:
        # rho -> 1: 低价方应占更高份额
        share = armington_share(gamma=0.5, p_dom=0.8, p_for=1.2, rho=0.999999)
        self.assertGreater(float(share), 0.9)

    def test_sigma_one_geometric_price(self) -> None:
        # rho=0 => sigma=1，对偶价格退化为几何平均
        p = armington_price(gamma=0.3, p_dom=2.0, p_for=8.0, rho=0.0)
        expected = np.exp(0.3 * np.log(2.0) + 0.7 * np.log(8.0))
        self.assertAlmostEqual(float(p), float(expected), places=10)

    def test_quantity_cobb_case(self) -> None:
        q = armington_quantity(gamma=0.25, x_dom=2.0, x_for=5.0, alpha=0.6, rho=0.0)
        expected = np.exp(0.6 * (0.25 * np.log(2.0) + 0.75 * np.log(5.0)))
        self.assertAlmostEqual(float(q), float(expected), places=10)


if __name__ == "__main__":
    unittest.main()
