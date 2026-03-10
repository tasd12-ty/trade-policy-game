from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_model.pipeline import bootstrap_simulator
from project_model.presets import create_symmetric_params


class TestDynamics(unittest.TestCase):
    def test_run_without_nan(self) -> None:
        params = create_symmetric_params(n=6)
        sim = bootstrap_simulator(params, tau_price=0.08, normalize_gap_by_supply=True)
        sim.run(10)

        for c in ("H", "F"):
            for s in sim.history[c]:
                self.assertTrue(np.all(np.isfinite(s.price)))
                self.assertTrue(np.all(np.isfinite(s.output)))
                self.assertGreater(float(np.min(s.price)), 0.0)

    def test_policy_interface(self) -> None:
        params = create_symmetric_params(n=6)
        sim = bootstrap_simulator(params, tau_price=0.08, normalize_gap_by_supply=True)

        old = sim.home_import_multiplier.copy()
        sim.apply_import_tariff("H", {2: 0.25})
        self.assertGreater(sim.home_import_multiplier[2], old[2])

        sim.apply_export_control("F", {3: 0.4})
        self.assertAlmostEqual(float(sim.export_multiplier["F"][3]), 0.4, places=10)


if __name__ == "__main__":
    unittest.main()
