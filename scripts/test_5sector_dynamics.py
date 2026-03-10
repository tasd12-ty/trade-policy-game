"""Test eco_model_v2 dynamics with real 5-sector IO data.

Loads CHN/USA parameters via compat bridge, normalizes scales,
and runs a dynamics simulation to verify stability.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
from eco_model_v2.compat import from_io_params_two_country
from eco_model_v2.types import CountryParams, TwoCountryParams
from eco_model_v2.simulator import TwoCountrySimulator


def normalize_country_params(cp: CountryParams, target_income: float = 1.0) -> CountryParams:
    """Normalize country params so that total factor income = target_income.

    Scales L, exports (quantities) while preserving ratios (alpha, beta, gamma).
    """
    current_income = sum(cp.L)  # w=r=1 assumed
    scale = target_income / current_income

    return CountryParams(
        alpha=cp.alpha,
        gamma=cp.gamma,
        rho=cp.rho,
        beta=cp.beta,
        A=cp.A,
        exports=cp.exports * scale,
        gamma_cons=cp.gamma_cons,
        rho_cons=cp.rho_cons,
        import_cost=cp.import_cost,
        L=cp.L * scale,
        Ml=cp.Ml,
        M_factors=cp.M_factors,
    )


def main():
    # ---- Load raw params ----
    params_raw = from_io_params_two_country(
        home_dirs=(
            "io_params/outputs/category_a_5s/CHN2017",
            "io_params/outputs/category_b_5s/CHN2017",
            "io_params/outputs/category_c_5s/CHN2017",
        ),
        foreign_dirs=(
            "io_params/outputs/category_a_5s/USA2017",
            "io_params/outputs/category_b_5s/USA2017",
            "io_params/outputs/category_c_5s/USA2017",
        ),
        Ml=0,  # all 5 sectors are tradable
        M_factors=2,
    )

    print("Raw CHN income:", sum(params_raw.home.L))
    print("Raw USA income:", sum(params_raw.foreign.L))

    # ---- Normalize both to income=1.0 ----
    home_norm = normalize_country_params(params_raw.home, target_income=1.0)
    foreign_norm = normalize_country_params(params_raw.foreign, target_income=1.0)
    params = TwoCountryParams(home=home_norm, foreign=foreign_norm)

    print(f"\nNormalized CHN: L={params.home.L}, exports={params.home.exports[:5]}")
    print(f"Normalized USA: L={params.foreign.L}, exports={params.foreign.exports[:5]}")
    print(f"CHN export/income: {params.home.exports[:5].sum():.4f}")
    print(f"USA export/income: {params.foreign.exports[:5].sum():.4f}")

    # ---- Create simulator ----
    sim = TwoCountrySimulator(
        params,
        tau=0.1,
        normalize_gap=False,
        numeraire=True,
    )
    sim.initialize()

    # ---- Run dynamics ----
    print("\n--- Running 200-step dynamics ---")
    for epoch in range(20):
        sim.run(10)
        sh = sim.history["H"][-1]
        sf = sim.history["F"][-1]
        print(
            f"  t={sim.t:4d}: "
            f"CHN price=[{sh.price[0]:.4f},{sh.price[1]:.4f},{sh.price[2]:.4f},{sh.price[3]:.4f},{sh.price[4]:.4f}] "
            f"inc={sh.income:.4f}  "
            f"USA price=[{sf.price[0]:.4f},{sf.price[1]:.4f},{sf.price[2]:.4f},{sf.price[3]:.4f},{sf.price[4]:.4f}] "
            f"inc={sf.income:.4f}"
        )

    # ---- Check stability ----
    print("\n--- Stability Check ---")
    sh = sim.history["H"][-1]
    sf = sim.history["F"][-1]

    for label, state, cp in [("CHN", sh, params.home), ("USA", sf, params.foreign)]:
        Nl = cp.Nl
        print(f"\n  {label}:")
        print(f"    prices:       {state.price}")
        print(f"    output[:Nl]:  {state.output[:Nl]}")
        print(f"    income:       {state.income:.6f}")
        print(f"    export_actual:{state.export_actual[:Nl]}")

        # Check for NaN/Inf
        has_nan = np.any(np.isnan(state.price)) or np.any(np.isnan(state.output))
        has_inf = np.any(np.isinf(state.price)) or np.any(np.isinf(state.output))
        price_range = state.price.max() / max(state.price.min(), 1e-30)
        print(f"    NaN: {has_nan}, Inf: {has_inf}")
        print(f"    price range: {state.price.min():.6f} — {state.price.max():.6f} (ratio={price_range:.1f})")

    # Check convergence: are prices still changing?
    if len(sim.history["H"]) >= 3:
        p_prev = sim.history["H"][-2].price
        p_curr = sim.history["H"][-1].price
        delta = np.abs(p_curr - p_prev) / np.maximum(np.abs(p_curr), 1e-10)
        print(f"\n  Price change (last step): max_rel_delta = {delta.max():.6e}")
        if delta.max() < 1e-4:
            print("  CONVERGED (relative price change < 1e-4)")
        elif delta.max() < 1e-2:
            print("  NEARLY CONVERGED (relative price change < 1e-2)")
        else:
            print("  NOT YET CONVERGED")


if __name__ == "__main__":
    main()
