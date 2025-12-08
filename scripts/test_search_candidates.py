
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eco_simu.agent_loop.policies.search import SearchPolicyAdapter

def test_cartesian_candidates():
    print("Testing Cartesian Candidate Generation...")
    
    # Setup: 1 sector, small grids
    # Tariff: 0.0 (neutral), 0.1
    # Quota: 1.0 (neutral), 0.8
    adapter = SearchPolicyAdapter(
        n_sectors=1,
        tariff_grid=[0.0, 0.1],
        quota_grid=[1.0, 0.8],
        target_sectors=[0],
    )
    
    # Mock actor and targets
    actor = "H"
    targets = [0]
    
    candidates = adapter._build_candidates(actor, targets)
    
    print(f"Generated {len(candidates)} candidates.")
    
    # Expected:
    # 1. No-op (initial)
    # 2. Tariff=0.0, Quota=1.0 -> No-op (duplicate, removed)
    # 3. Tariff=0.0, Quota=0.8 -> Pure Quota
    # 4. Tariff=0.1, Quota=1.0 -> Pure Tariff
    # 5. Tariff=0.1, Quota=0.8 -> Combined
    # Total unique should be 4: No-op, Pure Quota, Pure Tariff, Combined.
    
    for i, cand in enumerate(candidates):
        t = cand.get("import_tariff", {})
        q = cand.get("export_quota", {})
        print(f"Candidate {i}: Tariff={t}, Quota={q}")

    # Verification
    assert len(candidates) == 4, f"Expected 4 candidates, got {len(candidates)}"
    print("SUCCESS: Candidate count matches expectation.")

if __name__ == "__main__":
    test_cartesian_candidates()
