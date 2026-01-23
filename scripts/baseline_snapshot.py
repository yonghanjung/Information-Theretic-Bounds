"""
Deterministic baseline snapshot for fBound.

Purpose:
- Establish a stable reference output before AGENTS alignment edits.
- This is NOT a correctness test.
- This MUST NOT change estimator behavior.
"""

import json
import numpy as np

# ---- FIX SEEDS (critical) ----
np.random.seed(123)

def main():
    # TODO: adjust these imports to match your current layout
    # Example (change as needed):
    # from fBound.causal_bound import run_bound
    # from fBound.data_generating import make_toy_data

    # ---- Minimal data ----
    n = 50  # keep tiny and fast
    # X, A, Y = make_toy_data(n=n, seed=123)

    # ---- Run estimator ----
    # result = run_bound(X=X, A=A, Y=Y)

    # ---- Extract ONLY stable, scalar-ish outputs ----
    snapshot = {
        # Examples (adapt to your result object):
        # "upper_mean": float(np.nanmean(result.upper)),
        # "lower_mean": float(np.nanmean(result.lower)),
        # "upper_first5": np.asarray(result.upper[:5]).tolist(),
        # "lower_first5": np.asarray(result.lower[:5]).tolist(),
    }

    with open("artifacts/baseline_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print("Baseline snapshot written to artifacts/baseline_snapshot.json")

if __name__ == "__main__":
    main()
