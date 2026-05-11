import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from .graph_generation import generate_points, build_mst, build_knn_graph, build_rgg
from .physarum import (
    build_physarum_graph,
    build_physarum_demand_weighted,
)
from .metrics import compute_all_metrics, compute_cost

METHOD_ORDER = ["MST", "kNN", "RGG", "Physarum"]

_PHY_ITERS     = 500
_PHY_THRESHOLD = 0.05
_PHY_DT        = 0.1
_DEM_ITERS     = 800
_DEM_BATCH     = 50


def _record(n, trial, method, G, mst_cost, extra=None):
    row = {"n": n, "trial": trial, "method": method}
    row.update(compute_all_metrics(G, mst_cost=mst_cost))
    if extra:
        row.update(extra)
    return row


def run_main_comparison(
    n_values: Optional[List[int]] = None,
    num_trials: int = 30,
) -> pd.DataFrame:
    if n_values is None:
        n_values = [20, 50, 100]

    records: List[dict] = []

    for n in n_values:
        print(f"\n{'='*55}\nBaseline comparison  n={n}  ({num_trials} trials)\n{'='*55}")
        t0 = time.time()
        for trial in range(num_trials):
            seed = n * 1000 + trial
            pts  = generate_points(n, seed=seed)

            mst      = build_mst(pts)
            mst_cost = compute_cost(mst)

            graphs = {
                "MST":      mst,
                "kNN":      build_knn_graph(pts, k=3),
                "RGG":      build_rgg(pts, r=0.3),
                "Physarum": build_physarum_graph(
                    pts, num_iters=_PHY_ITERS, mu=1.0,
                    threshold=_PHY_THRESHOLD, seed=seed,
                ),
            }

            for method in METHOD_ORDER:
                records.append(_record(n, trial, method, graphs[method], mst_cost))

            if (trial + 1) % 10 == 0:
                print(f"  trial {trial+1:>3}/{num_trials}  [{time.time()-t0:.1f}s]")

    return pd.DataFrame(records)


DEMAND_ORDER  = ["Fixed", "Uniform", "HubSpoke"]
DEMAND_LABELS = {
    "Fixed":    "Fixed pair (Tero)",
    "Uniform":  "Uniform demand",
    "HubSpoke": "Hub-spoke demand",
}

def run_demand_comparison(
    n_values: Optional[List[int]] = None,
    num_trials: int = 30,
    num_iters: int = _DEM_ITERS,
    batch_size: int = _DEM_BATCH,
) -> pd.DataFrame:
    if n_values is None:
        n_values = [20, 50, 100]

    records: List[dict] = []

    for n in n_values:
        print(
            f"\n{'='*55}\nDemand comparison  n={n}  ({num_trials} trials)"
            f"  iters={num_iters}  batch={batch_size}\n{'='*55}"
        )
        t0 = time.time()
        for trial in range(num_trials):
            seed = n * 1000 + trial
            pts  = generate_points(n, seed=seed)
            mst_cost = compute_cost(build_mst(pts))

            for label, dem in [
                ("Fixed",    "fixed"),
                ("Uniform",  "uniform"),
                ("HubSpoke", "hubspoke"),
            ]:
                G = build_physarum_demand_weighted(
                    pts, num_iters=num_iters, mu=1.0,
                    threshold=_PHY_THRESHOLD, seed=seed,
                    demand=dem, batch_size=batch_size,
                )
                row = {"n": n, "trial": trial, "demand": label}
                row.update(compute_all_metrics(G, mst_cost=mst_cost))
                records.append(row)

            if (trial + 1) % 10 == 0:
                print(f"  trial {trial+1:>3}/{num_trials}  [{time.time()-t0:.1f}s]")

    return pd.DataFrame(records)


def run_lambda_sweep(
    n: int = 50,
    num_trials: int = 30,
    lambda_values: Optional[List[float]] = None,
    num_iters: int = _DEM_ITERS,
    batch_size: int = _DEM_BATCH,
) -> pd.DataFrame:
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    print(
        f"\n{'='*55}\nLambda sweep  n={n}  {num_trials} trials  "
        f"lambda in {lambda_values}\n{'='*55}"
    )
    records: List[dict] = []
    t0 = time.time()

    for trial in range(num_trials):
        seed = trial * 1000
        pts  = generate_points(n, seed=seed)
        mst_cost = compute_cost(build_mst(pts))

        for lam in lambda_values:
            if lam == 0.0:
                G = build_physarum_demand_weighted(
                    pts, num_iters=num_iters, mu=1.0,
                    threshold=_PHY_THRESHOLD, seed=seed,
                    demand="fixed", batch_size=batch_size,
                )
            elif lam == 1.0:
                G = build_physarum_demand_weighted(
                    pts, num_iters=num_iters, mu=1.0,
                    threshold=_PHY_THRESHOLD, seed=seed,
                    demand="uniform", batch_size=batch_size,
                )
            else:
                G = build_physarum_demand_weighted(
                    pts, num_iters=num_iters, mu=1.0,
                    threshold=_PHY_THRESHOLD, seed=seed,
                    demand="mixed", batch_size=batch_size,
                    lambda_mix=lam,
                )
            row = {"trial": trial, "lambda": lam, "n": n}
            row.update(compute_all_metrics(G, mst_cost=mst_cost))
            records.append(row)

        if (trial + 1) % 10 == 0:
            print(f"  trial {trial+1:>3}/{num_trials}  [{time.time()-t0:.1f}s]")

    return pd.DataFrame(records)


def save_results(
    df: pd.DataFrame,
    name: str,
    output_dir: str = "output",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, f"{name}_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"  raw  -> {raw_path}  ({len(df)} rows)")

    if "demand" in df.columns and "n" in df.columns:
        group_cols = ["demand", "n"]
    elif "lambda" in df.columns:
        group_cols = ["lambda"]
    elif "method" in df.columns and "n" in df.columns:
        group_cols = ["method", "n"]
    else:
        group_cols = ["n"]

    skip = {"trial"} | set(group_cols)
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]

    summary = (
        df.groupby(group_cols)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, c)) for c in summary.columns]

    summary_path = os.path.join(output_dir, f"{name}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  summary -> {summary_path}  ({len(summary)} rows)")

    return summary
