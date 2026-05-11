import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N          = 50
NUM_TRIALS = 30
MU_VALUES  = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
NUM_ITERS  = 500
OUTPUT_DIR = "output"

from src.graph_generation import generate_points, build_mst
from src.physarum import build_physarum_graph
from src.metrics import compute_all_metrics, compute_cost
from src.experiment import save_results
from src.visualization import plot_mu_sweep
from src.statistics_utils import spearman_bootstrap_ci


def _fmt(d: dict) -> str:
    keys = ["cost", "steiner_ratio", "algebraic_connectivity",
            "robustness", "cyclomatic_number", "network_entropy"]
    parts = []
    for k in keys:
        if k in d:
            short = {"cost": "cost", "steiner_ratio": "ρ",
                     "algebraic_connectivity": "λ₂", "robustness": "R",
                     "cyclomatic_number": "γ", "network_entropy": "H"}[k]
            parts.append(f"{short}={d[k]:.3f}")
    return "  ".join(parts)


def main():
    t_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  Experiment 3: mu sweep (standalone, debug mode)")
    print(f"  n={N}  trials={NUM_TRIALS}  iters={NUM_ITERS}")
    print(f"  mu values: {MU_VALUES}")
    print(f"  total runs: {NUM_TRIALS * len(MU_VALUES)}")
    print("=" * 65)

    records = []
    trial_times = []
    total_runs  = NUM_TRIALS * len(MU_VALUES)
    runs_done   = 0

    for trial in range(NUM_TRIALS):
        t_trial = time.time()
        seed = trial * 1000

        print(f"\n── trial {trial+1}/{NUM_TRIALS}  (seed={seed}) ──────────────────────")

        print(f"  [points]    generating {N} pts ... ", end="", flush=True)
        t = time.time()
        pts = generate_points(N, seed=seed)
        print(f"done ({time.time()-t:.2f}s)  "
              f"x∈[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]  "
              f"y∈[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]")

        print(f"  [MST]       building baseline ... ", end="", flush=True)
        t = time.time()
        mst      = build_mst(pts)
        mst_cost = compute_cost(mst)
        print(f"done ({time.time()-t:.2f}s)  cost={mst_cost:.4f}  "
              f"edges={mst.number_of_edges()}")

        for mu_idx, mu in enumerate(MU_VALUES):
            t_mu = time.time()
            runs_done += 1
            pct = 100 * runs_done / total_runs

            print(f"  [mu={mu:<5}]  building Physarum ... ", end="", flush=True)

            try:
                G = build_physarum_graph(
                    pts,
                    num_iters=NUM_ITERS,
                    mu=mu,
                    threshold=0.05,
                    seed=seed,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                print(f"           skipping (mu={mu}, trial={trial})")
                continue

            elapsed_mu = time.time() - t_mu
            print(f"done ({elapsed_mu:.2f}s)  "
                  f"edges={G.number_of_edges()}  nodes={G.number_of_nodes()}")

            print(f"           computing metrics ... ", end="", flush=True)
            t = time.time()
            try:
                metrics = compute_all_metrics(G, mst_cost=mst_cost)
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                continue
            print(f"done ({time.time()-t:.2f}s)")
            print(f"           {_fmt(metrics)}")

            row = {"trial": trial, "mu": mu, "n": N}
            row.update(metrics)
            records.append(row)

            elapsed_total = time.time() - t_total
            if runs_done > 1:
                eta = elapsed_total / runs_done * (total_runs - runs_done)
                eta_str = f"  ETA {eta:.0f}s"
            else:
                eta_str = ""
            print(f"           [{runs_done}/{total_runs}  {pct:.0f}%  "
                  f"wall={elapsed_total:.0f}s{eta_str}]")

        trial_elapsed = time.time() - t_trial
        trial_times.append(trial_elapsed)
        avg_trial = np.mean(trial_times)
        remaining_trials = NUM_TRIALS - (trial + 1)
        eta_trials = avg_trial * remaining_trials
        print(f"\n  trial {trial+1} done in {trial_elapsed:.1f}s  "
              f"(avg={avg_trial:.1f}s/trial  "
              f"ETA {eta_trials:.0f}s = {eta_trials/60:.1f} min remaining)")

    print(f"\n{'='*65}")
    print(f"  Building DataFrame from {len(records)} records ...")
    mu_df = pd.DataFrame(records)
    print(f"  Shape: {mu_df.shape}")
    print(f"  Columns: {list(mu_df.columns)}")
    print(f"  mu values seen: {sorted(mu_df['mu'].unique())}")
    print(f"  Missing values:\n{mu_df.isnull().sum()[mu_df.isnull().sum() > 0]}")

    print(f"\n  Per-mu summary (mean ± std):")
    summary_metrics = ["cost", "steiner_ratio", "algebraic_connectivity",
                       "robustness", "cyclomatic_number"]
    for mu in MU_VALUES:
        sub = mu_df[mu_df["mu"] == mu]
        row_str = f"  mu={mu:<5}"
        for m in summary_metrics:
            if m in sub.columns:
                short = {"cost": "cost", "steiner_ratio": "ρ",
                         "algebraic_connectivity": "λ₂",
                         "robustness": "R", "cyclomatic_number": "γ"}[m]
                row_str += f"  {short}={sub[m].mean():.3f}±{sub[m].std():.3f}"
        print(row_str)

    print(f"\n  Spearman ρ(mu, metric)  [95% bootstrap CI, 5000 resamples]:")
    for metric in ["cost", "steiner_ratio", "algebraic_connectivity",
                   "robustness", "cyclomatic_number", "network_entropy"]:
        if metric not in mu_df.columns:
            print(f"    {metric:30s}: NOT IN DATAFRAME — skipping")
            continue
        x = mu_df["mu"].values
        y = mu_df[metric].dropna().values
        if len(y) < len(x):
            mask = ~mu_df[metric].isna()
            x = mu_df.loc[mask, "mu"].values
        try:
            rho, lo, hi = spearman_bootstrap_ci(x, y, n_bootstrap=5_000, seed=7)
            sig = "***" if (lo > 0 or hi < 0) else "ns"
            print(f"    {metric:30s}: ρ={rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}")
        except Exception as e:
            print(f"    {metric:30s}: ERROR — {e}")

    print(f"\n  Saving results ...")
    t = time.time()
    summary = save_results(mu_df, "mu_sweep", OUTPUT_DIR)
    print(f"  Saved in {time.time()-t:.2f}s")
    print(f"    output/mu_sweep_raw.csv       ({len(mu_df)} rows)")
    print(f"    output/mu_sweep_summary.csv   ({len(summary)} rows)")

    print(f"\n  Generating plots ...")
    t = time.time()
    try:
        plot_mu_sweep(mu_df, OUTPUT_DIR)
        print(f"  plot_mu_sweep done ({time.time()-t:.2f}s)")
    except Exception as e:
        print(f"  plot_mu_sweep ERROR: {e}")
        traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Key outputs:")
    print(f"    output/mu_sweep_raw.csv")
    print(f"    output/mu_sweep_summary.csv")
    print(f"    output/mu_sweep_metrics.png")
    print(f"    output/mu_phase_diagram.png")
    print(f"    output/mu_lambda2_per_cost.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
