import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N                = 50
NUM_TRIALS       = 15
MU_VALUES        = [0.5, 1.0, 2.0]
CHECKPOINT_ITERS = [10, 25, 50, 100, 150, 200, 300, 500]
OUTPUT_DIR       = "output"

from src.graph_generation import generate_points
from src.physarum import build_physarum_with_tracking
from src.experiment import save_results
from src.visualization import plot_convergence


def main():
    t_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    max_iters   = max(CHECKPOINT_ITERS)
    total_runs  = NUM_TRIALS * len(MU_VALUES)
    runs_done   = 0

    print("=" * 65)
    print("  Experiment 4: convergence study (standalone, debug mode)")
    print(f"  n={N}  trials={NUM_TRIALS}  mu values={MU_VALUES}")
    print(f"  checkpoints={CHECKPOINT_ITERS}  (max iter={max_iters})")
    print(f"  total simulations: {total_runs}")
    print(f"  total checkpoint rows: {total_runs * len(CHECKPOINT_ITERS)}")
    print("=" * 65)

    records      = []
    trial_times  = []

    for trial in range(NUM_TRIALS):
        t_trial = time.time()
        seed    = trial * 5000

        print(f"\n── trial {trial+1}/{NUM_TRIALS}  (seed={seed}) ──────────────────────")

        print(f"  [points]  generating {N} pts ... ", end="", flush=True)
        t = time.time()
        pts = generate_points(N, seed=seed)
        print(f"done ({time.time()-t:.2f}s)  "
              f"x∈[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]  "
              f"y∈[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]")

        for mu in MU_VALUES:
            t_mu  = time.time()
            runs_done += 1
            pct   = 100 * runs_done / total_runs

            print(f"  [mu={mu}]  running {max_iters} iters with "
                  f"{len(CHECKPOINT_ITERS)} checkpoints ... ", end="", flush=True)

            try:
                _, checkpoints = build_physarum_with_tracking(
                    pts,
                    num_iters=max_iters,
                    mu=mu,
                    threshold=0.05,
                    seed=seed,
                    checkpoint_iters=CHECKPOINT_ITERS,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                print(f"         skipping (mu={mu}, trial={trial})")
                continue

            elapsed_mu = time.time() - t_mu
            print(f"done ({elapsed_mu:.2f}s)  got {len(checkpoints)} checkpoints")

            print(f"         {'iter':>6}  {'edges':>6}  {'entropy':>9}")
            print(f"         {'----':>6}  {'-----':>6}  {'-------':>9}")
            for cp in checkpoints:
                iteration = cp["iteration"]
                n_edges   = cp["n_surviving_edges"]
                entropy   = cp["conductivity_entropy"]
                print(f"         {iteration:>6}  {n_edges:>6}  {entropy:>9.4f}")
                records.append({
                    "trial":     trial,
                    "mu":        mu,
                    "n":         N,
                    **cp,
                })

            if len(checkpoints) >= 2:
                first_edges = checkpoints[0]["n_surviving_edges"]
                last_edges  = checkpoints[-1]["n_surviving_edges"]
                first_h     = checkpoints[0]["conductivity_entropy"]
                last_h      = checkpoints[-1]["conductivity_entropy"]
                delta_edges = last_edges - first_edges
                delta_h     = last_h - first_h
                print(f"         Δedges={delta_edges:+d} over experiment  "
                      f"Δentropy={delta_h:+.4f}")
                if abs(delta_edges) <= 1 and abs(delta_h) < 0.01:
                    print(f"         [converged by iter {CHECKPOINT_ITERS[-2]}]")
                else:
                    print(f"         [still evolving at iter {CHECKPOINT_ITERS[-1]}]")

            elapsed_total = time.time() - t_total
            if runs_done > 1:
                eta = elapsed_total / runs_done * (total_runs - runs_done)
                eta_str = f"  ETA {eta:.0f}s"
            else:
                eta_str = ""
            print(f"         [{runs_done}/{total_runs}  {pct:.0f}%  "
                  f"wall={elapsed_total:.0f}s{eta_str}]")

        trial_elapsed = time.time() - t_trial
        trial_times.append(trial_elapsed)
        avg_trial       = np.mean(trial_times)
        remaining       = NUM_TRIALS - (trial + 1)
        eta_trials      = avg_trial * remaining
        print(f"\n  trial {trial+1} done in {trial_elapsed:.1f}s  "
              f"(avg={avg_trial:.1f}s/trial  "
              f"ETA {eta_trials:.0f}s = {eta_trials/60:.1f} min remaining)")

    print(f"\n{'='*65}")
    print(f"  Building DataFrame from {len(records)} records ...")
    conv_df = pd.DataFrame(records)
    print(f"  Shape: {conv_df.shape}")
    print(f"  Columns: {list(conv_df.columns)}")
    print(f"  mu values seen: {sorted(conv_df['mu'].unique())}")
    print(f"  iterations seen: {sorted(conv_df['iteration'].unique())}")
    print(f"  Missing values:\n{conv_df.isnull().sum()[conv_df.isnull().sum() > 0]}")
    if conv_df.isnull().sum().sum() == 0:
        print(f"  (no missing values)")

    print(f"\n  Per-(mu, iteration) summary (mean ± std over {NUM_TRIALS} trials):")
    print(f"  {'mu':>5}  {'iter':>6}  {'edges (mean±std)':>20}  {'entropy (mean±std)':>22}")
    print(f"  {'--':>5}  {'----':>6}  {'----------------':>20}  {'------------------':>22}")
    for mu in MU_VALUES:
        for it in CHECKPOINT_ITERS:
            sub = conv_df[(conv_df["mu"] == mu) & (conv_df["iteration"] == it)]
            if len(sub) == 0:
                continue
            e_mean = sub["n_surviving_edges"].mean()
            e_std  = sub["n_surviving_edges"].std()
            h_mean = sub["conductivity_entropy"].mean()
            h_std  = sub["conductivity_entropy"].std()
            print(f"  {mu:>5}  {it:>6}  "
                  f"{e_mean:>7.1f} ± {e_std:>5.1f}          "
                  f"{h_mean:>7.4f} ± {h_std:>6.4f}")
        print()

    print(f"  Saving results ...")
    t = time.time()
    summary = save_results(conv_df, "convergence", OUTPUT_DIR)
    print(f"  Saved in {time.time()-t:.2f}s")
    print(f"    output/convergence_raw.csv     ({len(conv_df)} rows)")
    print(f"    output/convergence_summary.csv ({len(summary)} rows)")

    print(f"\n  Generating plots ...")
    t = time.time()
    try:
        plot_convergence(conv_df, OUTPUT_DIR)
        print(f"  plot_convergence done ({time.time()-t:.2f}s)")
        print(f"    output/convergence_study.png")
    except Exception as e:
        print(f"  plot_convergence ERROR: {e}")
        traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 65)


if __name__ == "__main__":
    main()
