import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

N          = 50
NUM_TRIALS = 30
MU_VALUES  = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
NUM_ITERS  = 800
BATCH_SIZE = 50
OUTPUT_DIR = "output"

from src.graph_generation import generate_points, build_mst
from src.physarum import build_physarum_demand_weighted
from src.metrics import compute_all_metrics, compute_cost

os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []
total = NUM_TRIALS * len(MU_VALUES) * 2
done  = 0
t0    = time.time()

for trial in range(NUM_TRIALS):
    seed = N * 1000 + trial
    pts  = generate_points(N, seed=seed)
    mst_cost = compute_cost(build_mst(pts))

    for mu in MU_VALUES:
        for demand in ("fixed", "uniform"):
            done += 1
            G = build_physarum_demand_weighted(
                pts, num_iters=NUM_ITERS, mu=mu,
                threshold=0.05, seed=seed,
                demand=demand, batch_size=BATCH_SIZE,
            )
            row = {"trial": trial, "mu": mu, "demand": demand, "n": N}
            row.update(compute_all_metrics(G, mst_cost=mst_cost))
            records.append(row)
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"[{done}/{total}] trial={trial} mu={mu} demand={demand}  "
                  f"ρ={row['steiner_ratio']:.3f} γ={row['cyclomatic_number']:.1f}  "
                  f"ETA {eta:.0f}s")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "mu_sweep_both_raw.csv"), index=False)
print(f"\nSaved mu_sweep_both_raw.csv  ({len(df)} rows)")

COLORS = {"fixed": "#1f77b4", "uniform": "#ff7f0e"}
LABELS = {"fixed": "Fixed demand", "uniform": "Uniform demand"}
METRICS = [
    ("steiner_ratio",          "Steiner ratio ρ"),
    ("algebraic_connectivity", "Algebraic connectivity λ₂"),
    ("robustness",             "Node-removal robustness"),
    ("cyclomatic_number",      "Cyclomatic number γ"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(
    "Effect of Reinforcement Exponent μ on Fixed vs. Uniform Demand  (n=50, 30 trials)",
    fontsize=11, fontweight="bold",
)

for ax, (metric, label) in zip(axes, METRICS):
    for demand in ("fixed", "uniform"):
        sub = df[df["demand"] == demand]
        grp = sub.groupby("mu")[metric].agg(mean="mean", std="std", count="count").reset_index()
        se  = grp["std"] / np.sqrt(grp["count"])
        ax.plot(grp["mu"], grp["mean"], "o-", color=COLORS[demand],
                label=LABELS[demand], linewidth=2, markersize=5)
        ax.fill_between(grp["mu"], grp["mean"] - 1.96*se, grp["mean"] + 1.96*se,
                        alpha=0.15, color=COLORS[demand])
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("μ")
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=8)

fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "mu_sweep_both.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
