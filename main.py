import os
import time
import warnings
import numpy as np

from src.graph_generation import generate_points, build_mst, build_knn_graph, build_rgg
from src.physarum import build_physarum_graph, build_physarum_demand_weighted, _hub_node
from src.metrics import compute_all_metrics, compute_cost
from src.experiment import (
    run_main_comparison,
    run_demand_comparison,
    run_lambda_sweep,
    save_results,
)
from src.visualization import (
    plot_main_comparison,
    plot_network_examples,
    plot_statistical_heatmap,
    plot_demand_comparison,
    plot_lambda_sweep,
    plot_demand_topology,
)
from src.statistics_utils import wilcoxon_pairwise

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"


def print_demand_stats(df, n: int = 50) -> None:
    sub = df[df["n"] == n]
    print(f"\n--- Pairwise Wilcoxon tests (n={n}, FDR corrected) ---")
    for metric in ["steiner_ratio", "algebraic_connectivity",
                   "robustness", "betweenness_gini"]:
        if metric not in sub.columns:
            continue
        groups = {
            d: sub[sub["demand"] == d][metric].dropna().values
            for d in ["Fixed", "Uniform", "HubSpoke"]
        }
        results = wilcoxon_pairwise(groups)
        print(f"  {metric}:")
        for (a, b), r in results.items():
            sig = ("***" if r["p_fdr"] < 0.001 else "**" if r["p_fdr"] < 0.01
                   else "*" if r["p_fdr"] < 0.05 else "ns")
            print(f"    {a:10s} vs {b:10s}  p_FDR={r['p_fdr']:.4e} {sig}"
                  f"  d={r['cohens_d']:+.3f}")


def print_lambda_summary(df) -> None:
    print("\n--- Lambda sweep summary (n=50) ---")
    metrics = ["steiner_ratio", "algebraic_connectivity",
               "robustness", "cyclomatic_number"]
    metrics = [m for m in metrics if m in df.columns]
    print(f"  {'lambda':>7}" +
          "".join(f"  {m[:10]:>12}" for m in metrics))
    for lam in sorted(df["lambda"].unique()):
        sub = df[df["lambda"] == lam]
        vals = "".join(f"  {sub[m].mean():>12.4f}" for m in metrics)
        print(f"  {lam:>7.2f}{vals}")


def main() -> None:
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  Physarum Network Analysis: Demand-Weighted Extension")
    print(f"  Outputs -> {OUTPUT_DIR}/")
    print("=" * 65)

    print("\n[1/3] Baseline comparison (MST / k-NN / RGG / Physarum-Fixed) ...")
    main_df      = run_main_comparison(n_values=[20, 50, 100], num_trials=30)
    main_summary = save_results(main_df, "main_comparison", OUTPUT_DIR)
    plot_main_comparison(main_summary, OUTPUT_DIR)

    for metric, label in [
        ("algebraic_connectivity", "Algebraic Connectivity λ₂"),
        ("cost",                   "Total Edge Cost"),
        ("robustness",             "Node-Removal Robustness"),
    ]:
        groups = {
            m: main_df[main_df["method"] == m][metric].dropna().values
            for m in ["MST", "kNN", "RGG", "Physarum"]
        }
        results = wilcoxon_pairwise(groups)
        plot_statistical_heatmap(results, label, OUTPUT_DIR)

    print("\n[2/3] Demand comparison (Fixed / Uniform / HubSpoke) ...")
    dem_df = run_demand_comparison(n_values=[20, 50, 100], num_trials=30)
    save_results(dem_df, "demand_comparison", OUTPUT_DIR)
    plot_demand_comparison(dem_df, OUTPUT_DIR)

    print_demand_stats(dem_df, n=50)

    for metric, label in [
        ("algebraic_connectivity", "Algebraic Connectivity λ₂"),
        ("robustness",             "Node-Removal Robustness"),
        ("steiner_ratio",          "Steiner Ratio"),
        ("betweenness_gini",       "Betweenness Gini"),
    ]:
        groups = {
            d: dem_df[dem_df["demand"] == d][metric].dropna().values
            for d in ["Fixed", "Uniform", "HubSpoke"]
        }
        results = wilcoxon_pairwise(groups)
        plot_statistical_heatmap(results, f"Demand {label}", OUTPUT_DIR)

    print("\n[3/3] Lambda sweep (n=50) ...")
    lam_df = run_lambda_sweep(
        n=50, num_trials=30,
        lambda_values=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    )
    save_results(lam_df, "lambda_sweep", OUTPUT_DIR)
    plot_lambda_sweep(lam_df, OUTPUT_DIR)
    print_lambda_summary(lam_df)

    print("\nGenerating network visualisations ...")
    n_vis, seed_vis = 50, 0
    pts = generate_points(n_vis, seed=seed_vis)
    mst = build_mst(pts)
    mst_cost = compute_cost(mst)
    hub = _hub_node(pts)

    graphs_baseline = {
        "MST":      mst,
        "kNN":      build_knn_graph(pts, k=3),
        "RGG":      build_rgg(pts, r=0.3),
        "Physarum": build_physarum_graph(
            pts, num_iters=500, mu=1.0, seed=seed_vis,
        ),
    }
    metric_rows = {m: compute_all_metrics(G, mst_cost=mst_cost)
                   for m, G in graphs_baseline.items()}
    plot_network_examples(graphs_baseline, pts, n=n_vis,
                          output_dir=OUTPUT_DIR, metric_rows=metric_rows)

    graphs_demand = {
        "Fixed":    build_physarum_demand_weighted(
            pts, num_iters=800, mu=1.0, seed=seed_vis,
            demand="fixed", batch_size=50,
        ),
        "Uniform":  build_physarum_demand_weighted(
            pts, num_iters=800, mu=1.0, seed=seed_vis,
            demand="uniform", batch_size=50,
        ),
        "HubSpoke": build_physarum_demand_weighted(
            pts, num_iters=800, mu=1.0, seed=seed_vis,
            demand="hubspoke", batch_size=50,
        ),
    }
    plot_demand_topology(graphs_demand, pts, n_vis, hub, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"All done in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"\nKey outputs in {OUTPUT_DIR}/:")
    print("  main_comparison_raw.csv / _summary.csv")
    print("  demand_comparison_raw.csv / _summary.csv")
    print("  lambda_sweep_raw.csv / _summary.csv")
    print("  demand_comparison_metrics.png  <- main demand comparison")
    print("  lambda_sweep.png               <- 4-panel λ sweep + Pareto")
    print("  demand_topology_n50.png        <- 1x3 topology panel")
    print("  network_comparison_n50.png     <- baseline 4-method topology")
    print("  pvalue_heatmap_*.png           <- FDR-corrected test matrices")


if __name__ == "__main__":
    main()
