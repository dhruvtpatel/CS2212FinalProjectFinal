import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import networkx as nx
import pandas as pd

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     ":",
})

METHOD_ORDER = ["MST", "kNN", "RGG", "Physarum"]

METHOD_COLORS = {
    "MST":      "#2166ac",
    "kNN":      "#d6604d",
    "RGG":      "#1a9850",
    "Physarum": "#762a83",
}

METHOD_LABELS = {
    "MST":      "MST",
    "kNN":      "k-NN (k=3)",
    "RGG":      "RGG (r=0.3)",
    "Physarum": "Physarum (μ=1)",
}

MU_CMAP = "plasma"


def _mu_norm(mu_values: list) -> mcolors.Normalize:
    return mcolors.Normalize(vmin=min(mu_values), vmax=max(mu_values))


def _save(fig, path: str) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


_MAIN_METRICS = [
    ("cost_mean",                   "cost_std",
     "Total Edge Cost",              "cost_vs_n.png"),
    ("efficiency_mean",             "efficiency_std",
     "Avg. Shortest Path Length",   "efficiency_vs_n.png"),
    ("robustness_mean",             "robustness_std",
     "Node-Removal Robustness",      "robustness_vs_n.png"),
    ("algebraic_connectivity_mean", "algebraic_connectivity_std",
     "Algebraic Connectivity λ₂",   "lambda2_vs_n.png"),
    ("network_entropy_mean",        "network_entropy_std",
     "Network Entropy H(w)",         "entropy_vs_n.png"),
    ("cyclomatic_number_mean",      "cyclomatic_number_std",
     "Cyclomatic Number γ",          "gamma_vs_n.png"),
]


def plot_main_comparison(
    summary_df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for mean_col, std_col, ylabel, fname in _MAIN_METRICS:
        if mean_col not in summary_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in METHOD_ORDER:
            sub = summary_df[summary_df["method"] == method].sort_values("n")
            ax.errorbar(
                sub["n"], sub[mean_col], yerr=sub[std_col],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker="o", linewidth=2, capsize=5,
            )
        ax.set_xlabel("Number of nodes n")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        _save(fig, os.path.join(output_dir, fname))

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHOD_ORDER:
        sub = summary_df[summary_df["method"] == method].sort_values("n")
        ax.scatter(
            sub["cost_mean"], sub["robustness_mean"],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method], s=80, zorder=5,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"n={int(row['n'])}",
                (row["cost_mean"], row["robustness_mean"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7.5, color=METHOD_COLORS[method],
            )
    ax.set_xlabel("Cost (total edge length)")
    ax.set_ylabel("Node-removal robustness")
    ax.set_title("Cost–Robustness Trade-off")
    ax.legend()
    _save(fig, os.path.join(output_dir, "cost_vs_robustness.png"))


def plot_multicommodity(
    raw_df: pd.DataFrame,
    powerlaw_result: Optional[dict] = None,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    phy_df   = raw_df[raw_df["method"].str.startswith("Physarum-k")].copy()
    union_df = raw_df[raw_df["method"].str.startswith("Union-k")].copy()
    mst_df   = raw_df[raw_df["method"] == "MST"]
    knn_df   = raw_df[raw_df["method"] == "kNN"]

    k_vals = sorted(phy_df["k_commodities"].unique())

    def _grp_stats(df, col):
        grp = df.groupby("k_commodities")[col].agg(["mean", "std", "count"])
        se  = grp["std"] / np.sqrt(grp["count"])
        return grp["mean"].reindex(k_vals), se.reindex(k_vals)

    mst_sr   = float(mst_df["steiner_ratio"].mean()) if "steiner_ratio" in mst_df.columns else 1.0
    knn_sr   = float(knn_df["steiner_ratio"].mean())
    mst_lam  = float(mst_df["algebraic_connectivity"].mean())
    knn_lam  = float(knn_df["algebraic_connectivity"].mean())
    mst_cost = float(mst_df["cost"].mean())
    knn_cost = float(knn_df["cost"].mean())

    sr_j,  sr_j_se   = _grp_stats(phy_df,   "steiner_ratio")
    sr_u,  sr_u_se   = _grp_stats(union_df, "steiner_ratio")
    lam_j, lam_j_se  = _grp_stats(phy_df,   "algebraic_connectivity")
    lam_u, lam_u_se  = _grp_stats(union_df, "algebraic_connectivity")
    c_j,   c_j_se    = _grp_stats(phy_df,   "cost")
    c_u,   c_u_se    = _grp_stats(union_df, "cost")

    lpc_j   = lam_j / c_j
    lpc_u   = lam_u / c_u
    knn_lpc = knn_lam / knn_cost
    mst_lpc = mst_lam / mst_cost if mst_cost > 0 else 0.0

    phy_col   = "#762a83"
    union_col = "#1b7837"
    mst_col   = METHOD_COLORS["MST"]
    knn_col   = METHOD_COLORS["kNN"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(
        "Multi-Commodity Physarum vs. Union of Independent Networks  (n=50, 30 trials)",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0]
    ax.errorbar(k_vals, sr_j.values, yerr=sr_j_se.values,
                color=phy_col,   marker="o", linewidth=2, capsize=5,
                label="Physarum joint")
    ax.errorbar(k_vals, sr_u.values, yerr=sr_u_se.values,
                color=union_col, marker="s", linewidth=2, capsize=5,
                linestyle="--", label="Union (k independent)")
    ax.axhline(mst_sr, color=mst_col, linestyle=":",  linewidth=1.5, label="MST")
    ax.axhline(knn_sr, color=knn_col, linestyle="-.", linewidth=1.5, label="k-NN")
    if powerlaw_result:
        a    = np.exp(powerlaw_result["log_a"])
        alph = powerlaw_result["alpha"]
        xs   = np.linspace(min(k_vals), max(k_vals), 100)
        ax.plot(xs, a * xs**alph, "k--", linewidth=1, alpha=0.5,
                label=f"fit: k^{alph:.2f}")
    ax.set_xlabel("Number of commodities k")
    ax.set_ylabel("Steiner ratio  (cost / MST cost)")
    ax.set_title("Cost Premium vs. k")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.errorbar(k_vals, lam_j.values, yerr=lam_j_se.values,
                color=phy_col,   marker="o", linewidth=2, capsize=5,
                label="Physarum joint")
    ax.errorbar(k_vals, lam_u.values, yerr=lam_u_se.values,
                color=union_col, marker="s", linewidth=2, capsize=5,
                linestyle="--", label="Union (k independent)")
    ax.axhline(mst_lam, color=mst_col, linestyle=":",  linewidth=1.5, label="MST")
    ax.axhline(knn_lam, color=knn_col, linestyle="-.", linewidth=1.5, label="k-NN")
    ax.set_xlabel("Number of commodities k")
    ax.set_ylabel("Algebraic connectivity λ₂")
    ax.set_title("Structural Robustness vs. k")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    ax.plot(k_vals, lpc_j.values, "o-",  color=phy_col,   linewidth=2,
            label="Physarum joint")
    ax.plot(k_vals, lpc_u.values, "s--", color=union_col, linewidth=2,
            label="Union (k independent)")
    ax.axhline(mst_lpc, color=mst_col, linestyle=":",  linewidth=1.5, label="MST")
    ax.axhline(knn_lpc, color=knn_col, linestyle="-.", linewidth=1.5, label="k-NN")
    ax.set_xlabel("Number of commodities k")
    ax.set_ylabel("λ₂ / cost")
    ax.set_title("Robustness per Unit Cost vs. k")
    ax.legend(fontsize=7.5)

    ax = axes[3]
    sc_j = ax.scatter(c_j.values, lam_j.values,
                      c=k_vals, cmap="plasma", s=100, zorder=5,
                      edgecolors=phy_col, linewidths=1.5, label="Joint")
    ax.scatter(c_u.values, lam_u.values,
               c=k_vals, cmap="plasma", s=80, zorder=5,
               marker="s", edgecolors=union_col, linewidths=1.5, label="Union")
    for km, cj, lj, cu, lu in zip(k_vals, c_j.values, lam_j.values,
                                   c_u.values, lam_u.values):
        ax.annotate(f"k={km}", (cj, lj), xytext=(4, 3),
                    textcoords="offset points", fontsize=7.5)
    ax.scatter([mst_cost], [mst_lam], marker="*", s=180, color=mst_col,
               zorder=6, label="MST")
    ax.scatter([knn_cost], [knn_lam], marker="^", s=140, color=knn_col,
               zorder=6, label="k-NN")
    plt.colorbar(sc_j, ax=ax, label="k")
    ax.set_xlabel("Mean total edge cost")
    ax.set_ylabel("Mean algebraic connectivity λ₂")
    ax.set_title("Cost–Robustness Pareto\n(circles=joint, squares=union)")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "multicommodity_sweep.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(k_vals, sr_j.values, yerr=1.96 * sr_j_se.values,
                color=phy_col,   marker="o", linewidth=2.5, capsize=6, zorder=5,
                label="Physarum (joint adaptation)")
    ax.errorbar(k_vals, sr_u.values, yerr=1.96 * sr_u_se.values,
                color=union_col, marker="s", linewidth=2.5, capsize=6, zorder=5,
                linestyle="--", label="Union (k independent runs)")
    ax.axhline(mst_sr, color=mst_col, linestyle=":",  linewidth=1.8,
               label=f"MST ({mst_sr:.2f})")
    ax.axhline(knn_sr, color=knn_col, linestyle="-.", linewidth=1.8,
               label=f"k-NN ({knn_sr:.2f})")
    if powerlaw_result:
        a    = np.exp(powerlaw_result["log_a"])
        alph = powerlaw_result["alpha"]
        alo  = powerlaw_result["alpha_ci_low"]
        ahi  = powerlaw_result["alpha_ci_high"]
        xs   = np.linspace(min(k_vals), max(k_vals), 200)
        ax.plot(xs, a * xs**alph, "k--", linewidth=1.2,
                label=fr"Power-law fit  $k^{{{alph:.2f}}}$  [{alo:.2f}, {ahi:.2f}]")
    ax.set_xlabel("Number of simultaneous commodities k")
    ax.set_ylabel("Steiner ratio  cost / cost(MST)")
    ax.set_title("Physarum's Cost Premium Grows with Routing Demand\n"
                 "(error bars: 95% CI)")
    ax.legend(fontsize=9)
    _save(fig, os.path.join(output_dir, "multicommodity_steiner_ratio.png"))


_MU_METRICS = [
    ("cost",                   "Total Edge Cost",             False),
    ("algebraic_connectivity", "Algebraic Connectivity λ₂",  True),
    ("network_entropy",        "Network Entropy H(w)",         True),
    ("cyclomatic_number",      "Cyclomatic Number γ",          True),
    ("efficiency",             "Avg. Shortest Path",          False),
    ("robustness",             "Node-Removal Robustness",      True),
    ("betweenness_gini",       "Betweenness Gini",            False),
    ("steiner_ratio",          "Steiner Ratio (Cost/MST)",    False),
]


def plot_mu_sweep(
    raw_df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mu_values = sorted(raw_df["mu"].unique().tolist())
    cmap = plt.get_cmap(MU_CMAP)
    norm = _mu_norm(mu_values)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Physarum Network Properties vs. Reinforcement Exponent μ  (n=50, 30 trials)",
        fontsize=12, fontweight="bold",
    )
    for ax, (metric, label, higher_better) in zip(axes.flat, _MU_METRICS):
        if metric not in raw_df.columns:
            ax.set_visible(False)
            continue
        grp = (
            raw_df.groupby("mu")[metric]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        se = grp["std"] / np.sqrt(grp["count"])
        ax.plot(grp["mu"], grp["mean"], "o-", color="#762a83", linewidth=2, markersize=5)
        ax.fill_between(
            grp["mu"],
            grp["mean"] - 1.96 * se,
            grp["mean"] + 1.96 * se,
            alpha=0.2, color="#762a83",
        )
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlabel("μ")
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=9)
        direction = "↑ better" if higher_better else "↓ better"
        ax.text(0.97, 0.97, direction, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "mu_sweep_metrics.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    grp2 = raw_df.groupby("mu")[["cost", "algebraic_connectivity"]].mean().reset_index()
    sc = ax.scatter(
        grp2["cost"], grp2["algebraic_connectivity"],
        c=grp2["mu"], cmap=MU_CMAP, norm=norm,
        s=110, zorder=5, edgecolors="k", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="μ")
    for _, row in grp2.iterrows():
        ax.annotate(
            f"μ={row['mu']:.2g}",
            (row["cost"], row["algebraic_connectivity"]),
            xytext=(5, 4), textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("Mean Total Edge Cost")
    ax.set_ylabel("Mean Algebraic Connectivity λ₂")
    ax.set_title("Cost–Robustness Phase Diagram\n(each point = one μ value)")
    _save(fig, os.path.join(output_dir, "mu_phase_diagram.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    grp3 = grp2.copy()
    grp3["lambda2_per_cost"] = grp3["algebraic_connectivity"] / grp3["cost"]
    ax.plot(grp3["mu"], grp3["lambda2_per_cost"], "o-", color="#762a83",
            linewidth=2, markersize=6)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.8, label="μ=1 (biological)")
    ax.set_xlabel("μ")
    ax.set_ylabel("λ₂ / cost  (robustness per unit cost)")
    ax.set_title("Cost-Normalised Algebraic Connectivity vs. μ")
    ax.legend()
    _save(fig, os.path.join(output_dir, "mu_lambda2_per_cost.png"))


def plot_convergence(
    df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mu_values = sorted(df["mu"].unique().tolist())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(mu_values)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Convergence of Physarum Dynamics by μ  (n=50)", fontsize=11)

    for mu, col in zip(mu_values, colors):
        sub = (
            df[df["mu"] == mu]
            .groupby("iteration")
            .agg(
                ne_mean=("n_surviving_edges",    "mean"),
                ne_std= ("n_surviving_edges",    "std"),
                h_mean= ("conductivity_entropy",  "mean"),
                h_std=  ("conductivity_entropy",  "std"),
            )
            .reset_index()
        )
        lbl = f"μ={mu:.2g}"
        ax1.plot(sub["iteration"], sub["ne_mean"], "o-", color=col, label=lbl, linewidth=2, ms=4)
        ax1.fill_between(
            sub["iteration"],
            sub["ne_mean"] - sub["ne_std"],
            sub["ne_mean"] + sub["ne_std"],
            alpha=0.15, color=col,
        )
        ax2.plot(sub["iteration"], sub["h_mean"], "o-", color=col, label=lbl, linewidth=2, ms=4)
        ax2.fill_between(
            sub["iteration"],
            sub["h_mean"] - sub["h_std"],
            sub["h_mean"] + sub["h_std"],
            alpha=0.15, color=col,
        )

    for ax, ylabel, title in [
        (ax1, "Number of surviving edges",    "Edge Count Convergence"),
        (ax2, "Conductivity entropy H(D)",    "Conductivity Entropy Convergence"),
    ]:
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "convergence_study.png"))


def _draw_network(
    ax,
    G: nx.Graph,
    points: np.ndarray,
    title: str,
    color: str,
    annotation: Optional[str] = None,
) -> None:
    pos = {i: (points[i, 0], points[i, 1]) for i in range(len(points))}
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    widths = (
        [0.6 + 2.8 * (1.0 - w / max(weights)) for w in weights]
        if weights else []
    )
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=color, alpha=0.75)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=40, node_color="k")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_aspect("equal")
    ax.axis("off")
    if annotation:
        ax.text(0.5, -0.04, annotation, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, style="italic", color="#444")


def plot_network_examples(
    graphs: Dict[str, nx.Graph],
    points: np.ndarray,
    n: int,
    output_dir: str = "output",
    metric_rows: Optional[Dict[str, dict]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(f"Network Structure Comparison  (n={n})",
                 fontsize=12, fontweight="bold")

    for ax, method in zip(axes.flat, METHOD_ORDER):
        G = graphs[method]
        ann = None
        if metric_rows and method in metric_rows:
            m = metric_rows[method]
            ann = (
                f"|E|={G.number_of_edges()}  γ={G.number_of_edges()-n+1}"
                f"  cost={m.get('cost', 0):.2f}"
                f"  λ₂={m.get('algebraic_connectivity', 0):.3f}"
            )
        _draw_network(ax, G, points, METHOD_LABELS[method], METHOD_COLORS[method], ann)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"network_comparison_n{n}.png"))


def plot_mu_topology_series(
    points: np.ndarray,
    mu_graphs: Dict[float, nx.Graph],
    n: int,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mu_values = sorted(mu_graphs.keys())
    cmap = plt.get_cmap(MU_CMAP)
    norm = _mu_norm(mu_values)

    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle(f"Physarum Network Topology vs. μ  (n={n})",
                 fontsize=12, fontweight="bold")

    for ax, mu in zip(axes.flat, mu_values):
        G = mu_graphs[mu]
        color = cmap(norm(mu))
        gamma = G.number_of_edges() - n + nx.number_connected_components(G)
        ann = f"|E|={G.number_of_edges()}  γ={gamma}"
        _draw_network(ax, G, points, f"μ={mu:.2g}", color, ann)

    for ax in axes.flat[len(mu_values):]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "physarum_mu_topology.png"))


def plot_statistical_heatmap(
    stat_results: dict,
    metric_name: str,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    labels = sorted({k for pair in stat_results for k in pair})
    n_labels = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}

    pmat = np.ones((n_labels, n_labels))
    dmat = np.zeros((n_labels, n_labels))

    for (a, b), res in stat_results.items():
        i, j = idx[a], idx[b]
        pmat[i, j] = pmat[j, i] = res.get("p_fdr", 1.0)
        dmat[i, j] = dmat[j, i] = abs(res.get("cohens_d", 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mat, title, fmt in [
        (axes[0], np.log10(pmat + 1e-300), f"log₁₀(p_FDR)  —  {metric_name}", ".2f"),
        (axes[1], dmat,                     f"|Cohen's d|  —  {metric_name}",  ".2f"),
    ]:
        im = ax.imshow(mat, cmap="RdYlGn_r" if "log" in title else "Blues",
                       aspect="auto")
        ax.set_xticks(range(n_labels))
        ax.set_yticks(range(n_labels))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)
        for i in range(n_labels):
            for j in range(n_labels):
                if i != j:
                    ax.text(j, i, f"{mat[i,j]:{fmt}}", ha="center", va="center",
                            fontsize=8)

    fig.tight_layout()
    safe = metric_name.replace(" ", "_").replace("λ", "lambda").replace("₂", "2")
    _save(fig, os.path.join(output_dir, f"pvalue_heatmap_{safe}.png"))


DEMAND_COLORS = {
    "Fixed":    "#2166ac",
    "Uniform":  "#d6604d",
    "HubSpoke": "#1a9850",
}
DEMAND_LABELS = {
    "Fixed":    "Fixed pair (Tero)",
    "Uniform":  "Uniform demand (this work)",
    "HubSpoke": "Hub-spoke demand",
}
DEMAND_ORDER = ["Fixed", "Uniform", "HubSpoke"]

_DEMAND_METRICS = [
    ("steiner_ratio",           "Steiner ratio ρ"),
    ("algebraic_connectivity",  "Algebraic connectivity λ₂"),
    ("cyclomatic_number",       "Cyclomatic number γ"),
    ("robustness",              "Node-removal robustness"),
    ("betweenness_gini",        "Betweenness Gini"),
    ("network_entropy",         "Network entropy H"),
]


def plot_demand_comparison(
    df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    available = [(m, lbl) for m, lbl in _DEMAND_METRICS if m in df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    n_vals = sorted(df["n"].unique())

    for ax, (metric, label) in zip(axes, available):
        for demand in DEMAND_ORDER:
            sub = df[df["demand"] == demand]
            means, cis = [], []
            for n in n_vals:
                vals = sub[sub["n"] == n][metric].dropna().values
                if len(vals) == 0:
                    means.append(np.nan); cis.append(np.nan); continue
                m_  = float(np.mean(vals))
                se  = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
                means.append(m_); cis.append(se)
            means = np.array(means, dtype=float)
            cis   = np.array(cis,   dtype=float)
            color = DEMAND_COLORS[demand]
            ax.plot(n_vals, means, "o-", color=color,
                    label=DEMAND_LABELS[demand], lw=1.8)
            ax.fill_between(n_vals, means - cis, means + cis,
                            alpha=0.15, color=color)
        ax.set_xlabel("n (nodes)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7.5)

    for ax in axes[len(available):]:
        ax.set_visible(False)

    fig.suptitle(
        "Demand-Weighted Physarum: Fixed vs Uniform vs Hub-Spoke Demand",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "demand_comparison_metrics.png"))


def plot_lambda_sweep(
    df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    lam_vals = sorted(df["lambda"].unique())
    col_uni  = DEMAND_COLORS["Uniform"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(
        "Demand Interpolation W=(1−λ)·W_fixed + λ·W_uniform  (n=50, 30 trials)",
        fontsize=11, fontweight="bold",
    )

    panel_metrics = [
        ("steiner_ratio",          "Steiner ratio ρ"),
        ("algebraic_connectivity", "Algebraic connectivity λ₂"),
        ("robustness",             "Node-removal robustness"),
    ]

    for ax, (metric, label) in zip(axes[:3], panel_metrics):
        means, cis = [], []
        for lam in lam_vals:
            vals = df[df["lambda"] == lam][metric].dropna().values
            if len(vals) == 0:
                means.append(np.nan); cis.append(np.nan); continue
            m_  = float(np.mean(vals))
            se  = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
            means.append(m_); cis.append(se)
        means = np.array(means, dtype=float)
        cis   = np.array(cis,   dtype=float)
        ax.plot(lam_vals, means, "o-", color=col_uni, linewidth=2, markersize=6)
        ax.fill_between(lam_vals, means - cis, means + cis, alpha=0.2, color=col_uni)
        ax.axvline(0.0, color=DEMAND_COLORS["Fixed"],  linestyle="--",
                   linewidth=1.0, alpha=0.8, label="λ=0 (Tero fixed)")
        ax.axvline(1.0, color=DEMAND_COLORS["Uniform"], linestyle="--",
                   linewidth=1.0, alpha=0.8, label="λ=1 (uniform)")
        ax.set_xlabel("λ  (demand interpolation weight)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)

    ax4 = axes[3]
    cost_m = [df[df["lambda"] == lam]["cost"].mean() for lam in lam_vals]
    lam2_m = [df[df["lambda"] == lam]["algebraic_connectivity"].mean()
              for lam in lam_vals]
    sc = ax4.scatter(
        cost_m, lam2_m,
        c=lam_vals, cmap="plasma", s=110,
        edgecolors="k", linewidths=0.6, zorder=5,
    )
    plt.colorbar(sc, ax=ax4, label="λ")
    for lam, cx, lx in zip(lam_vals, cost_m, lam2_m):
        ax4.annotate(f"λ={lam:.2g}", (cx, lx), xytext=(4, 3),
                     textcoords="offset points", fontsize=7.5)
    ax4.set_xlabel("Mean total edge cost")
    ax4.set_ylabel("Mean algebraic connectivity λ₂")
    ax4.set_title("Cost–Robustness Pareto\n(one point per λ)")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "lambda_sweep.png"))


def plot_demand_topology(
    graphs: Dict[str, nx.Graph],
    pts: np.ndarray,
    n: int,
    hub: int,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(f"Network Topology by Demand Distribution  (n={n})",
                 fontsize=12, fontweight="bold")

    for ax, demand in zip(axes, DEMAND_ORDER):
        G     = graphs[demand]
        color = DEMAND_COLORS[demand]
        pos   = {i: (float(pts[i, 0]), float(pts[i, 1])) for i in range(n)}

        weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        if weights:
            w_max  = max(weights)
            widths = [0.5 + 2.5 * (1.0 - w / w_max) for w in weights]
        else:
            widths = []

        nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                               edge_color=color, alpha=0.65)

        node_colors = []
        node_sizes  = []
        for v in G.nodes():
            if demand == "HubSpoke" and v == hub:
                node_colors.append("#e31a1c")
                node_sizes.append(160)
            else:
                node_colors.append("k")
                node_sizes.append(30)
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors, node_size=node_sizes)

        lam2  = nx.algebraic_connectivity(G) if nx.is_connected(G) else 0.0
        gamma = G.number_of_edges() - G.number_of_nodes() + 1
        cost  = sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
        gini  = float(np.array(
            list(nx.betweenness_centrality(G, weight="weight").values())
        ).std())
        ax.set_title(
            f"{DEMAND_LABELS[demand]}\n"
            f"|E|={G.number_of_edges()}  γ={gamma}  "
            f"λ₂={lam2:.3f}  cost={cost:.2f}",
            fontsize=9, color=color, fontweight="bold",
        )
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"demand_topology_n{n}.png"))
