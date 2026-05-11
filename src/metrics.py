from typing import Optional

import numpy as np
import networkx as nx


def compute_cost(G: nx.Graph) -> float:
    return float(sum(d["weight"] for _, _, d in G.edges(data=True)))


def compute_efficiency(G: nx.Graph) -> float:
    if not nx.is_connected(G):
        return float("inf")
    return float(nx.average_shortest_path_length(G, weight="weight"))


def compute_robustness(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0
    scores = []
    for v in G.nodes():
        H = G.copy()
        H.remove_node(v)
        if H.number_of_nodes() == 0:
            scores.append(1.0)
            continue
        largest = max(len(c) for c in nx.connected_components(H))
        scores.append(largest / (n - 1))
    return float(np.mean(scores))


def compute_algebraic_connectivity(G: nx.Graph) -> float:
    if not nx.is_connected(G):
        return 0.0
    try:
        return float(nx.algebraic_connectivity(G, weight=None, normalized=False))
    except Exception:
        return 0.0


def compute_network_entropy(G: nx.Graph) -> float:
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    if len(weights) == 0:
        return 0.0
    p = weights / weights.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))


def compute_cyclomatic_number(G: nx.Graph) -> int:
    return (
        G.number_of_edges()
        - G.number_of_nodes()
        + nx.number_connected_components(G)
    )


def compute_betweenness_gini(G: nx.Graph) -> float:
    bc = np.array(
        list(nx.betweenness_centrality(G, weight="weight").values()),
        dtype=float,
    )
    bc = np.sort(bc)
    n = len(bc)
    if n == 0 or bc.sum() == 0.0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.dot(idx, bc)) / (n * bc.sum()) - (n + 1.0) / n)


def compute_steiner_ratio(G: nx.Graph, mst_cost: float) -> float:
    if mst_cost <= 0.0:
        return float("inf")
    return compute_cost(G) / mst_cost


def compute_all_metrics(
    G: nx.Graph,
    mst_cost: Optional[float] = None,
) -> dict:
    return {
        "cost":                   compute_cost(G),
        "efficiency":             compute_efficiency(G),
        "robustness":             compute_robustness(G),
        "algebraic_connectivity": compute_algebraic_connectivity(G),
        "network_entropy":        compute_network_entropy(G),
        "cyclomatic_number":      compute_cyclomatic_number(G),
        "betweenness_gini":       compute_betweenness_gini(G),
        "num_edges":              G.number_of_edges(),
        "steiner_ratio": (
            compute_steiner_ratio(G, mst_cost) if mst_cost is not None else float("nan")
        ),
    }
