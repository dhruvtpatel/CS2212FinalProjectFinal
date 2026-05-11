from itertools import combinations

import numpy as np
import networkx as nx


def generate_points(
    n: int,
    seed: int,
    distribution: str = "uniform",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.uniform(0.0, 1.0, size=(n, 2))
    elif distribution == "clustered":
        k_centers = 3
        centers = rng.uniform(0.15, 0.85, size=(k_centers, 2))
        labels = rng.integers(0, k_centers, size=n)
        pts = np.array([
            centers[lbl] + rng.normal(0, 0.08, size=2) for lbl in labels
        ])
        return np.clip(pts, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")


def build_complete_graph(points: np.ndarray) -> nx.Graph:
    n = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in combinations(range(n), 2):
        dist = float(np.linalg.norm(points[i] - points[j]))
        G.add_edge(i, j, weight=dist)
    return G


def reconnect_graph(G: nx.Graph, points: np.ndarray) -> nx.Graph:
    G = G.copy()
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        best_u, best_v, best_dist = None, None, float("inf")
        for ci in range(len(components)):
            for cj in range(ci + 1, len(components)):
                for u in components[ci]:
                    for v in components[cj]:
                        d = float(np.linalg.norm(points[u] - points[v]))
                        if d < best_dist:
                            best_dist, best_u, best_v = d, u, v
        G.add_edge(best_u, best_v, weight=best_dist)
    return G


def build_mst(points: np.ndarray) -> nx.Graph:
    return nx.minimum_spanning_tree(build_complete_graph(points), weight="weight")


def build_knn_graph(points: np.ndarray, k: int = 3) -> nx.Graph:
    n = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        dists = sorted(
            (float(np.linalg.norm(points[i] - points[j])), j)
            for j in range(n) if j != i
        )
        for dist, j in dists[:k]:
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=dist)
    if not nx.is_connected(G):
        G = reconnect_graph(G, points)
    return G


def build_rgg(points: np.ndarray, r: float = 0.3) -> nx.Graph:
    n = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in combinations(range(n), 2):
        dist = float(np.linalg.norm(points[i] - points[j]))
        if dist <= r:
            G.add_edge(i, j, weight=dist)
    if not nx.is_connected(G):
        G = reconnect_graph(G, points)
    return G
