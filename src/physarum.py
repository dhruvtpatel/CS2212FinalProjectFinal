from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx
from scipy.linalg import solve, lu_factor, lu_solve, LinAlgError

from .graph_generation import reconnect_graph


def _farthest_pair(points: np.ndarray) -> Tuple[int, int]:
    n = len(points)
    best_d, src, snk = -1.0, 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            if d > best_d:
                best_d, src, snk = d, i, j
    return src, snk


def _assemble_and_solve(
    C_arr: np.ndarray,
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    ref_node: int,
    free_nodes: List[int],
    free_idx: dict,
    fi_both: np.ndarray,
    fj_both: np.ndarray,
    fi_only: np.ndarray,
    fj_only: np.ndarray,
    idx_both: np.ndarray,
    idx_only_i: np.ndarray,
    idx_only_j: np.ndarray,
    b_red: np.ndarray,
    n: int,
) -> np.ndarray:
    m = len(free_nodes)
    L_red = np.zeros((m, m))

    C_both = C_arr[idx_both]
    np.add.at(L_red, (fi_both, fi_both),  C_both)
    np.add.at(L_red, (fj_both, fj_both),  C_both)
    np.add.at(L_red, (fi_both, fj_both), -C_both)
    np.add.at(L_red, (fj_both, fi_both), -C_both)

    np.add.at(L_red, (fi_only, fi_only), C_arr[idx_only_i])
    np.add.at(L_red, (fj_only, fj_only), C_arr[idx_only_j])

    L_red += 1e-12 * np.eye(m)

    try:
        p_free = solve(L_red, b_red, assume_a="sym", check_finite=False)
    except LinAlgError:
        p_free, *_ = np.linalg.lstsq(L_red, b_red, rcond=None)

    p = np.zeros(n)
    for k, v in enumerate(free_nodes):
        p[v] = p_free[k]
    return p


def _physarum_core(
    points: np.ndarray,
    num_iters: int,
    dt: float,
    mu: float,
    decay_order: float,
    threshold: float,
    seed: Optional[int],
    checkpoint_iters: Optional[List[int]],
    alpha: float = 0.0,
    mu_min: float = 0.3,
    mu_max: float = 1.8,
):
    n = len(points)
    rng = np.random.default_rng(seed)

    edge_list = list(combinations(range(n), 2))
    num_edges = len(edge_list)
    i_arr = np.array([e[0] for e in edge_list], dtype=np.intp)
    j_arr = np.array([e[1] for e in edge_list], dtype=np.intp)
    L_arr = np.array(
        [float(np.linalg.norm(points[i] - points[j])) for i, j in edge_list]
    )
    L_arr = np.maximum(L_arr, 1e-10)

    if alpha != 0.0:
        L_mean = float(L_arr.mean())
        mu_arr = np.clip(mu * (L_arr / L_mean) ** alpha, mu_min, mu_max)
    else:
        mu_arr = None

    D = rng.uniform(0.5, 1.0, size=num_edges)

    source, sink = _farthest_pair(points)

    b = np.zeros(n)
    b[source] = -1.0
    b[sink]   =  1.0

    ref_node   = sink
    free_nodes = [v for v in range(n) if v != ref_node]
    free_idx   = {v: k for k, v in enumerate(free_nodes)}
    m          = len(free_nodes)

    b_red = np.array([-b[v] for v in free_nodes])

    mask_i_free = i_arr != ref_node
    mask_j_free = j_arr != ref_node
    mask_both   = mask_i_free &  mask_j_free
    mask_only_i = mask_i_free & ~mask_j_free
    mask_only_j = mask_j_free & ~mask_i_free

    fi_both  = np.array([free_idx[i_arr[k]] for k in np.where(mask_both)[0]],   dtype=np.intp)
    fj_both  = np.array([free_idx[j_arr[k]] for k in np.where(mask_both)[0]],   dtype=np.intp)
    fi_only  = np.array([free_idx[i_arr[k]] for k in np.where(mask_only_i)[0]], dtype=np.intp)
    fj_only  = np.array([free_idx[j_arr[k]] for k in np.where(mask_only_j)[0]], dtype=np.intp)
    idx_both   = np.where(mask_both)[0]
    idx_only_i = np.where(mask_only_i)[0]
    idx_only_j = np.where(mask_only_j)[0]

    checkpoint_set = set(checkpoint_iters) if checkpoint_iters else set()
    checkpoints: List[dict] = []

    for iteration in range(1, num_iters + 1):
        C_arr = D / L_arr
        p = _assemble_and_solve(
            C_arr, i_arr, j_arr, ref_node, free_nodes, free_idx,
            fi_both, fj_both, fi_only, fj_only,
            idx_both, idx_only_i, idx_only_j,
            b_red, n,
        )
        Q = C_arr * (p[i_arr] - p[j_arr])
        if mu_arr is not None:
            D = D + dt * (np.abs(Q) ** mu_arr - D ** decay_order)
        else:
            D = D + dt * (np.abs(Q) ** mu - D ** decay_order)
        D = np.maximum(D, 1e-10)

        if iteration in checkpoint_set:
            surviving = D > threshold
            n_edges   = int(np.sum(surviving))
            D_s       = D[surviving]
            D_norm    = D_s / D_s.sum()
            h         = float(-np.sum(D_norm * np.log(D_norm + 1e-15)))
            checkpoints.append({
                "iteration":             iteration,
                "n_surviving_edges":     n_edges,
                "conductivity_entropy":  h,
            })

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for k, (i, j) in enumerate(edge_list):
        if D[k] > threshold:
            G.add_edge(i, j, weight=float(L_arr[k]))

    if not nx.is_connected(G):
        G = reconnect_graph(G, points)

    return G, checkpoints


def build_physarum_graph(
    points: np.ndarray,
    num_iters: int = 500,
    dt: float = 0.1,
    mu: float = 1.0,
    decay_order: float = 1.0,
    threshold: float = 0.05,
    seed: Optional[int] = None,
) -> nx.Graph:
    G, _ = _physarum_core(
        points, num_iters, dt, mu, decay_order, threshold, seed, None,
        alpha=0.0,
    )
    return G


def build_physarum_adaptive_mu(
    points: np.ndarray,
    num_iters: int = 500,
    dt: float = 0.1,
    base_mu: float = 1.0,
    alpha: float = 1.0,
    mu_min: float = 0.3,
    mu_max: float = 1.8,
    threshold: float = 0.05,
    seed: Optional[int] = None,
) -> nx.Graph:
    G, _ = _physarum_core(
        points, num_iters, dt, base_mu, 1.0, threshold, seed, None,
        alpha=alpha, mu_min=mu_min, mu_max=mu_max,
    )
    return G


def select_source_sink_pairs(
    points: np.ndarray,
    k: int,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    n = len(points)
    if k == 1:
        return [_farthest_pair(points)]
    if 2 * k > n:
        raise ValueError(f"Cannot choose {k} non-overlapping pairs from {n} nodes.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [(int(perm[c]), int(perm[k + c])) for c in range(k)]


def build_physarum_union(
    points: np.ndarray,
    source_sink_pairs: List[Tuple[int, int]],
    num_iters: int = 500,
    dt: float = 0.1,
    mu: float = 1.0,
    decay_order: float = 1.0,
    threshold: float = 0.05,
    seed: Optional[int] = None,
) -> nx.Graph:
    n = len(points)
    union_edges: dict = {}
    for c, pair in enumerate(source_sink_pairs):
        c_seed = None if seed is None else seed + c
        G_c = build_physarum_multicommodity(
            points,
            k=1,
            source_sink_pairs=[pair],
            num_iters=num_iters,
            dt=dt,
            mu=mu,
            decay_order=decay_order,
            threshold=threshold,
            seed=c_seed,
        )
        for u, v, data in G_c.edges(data=True):
            key = (min(u, v), max(u, v))
            union_edges[key] = data.get("weight", 1.0)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), w in union_edges.items():
        G.add_edge(i, j, weight=w)
    if not nx.is_connected(G):
        G = reconnect_graph(G, points)
    return G


def build_physarum_multicommodity(
    points: np.ndarray,
    k: int = 1,
    source_sink_pairs: Optional[List[Tuple[int, int]]] = None,
    num_iters: int = 500,
    dt: float = 0.1,
    mu: float = 1.0,
    decay_order: float = 1.0,
    threshold: float = 0.05,
    seed: Optional[int] = None,
) -> nx.Graph:
    n = len(points)
    rng = np.random.default_rng(seed)

    edge_list = list(combinations(range(n), 2))
    E = len(edge_list)
    i_arr = np.array([e[0] for e in edge_list], dtype=np.intp)
    j_arr = np.array([e[1] for e in edge_list], dtype=np.intp)
    L_arr = np.maximum(
        np.array([float(np.linalg.norm(points[a] - points[b])) for a, b in edge_list]),
        1e-10,
    )

    D = rng.uniform(0.5, 1.0, size=E)

    if source_sink_pairs is None:
        if k == 1:
            src, snk = _farthest_pair(points)
            source_sink_pairs = [(src, snk)]
        else:
            if 2 * k > n:
                raise ValueError(
                    f"Cannot choose {k} non-overlapping pairs from {n} nodes."
                )
            perm = rng.permutation(n)
            source_sink_pairs = [
                (int(perm[c]), int(perm[k + c])) for c in range(k)
            ]

    ref_node   = 0
    m          = n - 1
    free_nodes = list(range(1, n))

    mask_both   = (i_arr != ref_node) & (j_arr != ref_node)
    mask_only_j = (i_arr == ref_node)

    fi_both = (i_arr[mask_both] - 1).astype(np.intp)
    fj_both = (j_arr[mask_both] - 1).astype(np.intp)
    fj_only = (j_arr[mask_only_j] - 1).astype(np.intp)

    idx_both   = np.where(mask_both)[0]
    idx_only_j = np.where(mask_only_j)[0]

    b_reds = np.zeros((k, m))
    for c, (src, snk) in enumerate(source_sink_pairs):
        if src != ref_node:
            b_reds[c, src - 1] += 1.0
        if snk != ref_node:
            b_reds[c, snk - 1] -= 1.0

    for _ in range(num_iters):
        C_arr = D / L_arr

        L_red = np.zeros((m, m))
        C_b = C_arr[idx_both]
        np.add.at(L_red, (fi_both, fi_both),  C_b)
        np.add.at(L_red, (fj_both, fj_both),  C_b)
        np.add.at(L_red, (fi_both, fj_both), -C_b)
        np.add.at(L_red, (fj_both, fi_both), -C_b)
        np.add.at(L_red, (fj_only, fj_only),  C_arr[idx_only_j])
        L_red += 1e-12 * np.eye(m)

        lu, piv = lu_factor(L_red, check_finite=False)

        total_abs_Q = np.zeros(E)
        for c in range(k):
            p_free = lu_solve((lu, piv), b_reds[c], check_finite=False)
            p      = np.empty(n)
            p[0]   = 0.0
            p[1:]  = p_free
            Q_c    = C_arr * (p[i_arr] - p[j_arr])
            total_abs_Q += np.abs(Q_c)

        D  = D + dt * (total_abs_Q ** mu - D ** decay_order)
        D  = np.maximum(D, 1e-10)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for idx, (a, b) in enumerate(edge_list):
        if D[idx] > threshold:
            G.add_edge(a, b, weight=float(L_arr[idx]))

    if not nx.is_connected(G):
        G = reconnect_graph(G, points)

    return G


def build_physarum_with_tracking(
    points: np.ndarray,
    num_iters: int = 500,
    dt: float = 0.1,
    mu: float = 1.0,
    decay_order: float = 1.0,
    threshold: float = 0.05,
    seed: Optional[int] = None,
    checkpoint_iters: Optional[List[int]] = None,
    alpha: float = 0.0,
    mu_min: float = 0.3,
    mu_max: float = 1.8,
) -> Tuple[nx.Graph, List[dict]]:
    if checkpoint_iters is None:
        checkpoint_iters = [10, 25, 50, 100, 200, 350, 500]
    return _physarum_core(
        points, num_iters, dt, mu, decay_order, threshold, seed,
        checkpoint_iters, alpha=alpha, mu_min=mu_min, mu_max=mu_max,
    )


def _hub_node(points: np.ndarray) -> int:
    dist_sums = np.array([
        float(np.sum(np.linalg.norm(points - points[i], axis=1)))
        for i in range(len(points))
    ])
    return int(np.argmin(dist_sums))


def build_physarum_demand_weighted(
    points: np.ndarray,
    num_iters: int = 800,
    dt: float = 0.1,
    mu: float = 1.0,
    threshold: float = 0.05,
    seed: Optional[int] = None,
    demand: str = "uniform",
    batch_size: int = 50,
    lambda_mix: float = 1.0,
) -> nx.Graph:
    n = len(points)
    rng = np.random.default_rng(seed)

    edge_list = list(combinations(range(n), 2))
    E = len(edge_list)
    i_arr = np.array([e[0] for e in edge_list], dtype=np.intp)
    j_arr = np.array([e[1] for e in edge_list], dtype=np.intp)
    L_arr = np.maximum(
        np.array([float(np.linalg.norm(points[a] - points[b]))
                  for a, b in edge_list]),
        1e-10,
    )

    D = rng.uniform(0.5, 1.0, size=E)

    source, sink = _farthest_pair(points)

    hub = _hub_node(points) if demand in ("hubspoke",) else 0

    all_pairs = np.array(edge_list, dtype=np.intp)
    n_pairs   = len(all_pairs)

    if demand == "hubspoke":
        hub_pairs = np.array(
            [(min(hub, j), max(hub, j)) for j in range(n) if j != hub],
            dtype=np.intp,
        )
        n_hub = len(hub_pairs)

    ref = 0
    m   = n - 1

    mask_both   = (i_arr != ref) & (j_arr != ref)
    mask_only_j = (i_arr == ref)

    fi_both  = (i_arr[mask_both]   - 1).astype(np.intp)
    fj_both  = (j_arr[mask_both]   - 1).astype(np.intp)
    fj_only  = (j_arr[mask_only_j] - 1).astype(np.intp)
    idx_both   = np.where(mask_both)[0]
    idx_only_j = np.where(mask_only_j)[0]

    for _ in range(num_iters):
        C_arr = D / L_arr

        L_red = np.zeros((m, m))
        C_b   = C_arr[idx_both]
        np.add.at(L_red, (fi_both, fi_both),  C_b)
        np.add.at(L_red, (fj_both, fj_both),  C_b)
        np.add.at(L_red, (fi_both, fj_both), -C_b)
        np.add.at(L_red, (fj_both, fi_both), -C_b)
        np.add.at(L_red, (fj_only, fj_only),  C_arr[idx_only_j])
        L_red += 1e-12 * np.eye(m)
        lu, piv = lu_factor(L_red, check_finite=False)

        if demand == "fixed":
            srcs = np.full(batch_size, source, dtype=np.intp)
            snks = np.full(batch_size, sink,   dtype=np.intp)
        elif demand == "uniform":
            idxs = rng.integers(n_pairs, size=batch_size)
            srcs = all_pairs[idxs, 0]
            snks = all_pairs[idxs, 1]
        elif demand == "hubspoke":
            idxs = rng.integers(n_hub, size=batch_size)
            srcs = hub_pairs[idxs, 0]
            snks = hub_pairs[idxs, 1]
        elif demand == "mixed":
            idxs_uni = rng.integers(n_pairs, size=batch_size)
            use_fixed = rng.random(batch_size) < (1.0 - lambda_mix)
            srcs = np.where(use_fixed, source, all_pairs[idxs_uni, 0]).astype(np.intp)
            snks = np.where(use_fixed, sink,   all_pairs[idxs_uni, 1]).astype(np.intp)
        else:
            raise ValueError(f"Unknown demand type: {demand!r}")

        B = np.zeros((m, batch_size))
        for k in range(batch_size):
            s, t = int(srcs[k]), int(snks[k])
            if s != ref:
                B[s - 1, k] += 1.0
            if t != ref:
                B[t - 1, k] -= 1.0

        P_free = lu_solve((lu, piv), B, check_finite=False)

        P = np.zeros((n, batch_size))
        P[1:, :] = P_free

        Q_batch = C_arr[:, np.newaxis] * (P[i_arr, :] - P[j_arr, :])

        avg_Q_mu = np.mean(np.abs(Q_batch) ** mu, axis=1)

        D = D + dt * (avg_Q_mu - D)
        D = np.maximum(D, 1e-10)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for idx, (a, b) in enumerate(edge_list):
        if D[idx] > threshold:
            G.add_edge(a, b, weight=float(L_arr[idx]))

    if not nx.is_connected(G):
        G = reconnect_graph(G, points)

    return G
