from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx
from scipy.linalg import lu_factor, lu_solve, LinAlgError

from .graph_generation import reconnect_graph
from .metrics import (
    compute_cost,
    compute_efficiency,
    compute_robustness,
    compute_algebraic_connectivity,
)

_SHAPE_SCALE = 0.05
_MU_LOW      = 0.1
_MU_HIGH     = 3.0


def _farthest_pair(points: np.ndarray) -> Tuple[int, int]:
    n = len(points)
    best_d, src, snk = -1.0, 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            if d > best_d:
                best_d, src, snk = d, i, j
    return src, snk


class PhysarumEnv:

    FEAT_DIM = 10

    def __init__(
        self,
        n_range: Tuple[int, ...] = (20, 50),
        k_range: Tuple[int, ...] = (1, 2, 3, 5),
        H: int = 10,
        physarum_iters_per_step: int = 50,
        dt: float = 0.1,
        threshold: float = 0.05,
        k_max: int = 5,
    ):
        self.n_range  = n_range
        self.k_range  = k_range
        self.H        = H
        self.iters    = physarum_iters_per_step
        self.dt       = dt
        self.threshold = threshold
        self.k_max    = k_max

        self.n = self.k = self.E = self.m = 0
        self.points = self.D = self.Q_last = None
        self.edge_list = self.i_arr = self.j_arr = self.L_arr = None
        self.pairs = self.b_reds = self.w = None
        self.fi_both = self.fj_both = self.fj_only = None
        self.idx_both = self.idx_only_j = None
        self.mst_cost = self.mst_efficiency = 0.0
        self.step_idx = 0

    def reset(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.n = int(rng.choice(self.n_range))
        self.k = int(rng.choice(self.k_range))
        self.w = rng.dirichlet([1.0, 1.0, 1.0]).astype(np.float32)

        self.points = rng.uniform(0, 1, size=(self.n, 2))

        edge_list  = list(combinations(range(self.n), 2))
        self.edge_list = edge_list
        self.E = len(edge_list)
        self.i_arr = np.array([e[0] for e in edge_list], dtype=np.intp)
        self.j_arr = np.array([e[1] for e in edge_list], dtype=np.intp)
        self.L_arr = np.maximum(
            np.array([np.linalg.norm(self.points[i] - self.points[j])
                      for i, j in edge_list]),
            1e-10,
        )

        G_c = nx.Graph()
        for idx, (i, j) in enumerate(edge_list):
            G_c.add_edge(i, j, weight=float(self.L_arr[idx]))
        G_mst = nx.minimum_spanning_tree(G_c, weight="weight")
        self.mst_cost = float(sum(d["weight"] for _, _, d in G_mst.edges(data=True)))
        self.mst_efficiency = compute_efficiency(G_mst)

        k_eff = min(self.k, self.n // 2)
        self.k = k_eff
        if k_eff == 1:
            self.pairs = [_farthest_pair(self.points)]
        else:
            perm = rng.permutation(self.n)
            self.pairs = [(int(perm[c]), int(perm[k_eff + c])) for c in range(k_eff)]

        self.D = rng.uniform(0.5, 1.0, size=self.E)

        self._setup_pressure_solve()
        self._setup_rhs()

        self.Q_last = np.zeros(self.E, dtype=np.float64)
        self.step_idx = 0

        return self._get_obs()

    def step(self, mu_actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        mu_arr = np.clip(np.asarray(mu_actions, dtype=np.float64), _MU_LOW, _MU_HIGH)

        total_abs_Q = np.zeros(self.E)
        for _ in range(self.iters):
            total_abs_Q = self._solve_pressures()
            self.D = self.D + self.dt * (total_abs_Q ** mu_arr - self.D)
            self.D = np.maximum(self.D, 1e-10)
        self.Q_last = total_abs_Q

        self.step_idx += 1
        done = self.step_idx >= self.H

        shaped = self._shaped_reward()
        if done:
            final_r, info = self._final_reward()
            reward = shaped + final_r
        else:
            reward = shaped
            info = {}

        return self._get_obs(), float(reward), done, info

    def _setup_pressure_solve(self) -> None:
        self.m = self.n - 1
        i_arr, j_arr = self.i_arr, self.j_arr

        mask_both   = (i_arr != 0) & (j_arr != 0)
        mask_only_j = (i_arr == 0)

        self.fi_both   = (i_arr[mask_both]   - 1).astype(np.intp)
        self.fj_both   = (j_arr[mask_both]   - 1).astype(np.intp)
        self.fj_only   = (j_arr[mask_only_j] - 1).astype(np.intp)
        self.idx_both   = np.where(mask_both)[0]
        self.idx_only_j = np.where(mask_only_j)[0]

    def _setup_rhs(self) -> None:
        b_reds = np.zeros((self.k, self.m))
        for c, (src, snk) in enumerate(self.pairs):
            if src != 0:
                b_reds[c, src - 1] += 1.0
            if snk != 0:
                b_reds[c, snk - 1] -= 1.0
        self.b_reds = b_reds

    def _solve_pressures(self) -> np.ndarray:
        C_arr = self.D / self.L_arr
        m = self.m

        L_red = np.zeros((m, m))
        C_b   = C_arr[self.idx_both]
        np.add.at(L_red, (self.fi_both, self.fi_both),  C_b)
        np.add.at(L_red, (self.fj_both, self.fj_both),  C_b)
        np.add.at(L_red, (self.fi_both, self.fj_both), -C_b)
        np.add.at(L_red, (self.fj_both, self.fi_both), -C_b)
        np.add.at(L_red, (self.fj_only, self.fj_only),  C_arr[self.idx_only_j])
        L_red += 1e-12 * np.eye(m)

        try:
            lu, piv = lu_factor(L_red, check_finite=False)
        except (LinAlgError, ValueError):
            return self.Q_last.copy()

        total_abs_Q = np.zeros(self.E)
        p = np.empty(self.n)
        for c in range(self.k):
            try:
                p_free = lu_solve((lu, piv), self.b_reds[c], check_finite=False)
            except (LinAlgError, ValueError):
                continue
            p[0]  = 0.0
            p[1:] = p_free
            Q_c   = C_arr * (p[self.i_arr] - p[self.j_arr])
            total_abs_Q += np.abs(Q_c)

        return total_abs_Q

    def _get_obs(self) -> np.ndarray:
        Q = self.Q_last
        D = self.D
        C = D / self.L_arr

        max_Q = Q.max() + 1e-10
        max_D = D.max() + 1e-10
        max_C = C.max() + 1e-10
        mean_Q = Q.mean() + 1e-10

        D_surv = D[D > self.threshold]
        if len(D_surv) > 1:
            p   = D_surv / D_surv.sum()
            H_D = float(-np.sum(p * np.log(p + 1e-15))) / np.log(len(D_surv))
        else:
            H_D = 0.0

        t_frac = float(self.step_idx) / self.H
        k_norm = float(self.k) / float(self.k_max)

        obs = np.column_stack([
            Q / max_Q,
            D / max_D,
            C / max_C,
            Q / mean_Q,
            np.full(self.E, t_frac),
            np.full(self.E, H_D),
            np.full(self.E, k_norm),
            np.full(self.E, self.w[0]),
            np.full(self.E, self.w[1]),
            np.full(self.E, self.w[2]),
        ]).astype(np.float32)

        return obs

    def _shaped_reward(self) -> float:
        D = self.D
        surviving = D > self.threshold
        n_surv = int(np.sum(surviving))

        if n_surv == 0:
            return float(-_SHAPE_SCALE)

        proxy_cost   = float(np.sum(self.L_arr[surviving]))
        steiner_proxy = proxy_cost / self.mst_cost
        cost_signal  = max(0.0, 1.0 - (steiner_proxy - 1.0) / 4.0)

        D_s   = D[surviving]
        p     = D_s / D_s.sum()
        H_D   = float(-np.sum(p * np.log(p + 1e-15)))
        H_norm = H_D / np.log(n_surv) if n_surv > 1 else 0.0

        gamma = max(0, n_surv - (self.n - 1)) / max(1, self.E - (self.n - 1))

        shaped = (
            self.w[0] * cost_signal
            + self.w[1] * float(H_norm)
            + self.w[2] * float(gamma)
        )
        return float(shaped) * _SHAPE_SCALE

    def _final_reward(self) -> Tuple[float, dict]:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for idx, (i, j) in enumerate(self.edge_list):
            if self.D[idx] > self.threshold:
                G.add_edge(i, j, weight=float(self.L_arr[idx]))
        if not nx.is_connected(G):
            G = reconnect_graph(G, self.points)

        cost    = compute_cost(G)
        rho     = cost / self.mst_cost
        lambda2 = compute_algebraic_connectivity(G)
        rob     = compute_robustness(G)
        eff     = compute_efficiency(G)

        cost_score    = float(max(0.0, 1.0 - (rho - 1.0) / 4.0))
        lambda2_score = float(min(lambda2, 5.0) / 5.0)
        rob_score     = float(rob)

        r = float(
            self.w[0] * cost_score
            + self.w[1] * lambda2_score
            + self.w[2] * rob_score
        )

        info = {
            "cost":                   float(cost),
            "steiner_ratio":          float(rho),
            "algebraic_connectivity": float(lambda2),
            "robustness":             float(rob),
            "efficiency":             float(eff),
            "n":   self.n,
            "k":   self.k,
            "w":   self.w.tolist(),
        }
        return r, info
