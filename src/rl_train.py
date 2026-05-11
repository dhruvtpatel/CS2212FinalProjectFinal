import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .rl_env import PhysarumEnv
from .rl_policy import ActorCritic


@dataclass
class PPOConfig:
    n_range: Tuple[int, ...] = (20, 50)
    k_range: Tuple[int, ...] = (1, 2, 3, 5)
    H:       int = 10
    physarum_iters_per_step: int = 50

    num_updates:        int = 400
    episodes_per_update: int = 8
    ppo_epochs:         int = 4
    lr:                 float = 3e-4
    lr_final_factor:    float = 0.1

    gamma:     float = 1.0
    gae_lambda: float = 0.95
    clip_eps:  float = 0.2
    entropy_coef: float = 0.01
    value_coef:   float = 0.5
    max_grad_norm: float = 0.5

    hidden: int = 128
    depth:  int = 3

    log_interval:  int = 25
    save_interval: int = 100
    output_dir: str = "output"
    device: str = "cpu"


def _collect_rollout(
    env: PhysarumEnv,
    policy: ActorCritic,
    config: PPOConfig,
    rng: np.random.Generator,
) -> Tuple[list, list, list, list, list, list, list, list]:
    policy.eval()

    obs_list:    List[torch.Tensor] = []
    act_list:    List[torch.Tensor] = []
    lp_list:     List[torch.Tensor] = []
    val_list:    List[torch.Tensor] = []
    rew_list:    List[float]        = []
    done_list:   List[bool]         = []
    ep_returns:  List[float]        = []
    ep_infos:    List[dict]         = []

    for _ in range(config.episodes_per_update):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        obs    = env.reset(ep_rng)
        ep_ret = 0.0

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(config.device)

            with torch.no_grad():
                raw_action, mu_action, log_prob, value = policy.act(obs_t)

            obs, reward, done, info = env.step(mu_action.cpu().numpy())

            obs_list.append(obs_t.cpu())
            act_list.append(raw_action.cpu())
            lp_list.append(log_prob.cpu())
            val_list.append(value.cpu())
            rew_list.append(float(reward))
            done_list.append(done)
            ep_ret += float(reward)

        ep_returns.append(ep_ret)
        if info:
            ep_infos.append(info)

    return obs_list, act_list, lp_list, val_list, rew_list, done_list, ep_returns, ep_infos


def _compute_gae(
    rewards:   List[float],
    values:    List[float],
    dones:     List[bool],
    gamma:     float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae   = 0.0

    for t in reversed(range(T)):
        if dones[t]:
            next_val = 0.0
            last_gae = 0.0
        else:
            next_val = values[t + 1] if t + 1 < T else 0.0

        delta    = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * gae_lambda * (0.0 if dones[t] else last_gae)
        advantages[t] = last_gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def _ppo_update(
    policy:     ActorCritic,
    optimizer:  torch.optim.Optimizer,
    obs_list:   list,
    act_list:   list,
    old_lp_arr: np.ndarray,
    adv_arr:    np.ndarray,
    ret_arr:    np.ndarray,
    config:     PPOConfig,
) -> float:
    policy.train()

    T       = len(obs_list)
    old_lp  = torch.tensor(old_lp_arr, dtype=torch.float32)
    adv_t   = torch.tensor(adv_arr,   dtype=torch.float32)
    ret_t   = torch.tensor(ret_arr,   dtype=torch.float32)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    total_loss = 0.0
    indices = np.arange(T)

    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        optimizer.zero_grad()

        step_losses: List[torch.Tensor] = []
        for i in indices:
            new_lp, val, ent = policy.evaluate(obs_list[i].to(config.device),
                                               act_list[i].to(config.device))

            ratio = torch.exp(new_lp - old_lp[i])
            surr1 = ratio * adv_t[i]
            surr2 = torch.clamp(ratio,
                                 1.0 - config.clip_eps,
                                 1.0 + config.clip_eps) * adv_t[i]

            actor_loss = -torch.min(surr1, surr2)
            value_loss = (val - ret_t[i]).pow(2)
            step_loss  = (actor_loss
                          + config.value_coef  * value_loss
                          - config.entropy_coef * ent) / T
            step_losses.append(step_loss)

        epoch_loss = torch.stack(step_losses).sum()
        epoch_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
        optimizer.step()

        total_loss += epoch_loss.item()

    return total_loss / config.ppo_epochs


def train(
    config: Optional[PPOConfig] = None,
    seed: int = 42,
) -> ActorCritic:
    if config is None:
        config = PPOConfig()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = PhysarumEnv(
        n_range=config.n_range,
        k_range=config.k_range,
        H=config.H,
        physarum_iters_per_step=config.physarum_iters_per_step,
        k_max=int(max(config.k_range)),
    )

    sample_obs = env.reset(np.random.default_rng(0))
    feat_dim   = sample_obs.shape[1]

    policy = ActorCritic(
        feat_dim=feat_dim,
        hidden=config.hidden,
        depth=config.depth,
    ).to(config.device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=config.lr_final_factor,
        total_iters=config.num_updates,
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\n{'='*62}")
    print(f"  PPO per-edge Physarum policy — {config.num_updates} updates")
    print(f"  n ∈ {config.n_range}  k ∈ {config.k_range}  H={config.H}")
    print(f"  {config.episodes_per_update} ep/update  ×  {config.H} steps  "
          f"×  {config.physarum_iters_per_step} Physarum iters")
    print(f"  Policy params: {n_params:,}")
    print(f"{'='*62}\n")

    t0          = time.time()
    all_returns: List[float] = []

    for update in range(1, config.num_updates + 1):
        (obs_list, act_list, lp_list, val_list,
         rew_list, done_list,
         ep_returns, ep_infos) = _collect_rollout(env, policy, config, rng)

        old_lp_arr = np.array([lp.item() for lp in lp_list], dtype=np.float32)
        val_arr    = [float(v.item()) for v in val_list]

        adv_arr, ret_arr = _compute_gae(
            rew_list, val_arr, done_list, config.gamma, config.gae_lambda
        )

        loss = _ppo_update(
            policy, optimizer,
            obs_list, act_list,
            old_lp_arr, adv_arr, ret_arr,
            config,
        )

        scheduler.step()
        all_returns.extend(ep_returns)

        if update % config.log_interval == 0:
            mean_ret     = float(np.mean(ep_returns))
            mean_ret_100 = float(np.mean(all_returns[-100:])) if len(all_returns) >= 100 else float(np.mean(all_returns))
            elapsed      = time.time() - t0
            print(
                f"  update {update:>4}/{config.num_updates}"
                f"  ret={mean_ret:+.3f}"
                f"  ret100={mean_ret_100:+.3f}"
                f"  loss={loss:.4f}"
                f"  [{elapsed:.0f}s]"
            )
            if ep_infos:
                info = ep_infos[-1]
                print(
                    f"          last ep: n={info.get('n','?')}"
                    f" k={info.get('k','?')}"
                    f"  ρ={info.get('steiner_ratio', float('nan')):.3f}"
                    f"  λ₂={info.get('algebraic_connectivity', float('nan')):.3f}"
                    f"  rob={info.get('robustness', float('nan')):.3f}"
                )

        if update % config.save_interval == 0:
            os.makedirs(config.output_dir, exist_ok=True)
            ckpt = os.path.join(config.output_dir, f"rl_policy_upd{update}.pt")
            torch.save(policy.state_dict(), ckpt)
            print(f"  [checkpoint -> {ckpt}]")

    os.makedirs(config.output_dir, exist_ok=True)
    final_path = os.path.join(config.output_dir, "rl_policy_final.pt")
    torch.save(policy.state_dict(), final_path)
    print(f"\nTraining complete ({time.time()-t0:.0f}s). Policy -> {final_path}")
    return policy


def compare_rl_vs_physarum(
    policy:       ActorCritic,
    n_values:     Tuple[int, ...] = (20, 50),
    k_values:     Tuple[int, ...] = (1, 2, 3, 5),
    num_episodes: int = 50,
    seed:         int = 9999,
    device:       str = "cpu",
) -> pd.DataFrame:
    from .physarum import build_physarum_multicommodity, select_source_sink_pairs
    from .metrics import compute_all_metrics
    from .graph_generation import build_mst
    from .experiment import _PHY_ITERS, _PHY_THRESHOLD

    rng    = np.random.default_rng(seed)
    policy = policy.to(device)
    policy.eval()

    env = PhysarumEnv(
        n_range=n_values,
        k_range=k_values,
        k_max=int(max(k_values)),
    )

    records = []
    for ep in range(num_episodes):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        obs    = env.reset(ep_rng)

        n, k, points = env.n, env.k, env.points
        pairs, mst_cost = env.pairs, env.mst_cost

        done     = False
        ep_ret   = 0.0
        rl_info  = {}
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, mu_action, _, _ = policy.act(obs_t, deterministic=True)
            obs, reward, done, info = env.step(mu_action.cpu().numpy())
            ep_ret += float(reward)
            if info:
                rl_info = info

        phy_seed = int(rng.integers(0, 2**31))
        G_std    = build_physarum_multicommodity(
            points,
            k=k,
            source_sink_pairs=pairs,
            num_iters=_PHY_ITERS,
            mu=1.0,
            threshold=_PHY_THRESHOLD,
            seed=phy_seed,
        )
        std_metrics = compute_all_metrics(G_std, mst_cost=mst_cost)

        row = {"n": n, "k": k, "ep_return": ep_ret}
        for key, val in rl_info.items():
            if isinstance(val, (int, float)):
                row[f"rl_{key}"] = val
        for key, val in std_metrics.items():
            row[f"std_{key}"] = val

        records.append(row)

        if (ep + 1) % 10 == 0:
            print(f"  eval episode {ep+1}/{num_episodes}")

    return pd.DataFrame(records)


def print_rl_comparison(df: pd.DataFrame) -> None:
    print("\n--- RL policy vs. standard Physarum(μ=1) ---")
    metrics = ["steiner_ratio", "algebraic_connectivity", "robustness", "efficiency"]
    header  = f"{'metric':30s}  {'RL (mean±std)':22s}  {'Physarum(μ=1)':22s}  {'Δ':>8s}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        rl_col  = f"rl_{m}"
        std_col = f"std_{m}"
        if rl_col not in df.columns or std_col not in df.columns:
            continue
        rl_vals  = df[rl_col].dropna()
        std_vals = df[std_col].dropna()
        rl_m, rl_s   = rl_vals.mean(),  rl_vals.std()
        std_m, std_s = std_vals.mean(), std_vals.std()
        delta = rl_m - std_m
        print(
            f"  {m:28s}  {rl_m:+.4f} ± {rl_s:.4f}  "
            f"{std_m:+.4f} ± {std_s:.4f}  {delta:+.4f}"
        )
