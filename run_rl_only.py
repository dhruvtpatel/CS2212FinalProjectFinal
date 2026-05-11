import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

N_RANGE              = (20, 50)
K_RANGE              = (1, 2, 3, 5)
H                    = 10
PHYSARUM_ITERS_STEP  = 50
NUM_UPDATES          = 400
EPISODES_PER_UPDATE  = 8
PPO_EPOCHS           = 4
LR                   = 3e-4
LR_FINAL_FACTOR      = 0.1
GAMMA                = 1.0
GAE_LAMBDA           = 0.95
CLIP_EPS             = 0.2
ENTROPY_COEF         = 0.01
VALUE_COEF           = 0.5
MAX_GRAD_NORM        = 0.5
HIDDEN               = 128
DEPTH                = 3
SEED                 = 42
OUTPUT_DIR           = "output"
EVAL_EPISODES        = 100
EVAL_N               = (20, 50)
EVAL_K               = (1, 2, 3, 5)
LOG_EVERY            = 1
VERBOSE_EVERY        = 50
CHECKPOINT_EVERY     = 100

from src.rl_env import PhysarumEnv
from src.rl_policy import ActorCritic
from src.physarum import build_physarum_multicommodity, select_source_sink_pairs
from src.metrics import compute_all_metrics
from src.graph_generation import build_mst
from src.experiment import _PHY_ITERS, _PHY_THRESHOLD


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < T else 0.0)
        delta    = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
        adv[t]   = last_gae
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def collect_rollout(env, policy, rng, n_episodes, device, verbose_ep=False):
    policy.eval()
    obs_list, act_list, lp_list, val_list = [], [], [], []
    rew_list, done_list = [], []
    ep_returns, ep_infos = [], []

    for ep_idx in range(n_episodes):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        obs    = env.reset(ep_rng)
        ep_ret = 0.0
        step   = 0
        done   = False

        if verbose_ep:
            n_ep, k_ep = env.n, env.k
            w_ep = env.w
            print(f"    ep {ep_idx+1}/{n_episodes}  n={n_ep} k={k_ep}  "
                  f"w=({w_ep[0]:.2f},{w_ep[1]:.2f},{w_ep[2]:.2f})  "
                  f"edges={obs.shape[0]}", flush=True)

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                raw_action, mu_action, log_prob, value = policy.act(obs_t)

            mu_np = mu_action.cpu().numpy()
            obs, reward, done, info = env.step(mu_np)

            obs_list.append(obs_t.cpu())
            act_list.append(raw_action.cpu())
            lp_list.append(log_prob.cpu())
            val_list.append(value.cpu())
            rew_list.append(float(reward))
            done_list.append(done)
            ep_ret += float(reward)

            if verbose_ep:
                print(f"      step {step+1}/{env.H}  "
                      f"reward={reward:+.4f}  "
                      f"μ min={mu_np.min():.3f} mean={mu_np.mean():.3f} "
                      f"max={mu_np.max():.3f}  "
                      f"logp={float(log_prob):.3f}  "
                      f"V={float(value):.3f}", flush=True)
            step += 1

        ep_returns.append(ep_ret)
        if info:
            ep_infos.append(info)
            if verbose_ep:
                rho = info.get("steiner_ratio", float("nan"))
                lam = info.get("algebraic_connectivity", float("nan"))
                rob = info.get("robustness", float("nan"))
                print(f"      → ep done  ret={ep_ret:+.4f}  "
                      f"ρ={rho:.3f}  λ₂={lam:.3f}  rob={rob:.3f}", flush=True)

    return obs_list, act_list, lp_list, val_list, rew_list, done_list, ep_returns, ep_infos


def ppo_update(policy, optimizer, obs_list, act_list, old_lp_arr,
               adv_arr, ret_arr, clip_eps, value_coef, entropy_coef,
               max_grad_norm, ppo_epochs, device):
    policy.train()
    T      = len(obs_list)
    old_lp = torch.tensor(old_lp_arr, dtype=torch.float32)
    adv_t  = torch.tensor(adv_arr, dtype=torch.float32)
    ret_t  = torch.tensor(ret_arr, dtype=torch.float32)
    adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    total_loss   = 0.0
    total_gnorm  = 0.0
    indices      = np.arange(T)

    for epoch in range(ppo_epochs):
        np.random.shuffle(indices)
        optimizer.zero_grad()
        step_losses = []

        for i in indices:
            new_lp, val, ent = policy.evaluate(
                obs_list[i].to(device), act_list[i].to(device)
            )
            ratio = torch.exp(new_lp - old_lp[i])
            surr1 = ratio * adv_t[i]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[i]
            loss  = (-torch.min(surr1, surr2)
                     + value_coef  * (val - ret_t[i]).pow(2)
                     - entropy_coef * ent) / T
            step_losses.append(loss)

        epoch_loss = torch.stack(step_losses).sum()
        epoch_loss.backward()
        gnorm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        total_loss  += epoch_loss.item()
        total_gnorm += float(gnorm)

    return total_loss / ppo_epochs, total_gnorm / ppo_epochs


def evaluate(policy, n_values, k_values, num_episodes, seed, device):
    rng = np.random.default_rng(seed)
    policy.eval()

    env = PhysarumEnv(
        n_range=n_values,
        k_range=k_values,
        k_max=int(max(k_values)),
    )

    records = []
    print(f"\n  Evaluating {num_episodes} episodes ...")

    for ep in range(num_episodes):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        obs    = env.reset(ep_rng)

        n_ep     = env.n
        k_ep     = env.k
        points   = env.points
        pairs    = env.pairs
        mst_cost = env.mst_cost

        done    = False
        ep_ret  = 0.0
        rl_info = {}
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, mu_action, _, _ = policy.act(obs_t, deterministic=True)
            mu_np = mu_action.cpu().numpy()
            obs, reward, done, info = env.step(mu_np)
            ep_ret += float(reward)
            if info:
                rl_info = info

        phy_seed = int(rng.integers(0, 2**31))
        G_std    = build_physarum_multicommodity(
            points,
            k=k_ep,
            source_sink_pairs=pairs,
            num_iters=_PHY_ITERS,
            mu=1.0,
            threshold=_PHY_THRESHOLD,
            seed=phy_seed,
        )
        std_m = compute_all_metrics(G_std, mst_cost=mst_cost)

        row = {"n": n_ep, "k": k_ep, "ep_return": ep_ret}
        for key, val in rl_info.items():
            if isinstance(val, (int, float)):
                row[f"rl_{key}"] = val
        for key, val in std_m.items():
            row[f"std_{key}"] = val
        records.append(row)

        if (ep + 1) % 10 == 0:
            rl_rho  = rl_info.get("steiner_ratio", float("nan"))
            std_rho = std_m.get("steiner_ratio", float("nan"))
            rl_lam  = rl_info.get("algebraic_connectivity", float("nan"))
            std_lam = std_m.get("algebraic_connectivity", float("nan"))
            print(f"  ep {ep+1:>3}/{num_episodes}  n={n_ep} k={k_ep}  "
                  f"ret={ep_ret:+.3f}  "
                  f"RL ρ={rl_rho:.3f} λ₂={rl_lam:.3f}  "
                  f"Phy ρ={std_rho:.3f} λ₂={std_lam:.3f}")

    return pd.DataFrame(records)


def print_eval_table(df):
    metrics = ["steiner_ratio", "algebraic_connectivity", "robustness", "efficiency"]
    print(f"\n{'─'*75}")
    print(f"  {'metric':30s}  {'RL (mean±std)':22s}  {'Physarum μ=1':22s}  {'Δ':>8s}")
    print(f"{'─'*75}")
    for m in metrics:
        rc, sc = f"rl_{m}", f"std_{m}"
        if rc not in df.columns or sc not in df.columns:
            print(f"  {m:30s}  (not in eval output)")
            continue
        rl_v  = df[rc].dropna()
        std_v = df[sc].dropna()
        rm, rs   = rl_v.mean(), rl_v.std()
        sm, ss   = std_v.mean(), std_v.std()
        delta    = rm - sm
        arrow    = "▲" if delta > 0 else "▼"
        print(f"  {m:30s}  {rm:+.4f} ± {rs:.4f}    "
              f"{sm:+.4f} ± {ss:.4f}    {delta:+.4f} {arrow}")
    print(f"{'─'*75}")

    print(f"\n  Per (n, k) breakdown:")
    for (n, k), grp in df.groupby(["n", "k"]):
        rl_rho  = grp["rl_steiner_ratio"].mean()  if "rl_steiner_ratio"  in grp else float("nan")
        std_rho = grp["std_steiner_ratio"].mean() if "std_steiner_ratio" in grp else float("nan")
        rl_lam  = grp["rl_algebraic_connectivity"].mean()  if "rl_algebraic_connectivity"  in grp else float("nan")
        std_lam = grp["std_algebraic_connectivity"].mean() if "std_algebraic_connectivity" in grp else float("nan")
        rl_rob  = grp["rl_robustness"].mean()  if "rl_robustness"  in grp else float("nan")
        std_rob = grp["std_robustness"].mean() if "std_robustness" in grp else float("nan")
        print(f"    n={n:<3} k={k}  "
              f"ρ  RL={rl_rho:.3f} Phy={std_rho:.3f} Δ={rl_rho-std_rho:+.3f}  |  "
              f"λ₂ RL={rl_lam:.3f} Phy={std_lam:.3f} Δ={rl_lam-std_lam:+.3f}  |  "
              f"R  RL={rl_rob:.3f} Phy={std_rob:.3f} Δ={rl_rob-std_rob:+.3f}")


def main():
    t_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cpu"

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("=" * 65)
    print("  Experiment 5: PPO per-edge Physarum policy (full training)")
    print(f"  n ∈ {N_RANGE}  k ∈ {K_RANGE}  H={H}  iters/step={PHYSARUM_ITERS_STEP}")
    print(f"  {NUM_UPDATES} updates × {EPISODES_PER_UPDATE} ep/update × {H} steps")
    total_physarum_runs = NUM_UPDATES * EPISODES_PER_UPDATE
    print(f"  Total Physarum simulations (training): ~{total_physarum_runs:,}")
    print("=" * 65)

    print("\n[setup] building environment ... ", end="", flush=True)
    env = PhysarumEnv(
        n_range=N_RANGE,
        k_range=K_RANGE,
        H=H,
        physarum_iters_per_step=PHYSARUM_ITERS_STEP,
        k_max=int(max(K_RANGE)),
    )
    sample_obs = env.reset(np.random.default_rng(0))
    feat_dim   = sample_obs.shape[1]
    print(f"done  feat_dim={feat_dim}  "
          f"sample edges={sample_obs.shape[0]} (n={env.n}, k={env.k})")

    print("[setup] building policy ... ", end="", flush=True)
    policy = ActorCritic(feat_dim=feat_dim, hidden=HIDDEN, depth=DEPTH).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"done  params={n_params:,}  hidden={HIDDEN}  depth={DEPTH}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=LR_FINAL_FACTOR,
        total_iters=NUM_UPDATES,
    )
    print(f"[setup] Adam lr={LR:.0e} → {LR*LR_FINAL_FACTOR:.0e} over {NUM_UPDATES} updates")
    print(f"        PPO clip={CLIP_EPS}  entropy={ENTROPY_COEF}  "
          f"value={VALUE_COEF}  grad_clip={MAX_GRAD_NORM}")

    print(f"\n{'─'*65}")
    print(f"  Starting training  (log every {LOG_EVERY}, "
          f"verbose every {VERBOSE_EVERY}, checkpoint every {CHECKPOINT_EVERY})")
    print(f"{'─'*65}\n")

    t_train       = time.time()
    all_returns   = []
    all_losses    = []
    all_gnorms    = []
    update_times  = []

    for update in range(1, NUM_UPDATES + 1):
        t_upd = time.time()
        verbose_ep = (update % VERBOSE_EVERY == 0)

        if verbose_ep:
            print(f"\n{'━'*65}")
            print(f"  UPDATE {update}/{NUM_UPDATES}  [verbose rollout]")
            print(f"{'━'*65}")

        (obs_list, act_list, lp_list, val_list,
         rew_list, done_list,
         ep_returns, ep_infos) = collect_rollout(
            env, policy, rng,
            n_episodes=EPISODES_PER_UPDATE,
            device=device,
            verbose_ep=verbose_ep,
        )

        old_lp_arr = np.array([lp.item() for lp in lp_list], dtype=np.float32)
        val_arr    = [float(v.item()) for v in val_list]
        adv_arr, ret_arr = compute_gae(
            rew_list, val_arr, done_list, GAMMA, GAE_LAMBDA
        )

        if verbose_ep:
            print(f"\n  GAE  adv: mean={adv_arr.mean():+.4f}  "
                  f"std={adv_arr.std():.4f}  "
                  f"min={adv_arr.min():+.4f}  max={adv_arr.max():+.4f}")
            print(f"       ret: mean={ret_arr.mean():+.4f}  "
                  f"std={ret_arr.std():.4f}")
            print(f"       steps in rollout: {len(rew_list)}")

        loss, gnorm = ppo_update(
            policy, optimizer,
            obs_list, act_list,
            old_lp_arr, adv_arr, ret_arr,
            CLIP_EPS, VALUE_COEF, ENTROPY_COEF,
            MAX_GRAD_NORM, PPO_EPOCHS, device,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        upd_elapsed = time.time() - t_upd

        all_returns.extend(ep_returns)
        all_losses.append(loss)
        all_gnorms.append(gnorm)
        update_times.append(upd_elapsed)

        mu_samples = []
        ep_obs = obs_list[-H:] if len(obs_list) >= H else obs_list
        policy.eval()
        with torch.no_grad():
            for obs_t in ep_obs:
                _, mu_a, _, _ = policy.act(obs_t.to(device))
                mu_samples.append(mu_a.cpu().numpy())
        policy.train()
        if mu_samples:
            all_mu = np.concatenate(mu_samples)
            mu_min, mu_mean, mu_max = all_mu.min(), all_mu.mean(), all_mu.max()
        else:
            mu_min = mu_mean = mu_max = float("nan")

        if update % LOG_EVERY == 0 or update == 1:
            elapsed = time.time() - t_train
            avg_upd = np.mean(update_times[-20:])
            eta     = avg_upd * (NUM_UPDATES - update)
            mean_ret_now   = float(np.mean(ep_returns))
            mean_ret_100   = float(np.mean(all_returns[-100:])) if len(all_returns) >= 100 else float(np.mean(all_returns))
            mean_loss_20   = float(np.mean(all_losses[-20:]))
            mean_gnorm_20  = float(np.mean(all_gnorms[-20:]))

            print(f"  upd {update:>4}/{NUM_UPDATES}"
                  f"  ret={mean_ret_now:+.3f}"
                  f"  ret100={mean_ret_100:+.3f}"
                  f"  loss={mean_loss_20:.4f}"
                  f"  gnorm={mean_gnorm_20:.3f}"
                  f"  lr={current_lr:.2e}"
                  f"  μ∈[{mu_min:.2f},{mu_max:.2f}] avg={mu_mean:.2f}"
                  f"  {upd_elapsed:.1f}s/upd"
                  f"  ETA {eta/60:.1f}min", flush=True)

        if verbose_ep:
            print(f"\n  Update {update} detailed stats:")
            print(f"    loss (this update)  = {loss:.5f}")
            print(f"    grad norm           = {gnorm:.4f}")
            print(f"    ep returns this upd = {[f'{r:+.3f}' for r in ep_returns]}")
            print(f"    return mean/std     = {np.mean(ep_returns):+.4f} / {np.std(ep_returns):.4f}")
            print(f"    action μ  min/mean/max = {mu_min:.3f} / {mu_mean:.3f} / {mu_max:.3f}")
            print(f"    LR                  = {current_lr:.3e}")
            print(f"    Wall time so far    = {(time.time()-t_train)/60:.1f} min")

            if ep_infos:
                info = ep_infos[-1]
                print(f"    Last episode final:")
                print(f"      n={info.get('n','?')}  k={info.get('k','?')}")
                print(f"      steiner_ratio          = {info.get('steiner_ratio', float('nan')):.4f}")
                print(f"      algebraic_connectivity = {info.get('algebraic_connectivity', float('nan')):.4f}")
                print(f"      robustness             = {info.get('robustness', float('nan')):.4f}")
                print(f"      cyclomatic_number      = {info.get('cyclomatic_number', float('nan')):.1f}")
                print(f"      network_entropy        = {info.get('network_entropy', float('nan')):.4f}")

            window = min(50, len(all_returns))
            print(f"    Rolling ret (last {window} ep) = {np.mean(all_returns[-window:]):+.4f}")
            window_loss = min(20, len(all_losses))
            print(f"    Rolling loss (last {window_loss} upd) = {np.mean(all_losses[-window_loss:]):.5f}")

        if update % CHECKPOINT_EVERY == 0:
            ckpt = os.path.join(OUTPUT_DIR, f"rl_policy_upd{update}.pt")
            torch.save(policy.state_dict(), ckpt)
            elapsed = time.time() - t_train
            avg_upd = np.mean(update_times)
            eta     = avg_upd * (NUM_UPDATES - update)
            print(f"\n  [checkpoint] saved -> {ckpt}")
            print(f"  [checkpoint] {update}/{NUM_UPDATES} updates  "
                  f"wall={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
            print(f"  [checkpoint] mean ret (all) = {np.mean(all_returns):+.4f}  "
                  f"mean loss (all) = {np.mean(all_losses):.5f}\n")

    final_path = os.path.join(OUTPUT_DIR, "rl_policy_final.pt")
    torch.save(policy.state_dict(), final_path)
    train_elapsed = time.time() - t_train
    print(f"\n{'='*65}")
    print(f"  Training complete in {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")
    print(f"  Final policy -> {final_path}")
    print(f"  Mean return over all training: {np.mean(all_returns):+.4f}")
    print(f"  Mean return (last 100 ep):     {np.mean(all_returns[-100:]):+.4f}")
    print(f"  Mean loss (last 20 updates):   {np.mean(all_losses[-20:]):.5f}")

    print(f"\n{'─'*65}")
    print(f"  Evaluation: RL policy vs Physarum(μ=1)")
    print(f"  {EVAL_EPISODES} episodes  n∈{EVAL_N}  k∈{EVAL_K}")
    print(f"{'─'*65}")

    try:
        eval_df = evaluate(
            policy,
            n_values=EVAL_N,
            k_values=EVAL_K,
            num_episodes=EVAL_EPISODES,
            seed=9999,
            device=device,
        )
        csv_path = os.path.join(OUTPUT_DIR, "rl_evaluation.csv")
        eval_df.to_csv(csv_path, index=False)
        print(f"\n  Saved -> {csv_path}  ({len(eval_df)} rows)")
        print_eval_table(eval_df)
    except Exception as e:
        print(f"  Evaluation ERROR: {e}")
        traceback.print_exc()

    total_elapsed = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  All done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Outputs:")
    print(f"    {OUTPUT_DIR}/rl_policy_final.pt")
    for upd in range(CHECKPOINT_EVERY, NUM_UPDATES + 1, CHECKPOINT_EVERY):
        print(f"    {OUTPUT_DIR}/rl_policy_upd{upd}.pt")
    print(f"    {OUTPUT_DIR}/rl_evaluation.csv")
    print(f"\n  Copy these values into Table 3 of paper.tex:")
    if "eval_df" in dir() and eval_df is not None and len(eval_df) > 0:
        for col_pair, label in [
            ("rl_steiner_ratio",          "RL_RHO / STD_RHO"),
            ("rl_algebraic_connectivity", "RL_LAM / STD_LAM"),
            ("rl_robustness",             "RL_ROB / STD_ROB"),
            ("rl_efficiency",             "RL_EFF / STD_EFF"),
        ]:
            std_col = col_pair.replace("rl_", "std_")
            if col_pair in eval_df.columns and std_col in eval_df.columns:
                rm = eval_df[col_pair].mean()
                rs = eval_df[col_pair].std()
                sm = eval_df[std_col].mean()
                ss = eval_df[std_col].std()
                print(f"    [{label}]:  RL={rm:.3f}±{rs:.3f}  Phy={sm:.3f}±{ss:.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
