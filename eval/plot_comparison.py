"""
Training curve comparison: PPO, MoE-PPO, SAC, TD3, FastTD3
Usage: python plot_comparison.py [--out path/to/output.pdf]
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import os
# Point IGE_DIR at your IsaacGymEnvs/isaacgymenvs checkout, or edit this directly.
RUNS_DIR = os.path.join(os.environ.get("IGE_DIR", "."), "runs")

# PPO and MoE-PPO use the exact n=5 paper seeds (tab:results_main); all runs
# verified complete (6000 iters ~1179M env steps) with final rewards/step
# matching results_v3_tables.txt. Seed IDs noted in comments.
ALGO_RUNS = {
    "PPO": [
        f"{RUNS_DIR}/DiabloBalanceGrasp_ppp_6000_seed2021_10-23-10-14",  # 2021
        f"{RUNS_DIR}/DiabloBalanceGrasp_ppp_6000_1337_13-13-43-58",      # 1337
        f"{RUNS_DIR}/PPO_seed789_27-01-16-05",                          # 789
        f"{RUNS_DIR}/PPO_seed123_26-15-21-54",                          # 123
        f"{RUNS_DIR}/PPO_seed42_26-15-21-54",                           # 42
    ],
    "MoE-PPO (k=2)": [
        f"{RUNS_DIR}/DiabloBalanceGrasp_moeELU_k2_seed1337_21-20-54-01",  # 1337
        f"{RUNS_DIR}/ELU_k2_seed777_30-01-12-26",                        # 777
        f"{RUNS_DIR}/ELU_k2_seed789_29-18-43-45",                        # 789
        f"{RUNS_DIR}/ELU_k2_seed2024_01-15-57-27",                       # 2024
        f"{RUNS_DIR}/ELU_k2_seed3407_01-18-21-57",                       # 3407
    ],
    # SAC/TD3: 3 seeds each, all run to the same 1179.6M-step budget as the
    # on-policy methods. All fail to learn (SAC peaks ~8.4k, TD3 ~2.9k), so the
    # failure is reported over n=3 rather than a single run.
    "SAC": [
        f"{RUNS_DIR}/DiabloBalanceGraspSAC_11-19-54-50",
        f"{RUNS_DIR}/DiabloBalanceGraspSAC_2021_12-18-05-44",  # 2021
        f"{RUNS_DIR}/BalanceGraspSAC_4242_13-02-45-03",        # 4242
    ],
    "TD3": [
        f"{RUNS_DIR}/DiabloBalanceGraspTD3_12-05-29-13",
        f"{RUNS_DIR}/DiabloBalanceGraspTD3_2021_12-14-03-52",  # 2021
        f"{RUNS_DIR}/DiabloBalanceGraspTD3_4242_12-22-37-19",  # 4242
    ],
    "FastTD3": [
        f"{RUNS_DIR}/FastTD3/IGE-DiabloBalanceGrasp__FastTD3__1337",
        f"{RUNS_DIR}/FastTD3/IGE-DiabloBalanceGrasp__FastTD3__2021",
        f"{RUNS_DIR}/FastTD3/IGE-DiabloBalanceGrasp__FastTD3__789",
        f"{RUNS_DIR}/FastTD3/IGE-DiabloBalanceGrasp__FastTD3__123",
        f"{RUNS_DIR}/FastTD3/IGE-DiabloBalanceGrasp__FastTD3__42",
    ],
}

COLORS = {
    "PPO":            "#1f77b4",   # blue
    "MoE-PPO (k=2)":  "#ff7f0e",   # orange
    "SAC":            "#2ca02c",   # green
    "TD3":            "#d62728",   # red
    "FastTD3":        "#9467bd",   # purple
}

# Per-algo TensorBoard metric + x-axis scaling. rl_games logs "rewards/step"
# already in env-step units; FastTD3 logs "episode_reward" indexed by iteration
# (1 iter = num_envs = 1024 samples), so multiply its step by 1024.
ALGO_CFG = {
    "FastTD3": {"tag": "episode_reward", "x_scale": 1024.0},
}
DEFAULT_CFG = {"tag": "rewards/step", "x_scale": 1.0}

METRIC = "rewards/step"
SIGMA  = 30    # gaussian smoothing kernel (in data-point units)
N_PTS  = 600   # resample to this many points for alignment


def _scalar_dir(path: str) -> str:
    """rl_games writes events under <run>/summaries; extra-seed and FastTD3
    runs may keep them directly in <run>. Return the dir holding event files."""
    if glob.glob(os.path.join(path, "events.out.tfevents*")):
        return path
    sub = os.path.join(path, "summaries")
    if os.path.isdir(sub):
        return sub
    return path


def load_run(path: str, metric: str, x_scale: float = 1.0):
    ea = EventAccumulator(_scalar_dir(path), size_guidance={"scalars": 0})
    ea.Reload()
    if metric not in ea.Tags()["scalars"]:
        return None, None
    scalars = ea.Scalars(metric)
    steps  = np.array([s.step  for s in scalars], dtype=float) * x_scale
    values = np.array([s.value for s in scalars], dtype=float)
    return steps, values


def smooth(values, sigma):
    return gaussian_filter1d(values, sigma=sigma)


def resample(steps, values, x_new):
    return np.interp(x_new, steps, values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="training_comparison.pdf")
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--fasttd3-eval", action="store_true",
                        help="Plot FastTD3's deterministic eval_avg_return "
                             "(noise-free policy) instead of the noisy "
                             "collection-time episode_reward. Fairer vs PPO/MoE.")
    args = parser.parse_args()

    if args.fasttd3_eval:
        ALGO_CFG["FastTD3"] = {"tag": "eval_avg_return", "x_scale": 1024.0}

    # Find global x range (env steps)
    max_step = 0.0
    for algo, paths in ALGO_RUNS.items():
        cfg = ALGO_CFG.get(algo, DEFAULT_CFG)
        for p in paths:
            s, _ = load_run(p, cfg["tag"], cfg["x_scale"])
            if s is not None and len(s) > 0:
                max_step = max(max_step, s[-1])

    x_common = np.linspace(0, max_step, N_PTS)
    x_common_M = x_common / 1e6

    fig, ax = plt.subplots(figsize=(8, 5))

    for algo, paths in ALGO_RUNS.items():
        cfg = ALGO_CFG.get(algo, DEFAULT_CFG)
        curves = []
        for p in paths:
            steps, values = load_run(p, cfg["tag"], cfg["x_scale"])
            if steps is None or len(steps) < 10:
                print(f"  skip {p} (no data)")
                continue
            # Scale smoothing to series length: eval_avg_return has ~115 pts
            # vs ~6-11k for the step-wise scalars, so a fixed sigma would
            # flatten it. Cap at SIGMA; keep it light for short series.
            sigma_eff = min(SIGMA, max(1, len(values) // 20))
            y_smooth = smooth(values, sigma=sigma_eff)
            y_resampled = resample(steps, y_smooth, x_common)
            curves.append(y_resampled)

        if not curves:
            print(f"[{algo}] no valid runs found")
            continue

        curves = np.array(curves)
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0) if len(curves) > 1 else np.zeros_like(mean)
        color  = COLORS[algo]

        ax.plot(x_common_M, mean, color=color, label=algo, linewidth=1.8)
        ax.fill_between(x_common_M, mean - std, mean + std,
                        color=color, alpha=0.18, linewidth=0)
        print(f"[{algo}] seeds={len(curves)}, final={mean[-1]:.0f} ± {std[-1]:.0f}")

    ax.set_xlabel("Million environment steps", fontsize=12)
    ax.set_ylabel("Mean episode reward", fontsize=12)
    ax.set_title("Algorithm Comparison", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"
    ))
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
