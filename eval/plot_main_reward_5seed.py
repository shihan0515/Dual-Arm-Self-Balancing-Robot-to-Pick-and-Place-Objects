"""
Main training-reward figure for the thesis: mean +/- 1 std across the five
ELU MoE-PPO (k=2) seeds reported in Table tab:results_main.

Unlike plot_reward_curves.py (which plots a single run, with the faint band
being that run's *unsmoothed* data), the shaded band here really is the
across-seed standard deviation.

Usage: python plot_main_reward_5seed.py [out.png]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# The exact five seeds of results_v3_tables.txt / tab:results_main
SEED_RUNS = {
    1337: "runs/DiabloBalanceGrasp_moeELU_k2_seed1337_21-20-54-01",
     777: "runs/ELU_k2_seed777_30-01-12-26",
     789: "runs/ELU_k2_seed789_29-18-43-45",
    2024: "runs/ELU_k2_seed2024_01-15-57-27",
    3407: "runs/ELU_k2_seed3407_01-18-21-57",
}

TAG   = "rewards/iter"
COLOR = "#4C72B0"
EMA_W = 0.92


def smooth(v, w=EMA_W):
    out, last = [], v[0]
    for x in v:
        last = last * w + x * (1 - w)
        out.append(last)
    return np.array(out)


def load(run_dir):
    sd = os.path.join(run_dir, "summaries")
    ea = EventAccumulator(sd if os.path.isdir(sd) else run_dir,
                          size_guidance={"scalars": 0})
    ea.Reload()
    sc = ea.Scalars(TAG)
    return (np.array([e.step for e in sc], dtype=float),
            np.array([e.value for e in sc], dtype=float))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "thesis_reward_main_5seed.png"

    curves = []
    for seed, d in SEED_RUNS.items():
        steps, values = load(d)
        curves.append((steps, smooth(values)))
        print(f"  seed {seed:5d}: {len(steps)} epochs, final {curves[-1][1][-1]:.0f}")

    n = min(len(s) for s, _ in curves)
    grid = curves[0][0][:n]
    stack = np.vstack([y[:n] for _, y in curves])
    mean, std = stack.mean(0), stack.std(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(grid, mean - std, mean + std, color=COLOR, alpha=0.20, lw=0)
    ax.plot(grid, mean, color=COLOR, lw=2.0, label="MoE-PPO ($k$=2)")

    ax.set_title("Mean Episode Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Epoch", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.suptitle("Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")

    print("\n=== landmarks (across-seed mean) ===")
    for ep in [0, 250, 500, 750, 1000, 1500, 2000, 2200, 2500, 3000,
               3500, 4000, 4500, 5000, 5500, 6000]:
        i = min(ep, n - 1)
        print(f"  epoch {ep:5d}: {mean[i]:8.0f} +/- {std[i]:7.0f}")


if __name__ == "__main__":
    main()
