"""
Plot training reward curves from TensorBoard event files.

Usage:
    # Single run
    python plot_reward_curves.py <run_dir> [output_prefix]

    # Compare two runs (PPO vs MoE-PPO)
    python plot_reward_curves.py <ppo_run_dir> [claude_run_dir] [output_prefix]

Example:
    python plot_reward_curves.py \\
        runs/DiabloBalanceGrasp_Neo_object_vel_08-19-45-18 \\
        runs/DiabloBalanceGraspClaude_net_contact_08-19-43-58 \\
        thesis_reward
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Installing tensorboard...")
    os.system("pip install tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PPO_COLOR    = "#4C72B0"
CLAUDE_COLOR = "#DD8452"

# Scalars to plot (actual tag names used by this training setup)
REWARD_TAG   = "rewards/iter"
LENGTH_TAG   = "episode_lengths/iter"

# Phase progression metrics (easy → hard), ordered for thesis
EXTRA_TAGS   = [
    ("metrics/phase_1_rate/iter",  "Phase1: Grasp Rate"),
    ("metrics/phase_2_rate/iter",  "Phase2: Place Entry Rate"),
    ("metrics/phase_3_rate/iter",  "Phase3: Release Rate"),
    ("metrics/fall_rate/iter",     "Fall Rate"),
]

# Reward component breakdown
COMPONENT_TAGS = [
    ("rewards/grasp/iter",     "Grasp Reward"),
    ("rewards/lift/iter",      "Lift Reward"),
    ("rewards/height/iter",    "Height Reward"),
    ("rewards/placement/iter", "Placement Reward"),
    ("rewards/transport/iter", "Transport Reward"),
    ("rewards/balance/iter",   "Balance Reward"),
]


def load_scalars(run_dir: str) -> dict:
    """Load all scalars from TensorBoard event files in <run_dir>/summaries/"""
    summary_dir = Path(run_dir) / "summaries"
    if not summary_dir.exists():
        summary_dir = Path(run_dir)

    ea = EventAccumulator(str(summary_dir), size_guidance={"scalars": 0})
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    print(f"  Available scalars: {available}")

    data = {}
    for tag in available:
        events = ea.Scalars(tag)
        data[tag] = {
            "step":  np.array([e.step  for e in events]),
            "value": np.array([e.value for e in events]),
        }
    return data


def smooth(values: np.ndarray, weight: float = 0.92) -> np.ndarray:
    """Exponential moving average smoothing (same as TensorBoard)."""
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)


def plot_tag(ax, data: dict, tag: str, label: str, color: str,
             smooth_w: float = 0.92, alpha_raw: float = 0.15):
    if tag not in data:
        return False
    steps  = data[tag]["step"]
    values = data[tag]["value"]
    sm     = smooth(values, smooth_w)
    ax.plot(steps, values, color=color, alpha=alpha_raw, linewidth=0.8)
    ax.plot(steps, sm,     color=color, linewidth=2.0, label=label)
    return True


# ── Main reward + episode length figure ──────────────────────────────────────
def plot_main(runs: list, out_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for run_dir, label, color in runs:
        print(f"\nLoading {label}: {run_dir}")
        data = load_scalars(run_dir)
        plot_tag(ax, data, REWARD_TAG, label, color)

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
    print(f"\nSaved → {out_path}")
    plt.close()


# ── Custom metrics figure ─────────────────────────────────────────────────────
def plot_metrics(runs: list, out_path: str):
    # Filter to tags that actually exist in data
    first_data = load_scalars(runs[0][0])
    available = [(tag, lbl) for tag, lbl in EXTRA_TAGS if tag in first_data]

    if not available:
        print("No phase metrics found, skipping metrics plot.")
        return

    n   = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for run_dir, label, color in runs:
        data = load_scalars(run_dir)
        for ax, (tag, _) in zip(axes, available):
            plot_tag(ax, data, tag, label, color)

    for ax, (tag, title) in zip(axes, available):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Epoch", fontsize=10)
        ax.set_ylabel("Rate", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Phase Progression During Training", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── Reward component breakdown — one file per component ──────────────────────
def plot_reward_components(runs: list, prefix: str):
    first_data = load_scalars(runs[0][0])
    available = [(tag, lbl) for tag, lbl in COMPONENT_TAGS if tag in first_data]

    if not available:
        print("No reward component tags found, skipping components plot.")
        return

    # file-safe name mapping
    tag_to_filename = {
        "rewards/grasp/iter":     "grasp",
        "rewards/lift/iter":      "lift",
        "rewards/height/iter":    "height",
        "rewards/placement/iter": "placement",
        "rewards/transport/iter": "transport",
        "rewards/balance/iter":   "balance",
    }

    for tag, title in available:
        fig, ax = plt.subplots(figsize=(7, 5))

        for run_dir, label, color in runs:
            data = load_scalars(run_dir)
            plot_tag(ax, data, tag, label, color)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Epoch", fontsize=11)
        ax.set_ylabel("Reward", fontsize=11)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fname = tag_to_filename.get(tag, tag.replace("/", "_"))
        out_path = f"{prefix}_{fname}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    # Parse --labels label1,label2 (optional)
    labels_override = None
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "--labels" and i + 1 < len(args):
            labels_override = args[i + 1].split(",")
            i += 2
        else:
            filtered.append(args[i])
            i += 1
    args = filtered

    # Detect: last arg is output prefix if it doesn't look like a run dir
    run_dirs = []
    prefix   = "thesis_reward"

    for a in args:
        if Path(a).is_dir():
            run_dirs.append(a)
        else:
            prefix = a

    if not run_dirs:
        print("No valid run directories found.")
        sys.exit(1)

    colors = [PPO_COLOR, CLAUDE_COLOR, "#55A868", "#C44E52"]
    labels = labels_override if labels_override else ["PPO", "MoE-PPO", "Run3", "Run4"]

    runs = [(d, labels[i], colors[i]) for i, d in enumerate(run_dirs)]

    plot_main(runs,              f"{prefix}_main.png")
    plot_metrics(runs,           f"{prefix}_metrics.png")
    plot_reward_components(runs, prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
