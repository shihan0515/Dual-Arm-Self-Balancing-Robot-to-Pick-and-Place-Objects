# DIABLO Balance Grasp — Thesis Code

Source code for the thesis:
**"Mixture-of-Experts Proximal Policy Optimization for Self-Balancing Mobile Manipulation"**

Author: Neo Wang (conon068@gmail.com)

---

## Overview

This repository contains the reinforcement learning environment and training code for a DIABLO wheeled-balance robot performing pick-and-place tasks using:

- **PPO baseline** — standard single-head actor network
- **MoE-PPO** — mixture-of-experts actor with N expert heads and a learned gate network

The robot must simultaneously maintain inverted-pendulum balance while the arm grasps, lifts, and places objects onto a target platform.

---

## Directory Structure

```
├── tasks/
│   ├── diablo_balance_grasp.py          # Main task (PPO + MoE-PPO, 4-phase state machine)
│   └── diablo_balance_grasp_claude.py   # MoE-PPO task variant
│
├── cfg/
│   ├── task/
│   │   ├── DiabloBalanceGrasp.yaml      # Environment config (PPO)
│   │   └── DiabloBalanceGraspClaude.yaml # Environment config (MoE-PPO)
│   └── train/
│       ├── DiabloBalanceGraspPPO.yaml        # PPO training hyperparameters
│       └── DiabloBalanceGraspClaudePPO.yaml  # MoE-PPO training hyperparameters
│
├── network/
│   └── network_builder_moe.py           # Modified rl_games network builder with MoE support
│
├── eval/
│   ├── shell_eval_ppo.sh        # PPO evaluation script
│   ├── shell_eval_claude.sh     # MoE-PPO evaluation script
│   ├── run_all_eval.sh          # Run both evaluations + parse + plot
│   ├── parse_eval_to_excel.py   # Parse log files → Excel
│   ├── plot_reward_curves.py    # Plot training reward curves
│   └── plot_eval_results.py     # Plot evaluation bar charts
│
└── docs/
    ├── EVALUATION_GUIDE.md      # Step-by-step evaluation workflow
    ├── MoE_ELU_RESTORE.md       # How to restore ELU MoE architecture
    └── MoE_SIMPLE_RESTORE.md    # How to restore simple (linear) MoE architecture
```

---

## Task Design

### 4-Phase State Machine

| Phase | Name | Transition Condition |
|-------|------|----------------------|
| 0 | APPROACH | `dist_to_pregrasp < 0.25 m` AND balance stable |
| 1 | GRASP | `object_z − initial_z > 0.02 m` (object lifted) |
| 2 | PLACE | `dist_xy < 0.02 m` AND `dist_z < 0.02 m` |
| 3 | RELEASE | Episode end / success |

### Observation Space (72-dim)

```
progress(1), base_lin_vel(3), base_ang_vel(3), base_quat(4),
base_height_norm(1), leg_pos_norm(4), leg_vel(4), wheel_vel(2),
arm_pos_norm(4), arm_vel(4), eef_pos(3), eef_rot(4),
object_pos(3), object_rot(4), handle_pos(3),
eef→handle(3), robot→handle(3), phase(1), prev_actions(9),
platform_pos(3), rel_bottom_to_plat(3), object_one_hot(3)
```

### Action Space (9-dim)

```
[0]   chassis height command
[1]   chassis pitch command
[2]   left wheel velocity
[3]   right wheel velocity
[4–7] right arm joints (4 DOF: shoulder pitch, shoulder roll, elbow, wrist)
[8]   gripper open/close
```

### Reward Function (15 components)

| Component | Description |
|-----------|-------------|
| `balance` | Penalises pitch deviation from upright |
| `alive` | Per-step bonus for not falling |
| `height` | Gaussian reward for target chassis height (phase-dependent) |
| `approach` | Exponential distance reward to pre-grasp point (Phase 0) |
| `dist` | EEF-to-handle distance reward (Phase 1) |
| `rot` | Gripper alignment reward |
| `grasp` | Contact force reward |
| `lift` | Object height above initial position |
| `transport` | Object stability during transport |
| `placement` | XY + Z proximity to target platform |
| `release` | Reward for releasing at target |
| `retreat` | Post-placement arm retraction |
| `orient` | Object upright orientation |
| `grasp_balance` | Balance bonus while grasping |
| `success` | Sparse bonus on task completion |

---

## MoE-PPO Architecture

### Current checkpoint architecture (ELU version)

```
Shared Backbone:  obs(72) → Linear(512) → ELU → Linear(256) → ELU → Linear(128) → ELU

Expert heads × N:
  Linear(128 → 64) → ELU → Linear(64 → 9)

Gate:
  Linear(128 → 32) → ELU → Linear(32 → N) → Softmax

Output:
  mu = Σ gate_weight_i × expert_i(backbone_out)
```

See `docs/MoE_ELU_RESTORE.md` and `docs/MoE_SIMPLE_RESTORE.md` for architecture switching instructions.

---

## Installation

1. Install [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) following its official guide.
2. Install the **bundled rl_games** (contains the MoE patch) instead of the stock version:

```bash
pip install -e rl_games-1.6.1
```

> The only modification from upstream rl_games is `rl_games/algos_torch/network_builder.py`,
> which adds the `moe_num_actors` expert-head / gate logic.
> See `docs/MoE_ELU_RESTORE.md` and `docs/MoE_SIMPLE_RESTORE.md` for architecture details.

3. Install this package:

```bash
pip install -e .
```

---

## Training

```bash
# PPO baseline
python train.py task=DiabloBalanceGrasp train=DiabloBalanceGraspPPO \
    moe_num_actors=1 headless=True

# MoE-PPO (N=2 experts)
python train.py task=DiabloBalanceGrasp train=DiabloBalanceGraspPPO \
    moe_num_actors=2 headless=True
```

**Training hardware:** NVIDIA GeForce RTX 5070 Ti (16 GB),
2,048 parallel environments, ~7 hours for 6,000 epochs.

---

## Evaluation

See `docs/EVALUATION_GUIDE.md` for the full workflow.

```bash
# Quick start: run both PPO and MoE-PPO evaluation + parse + plot
bash eval/run_all_eval.sh
```

### Evaluation Metrics

| Metric | Condition |
|--------|-----------|
| Phase 2 Entry Rate | Object grasped and lifted (Phase 1→2 triggered) |
| Tight Contact Rate | `d_xy < 5 cm` AND `\|Δz\| < 2 cm` |
| Partial Success Rate | `d_xy < 5 cm` AND `Δz ∈ [−2, +1] cm` |
