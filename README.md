# DIABLO Balance Grasp — Thesis Code

Source code for the thesis
**"Mixture-of-Experts Proximal Policy Optimization for Self-Balancing Mobile Manipulation."**

Author: Neo Wang (conon068@gmail.com)

A DIABLO wheeled-balance robot with a dual-arm upper body learns to pick an
object from a table and place it on a platform. The hard part is that the
chassis is an inverted pendulum: it has to keep itself upright while the arm
reaches, lifts, and places, and the payload keeps changing the dynamics.

## Demonstration videos

| | |
|---|---|
| **Scenario 1 — fixed-base manipulation** | [▶ watch](https://drive.google.com/file/d/1EChZMiL5lno2W-4UHKll3jUtnmPaTxEV/view) |
| **Scenario 2 — self-balancing mobile grasping** | [▶ watch](https://drive.google.com/file/d/13W73F0e0umDTXaq9M_lTE36xbI1EmzMv/view) |

Scenario 2 is the primary result: the chassis balances itself throughout, and
the base visibly adjusts its height and pitch to compensate for the payload
while the arm carries and places the object.

---

## Contents

This repository holds the **source files that matter for the thesis**: the two
task environments, their configs, the modified `rl_games` that implements the
Mixture-of-Experts actor, and the training and evaluation drivers.

It is a curated subset, not a standalone runnable package — there is no
`train.py` or `setup.py` here. Those come from NVIDIA's IsaacGymEnvs, into
which the files in this repository are dropped; the installation section below
walks through it.

The raw per-seed evaluation logs behind every number in the thesis are not
published here for size reasons. They are available on request
(conon068@gmail.com).

```
├── tasks/
│   ├── diablo_balance_grasp.py          # Scenario 2 — self-balancing mobile grasping
│   ├── diablo_balance_grasp_claude.py   #   variant used during development
│   └── diablo_graspcustom3.py           # Scenario 1 — fixed-base manipulation
│
├── cfg/
│   ├── task/                            # environment configs
│   └── train/                           # PPO / SAC / TD3 training configs
│
├── network/
│   └── network_builder_moe.py           # copy of the patched rl_games network builder
│
├── rl_games-1.6.1/                      # the patched rl_games (authoritative copy)
│   └── rl_games/algos_torch/
│       ├── network_builder.py           #   ← the MoE actor lives here
│       └── sac_agent.py, td3_agent.py   #   baselines
│
├── scripts/                             # training drivers (chain_*.sh)
├── eval/                                # evaluation drivers, plotting, aggregation
├── docs/                                # evaluation guide
└── assets/                              # robot and object URDFs
```

---

## Two experimental scenarios

| | Scenario 1 (fixed base) | Scenario 2 (self-balancing) |
|---|---|---|
| Chassis | rigidly fixed | free-floating, actively balanced |
| Task file | `diablo_graspcustom3.py` | `diablo_balance_grasp.py` |
| Action space | 5-D: 4 arm joints + gripper¹ | 9-D task space |
| Observation | 92-D | 72-D |
| Phase machine | none (single stage) | four phases |
| Metrics | SR, final L2 error | PSR, minL2 |

¹ The Scenario 1 config declares a 14-dimensional action vector for
compatibility with an earlier task definition; only five dimensions are read by
the controller.

Scenario 2 is the primary study. Scenario 1 isolates manipulation, to show the
MoE gain exists before any locomotion coupling is added.

---

## Installation

```bash
# 1. IsaacGym Preview 4 — https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .

# 2. NVIDIA IsaacGymEnvs, which provides train.py and the framework
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd IsaacGymEnvs && pip install -e . && cd ..

# 3. This repository
git clone https://github.com/shihan0515/Dual-Arm-Self-Balancing-Robot-to-Pick-and-Place-Objects.git thesis-code

# 4. Drop the task files, configs and assets into IsaacGymEnvs
cp thesis-code/tasks/*.py        IsaacGymEnvs/isaacgymenvs/tasks/
cp thesis-code/cfg/task/*.yaml   IsaacGymEnvs/isaacgymenvs/cfg/task/
cp thesis-code/cfg/train/*.yaml  IsaacGymEnvs/isaacgymenvs/cfg/train/
cp -r thesis-code/assets/urdf/*  IsaacGymEnvs/assets/urdf/

# 5. The bundled rl_games — NOT the PyPI version
cd thesis-code/rl_games-1.6.1 && pip install -e . && cd ../..

# 6. Every shell you train or evaluate in
export PYTHONPATH=/path/to/IsaacGymEnvs:$PYTHONPATH
```

Then register the tasks in `IsaacGymEnvs/isaacgymenvs/tasks/__init__.py`:

```python
from .diablo_balance_grasp import DiabloBalanceGrasp
from .diablo_graspcustom3 import DiabloGraspCustom3

isaacgym_task_map = {
    ...
    "DiabloBalanceGrasp":    DiabloBalanceGrasp,
    "DiabloBalanceGraspSAC": DiabloBalanceGrasp,
    "DiabloGraspCustom3":    DiabloGraspCustom3,
}
```

and add the MoE switches to `isaacgymenvs/cfg/config.yaml`:

```yaml
moe_num_actors: 1
moe_expert_hidden: 0    # >0 builds Linear→ELU→Linear expert heads
moe_gate_hidden: 0      # >0 builds Linear→ELU→Linear gate
```

**Step 5 is not optional.** The MoE actor exists only in the bundled
`rl_games`. Installing `rl_games` from PyPI does not error — it silently gives
you a plain MLP actor, and every MoE result collapses to the PPO baseline.

Verify with a short run; this line must appear at startup:

```
[MoE] 2 experts | expert: 128→64→9 | gate: 128→32→2
```

---

## Architecture switches

The actor variant is chosen entirely from the command line. The
`docs/MoE_*_RESTORE.md` files describe an older workflow that required editing
`network_builder.py` by hand; they are kept only as history and **should not be
followed**.

| Configuration | Flags |
|---|---|
| PPO baseline | *(none)* |
| MoE, simple linear heads | `moe_num_actors=K` |
| **MoE, ELU heads — the proposed method** | `moe_num_actors=K moe_expert_hidden=64 moe_gate_hidden=32` |

`moe_expert_hidden=0` (the default) builds each expert as a single
`Linear(128→9)` and the gate as `Linear(128→K)`. Values `>0` build
`Linear→ELU→Linear` heads. **Omitting the two hidden-size flags trains the
simple variant, not the proposed one** — in the thesis that variant converged on
only 3 of 5 seeds and scored 83.49 % macro PSR, against 93.08 % for the ELU
version. Checkpoints of the two variants are not interchangeable.

The shared backbone is `MLP [512→256→128]` with ELU in every configuration.

---

## Training

```bash
# PPO baseline
python train.py task=DiabloBalanceGrasp train=DiabloBalanceGraspPPO \
    headless=True seed=1337 experiment=PPO_seed1337

# Proposed: ELU MoE-PPO, k=2
python train.py task=DiabloBalanceGrasp train=DiabloBalanceGraspPPO \
    headless=True seed=1337 \
    moe_num_actors=2 moe_expert_hidden=64 moe_gate_hidden=32 \
    experiment=ELU_k2_seed1337

# Scenario 1 (fixed base); the object is selected by objectRoot in the task yaml
python train.py task=DiabloGraspCustom3 headless=True \
    seed=42 moe_num_actors=2 experiment=s1_moe_k2_seed42
```

Seeds used in the thesis (5 per configuration):

| Configuration | Seeds |
|---|---|
| PPO | 2021, 1337, 789, 123, 42 |
| ELU MoE k=2 (proposed) | 1337, 777, 789, 2024, 3407 |
| ELU MoE k=3 | 1337, 42, 999, 2021, 3407 |

Batch drivers are in `scripts/`. They expect two environment variables:

```bash
export IGE_ROOT=/path/to/IsaacGymEnvs
export IGE_DIR=$IGE_ROOT/isaacgymenvs
bash scripts/chain_moeELU_train.sh 0
```

### Reward ablations

```bash
python train.py ... task.env.ablation_no_phase_gate=True    # staged unlocking off
python train.py ... task.env.ablation_no_alive_bonus=True   # balance incentive off
```

### Cross-family baselines

```bash
python train.py task=DiabloBalanceGraspSAC train=DiabloBalanceGraspSACPPO headless=True seed=2021
python train.py task=DiabloBalanceGrasp    train=DiabloBalanceGraspTD3PPO headless=True seed=2021
bash scripts/chain_fasttd3_train.sh 0 8 150000 1337 2021 789 123 42   # needs github.com/younggyoseo/FastTD3
```

**Hardware.** RTX 5070 Ti (16 GB), 2,048 parallel environments, ~7 hours for
6,000 epochs (≈1,179.6 M environment steps).

---

## Task design (Scenario 2)

### Four-phase state machine

Each row gives the condition for *leaving* that phase.

| Phase | Name (code constant) | Leaves the phase when |
|---|---|---|
| 0 | Approach (`PHASE_APPROACH`) | robot within `graspApproachDist` (0.35 m) of the pre-grasp point, and balanced |
| 1 | Grasp (`PHASE_GRASP`) | object lifted more than 0.02 m |
| 2 | Transport (`PHASE_PLACE`) | object within 0.02 m of the platform centre, horizontally and vertically |
| 3 | Release (`PHASE_RELEASE`) | terminal — gripper opens, arm retreats |

Note the constant names: `PHASE_PLACE = 2` is the *transport and alignment*
phase; the actual placement and release happen in `PHASE_RELEASE = 3`.

### Phase-gated rewards

`grasp` is active in Phase 1, `transport` in Phases 2–3, and `placement` only in
Phase 3. Everything downstream of the current phase is held at exactly zero,
which is what makes the reward act as an implicit curriculum. Removing this
gating collapses the success rate to zero.

### Observation (72-D) and action (9-D)

```
obs: progress(1) base_lin_vel(3) base_ang_vel(3) base_quat(4) height_norm(1)
     leg_pos(4) leg_vel(4) wheel_vel(2) arm_pos(4) arm_vel(4)
     eef_pos(3) eef_rot(4) object_pos(3) object_rot(4) handle_pos(3)
     eef→handle(3) robot→handle(3) phase(1) prev_actions(9)
     platform_pos(3) bottom→platform(3) object_one_hot(3)

act: [0] chassis height   [1] chassis pitch   [2,3] wheel velocities
     [4-7] right arm joint increments        [8] gripper
```

---

## Evaluation

Each checkpoint is evaluated on 3 objects × 3 evaluation seeds (42, 7, 123) with
500 environments and the strict 10-consecutive-step criterion.

```bash
export IGE_ROOT=/path/to/IsaacGymEnvs
bash eval/run_all_eval.sh          # edit the checkpoint paths at the top first
```

A single manual run:

```bash
python train.py task=DiabloBalanceGrasp train=DiabloBalanceGraspPPO \
    headless=True test=True num_envs=500 seed=42 \
    checkpoint=runs/<run>/nn/<run>.pth \
    moe_num_actors=2 moe_expert_hidden=64 moe_gate_hidden=32 \
    task.env.eval_mode=True \
    task.env.eval_object_name=mug \
    task.env.success_hold_steps=10 \
    task.env.latch_hold_steps=10
```

**`task.env.eval_mode=True` is required** — without it the PSR, minL2 and
failure-mode counters are never populated and the summary comes out blank.
**`*_hold_steps=10` is also required** to match the thesis; the default of 1
judges instantaneously, counting collision bounces and momentary grazes as
achievements, and reports rates well above the true ones.

### Metrics, as implemented

| Metric | Condition (must hold for 10 consecutive steps) |
|---|---|
| **PSR** — partial success | phase ≥ 2, object within 5 cm of the platform centre laterally and between −2 cm and +1 cm of its surface vertically. Latched: achieving it once anywhere in the episode counts. |
| **SR** — full success | phase 3, object on the platform and upright, chassis balanced, gripper open, **and the end-effector retreated ≥ 12 cm backwards** along the robot's own −X axis. |
| **minL2** | smallest object-to-platform-centre distance reached at any point (no hold requirement). |

Scenario 2 reports PSR and minL2 rather than SR because the release itself is
the fragile moment: as the fingers open, residual chassis oscillation can catch
the object and tip it, so a run that placed the object accurately can still fail
the retreat requirement. PSR keeps that distinction visible.

Failures are split into *timeout*, *fall*, and *drop*, which sum to `1 − PSR`.

---

## Results

Scenario 2, macro-averaged PSR over the three objects, n = 5 training seeds:

| Method | Converged seeds | Macro PSR |
|---|---|---|
| **ELU MoE-PPO (k=2)** | 5/5 | **93.08 %** |
| ELU MoE-PPO (k=3) | 5/5 | 92.86 % |
| PPO | 5/5 | 88.20 % |
| MoE, simple heads (k=3) | 3/5 | 91.40 %¹ |
| MoE, simple heads (k=2) | 3/5 | 83.49 %¹ |
| FastTD3 | 1/5 | 30.55 % |
| SAC, TD3 | 0/3 | never learns to grasp |

¹ averaged over converged seeds only; scoring the failed seeds as 0 % gives
50.1 % for k=2.

Reward ablation, single seed: the full design reaches 0.968 macro PSR; without
the alive bonus 0.806; without the phase-gate 0.000, with 97.6 % of episodes
timing out because the robot never approaches the object.

Per-seed numbers and the raw evaluation logs are available on request
(conon068@gmail.com).

---
