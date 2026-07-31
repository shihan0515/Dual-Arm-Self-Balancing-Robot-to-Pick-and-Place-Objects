#!/bin/bash
# 3 runs: ELU-MoE (moe_expert_hidden=64 moe_gate_hidden=32)
#   k=2 seed=2021 → k=2 seed=1337 → k=3 seed=1337
#   (k=3 seed=2021 已在別張顯卡跑，跳過)
# log → chain_moeELU_train.log
# Usage: bash chain_moeELU_train.sh [GPU_ID]
#   GPU_ID default = 0

set -e
cd "${IGE_DIR:?set IGE_DIR to your IsaacGymEnvs/isaacgymenvs checkout}"
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

GPU=${1:-0}

RUNS=(
    "2 2021 DiabloBalanceGrasp_moeELU_k2_seed2021"
    "2 1337 DiabloBalanceGrasp_moeELU_k2_seed1337"
    "3 1337 DiabloBalanceGrasp_moeELU_k3_seed1337"
)

for entry in "${RUNS[@]}"; do
    read -r K SEED EXP <<< "$entry"
    echo "$(date '+%F %T') ▶ START  k=${K} seed=${SEED}  experiment=${EXP}"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        task=DiabloBalanceGrasp \
        train=DiabloBalanceGraspPPO \
        headless=True \
        seed=${SEED} \
        moe_num_actors=${K} \
        moe_expert_hidden=64 \
        moe_gate_hidden=32 \
        experiment=${EXP}
    echo "$(date '+%F %T') ✓ DONE   k=${K} seed=${SEED}  exit=$?"
    echo "---"
done

echo "$(date '+%F %T') 全部 3 個 run 完成"
