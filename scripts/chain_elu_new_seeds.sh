#!/bin/bash
# 4 runs: ELU-MoE 新 seed
#   k=2 seed=2024 → k=2 seed=3407 → k=3 seed=2024 → k=3 seed=3407
# 等待 eval PID 結束後才啟動（避免與 eval 搶 GPU 0）
# Usage: bash chain_elu_new_seeds.sh [EVAL_PID] [GPU_ID]

set -e
cd "${IGE_DIR:?set IGE_DIR to your IsaacGymEnvs/isaacgymenvs checkout}"
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

EVAL_PID=${1:-665819}
GPU=${2:-0}

if kill -0 "$EVAL_PID" 2>/dev/null; then
    echo "$(date '+%F %T') 等待 eval PID ${EVAL_PID} 結束..."
    while kill -0 "$EVAL_PID" 2>/dev/null; do
        sleep 30
    done
    echo "$(date '+%F %T') PID ${EVAL_PID} 已結束，開始訓練"
fi

RUNS=(
    "2 2024 ELU_k2_seed2024"
    "2 3407 ELU_k2_seed3407"
    "3 2024 ELU_k3_seed2024"
    "3 3407 ELU_k3_seed3407"
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

echo "$(date '+%F %T') 全部 4 個 run 完成"
