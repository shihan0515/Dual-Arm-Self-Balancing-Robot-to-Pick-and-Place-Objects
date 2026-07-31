#!/bin/bash
# Reward ablation — ELU MoE-PPO k=2 seed=1337
#   GPU_A (預設 0): no_phase_gate
#   GPU_B (預設 1): no_alive_bonus
# Usage: bash chain_ablation_reward_elu_k2.sh [GPU_A] [GPU_B]

set -e
cd "${IGE_DIR:?set IGE_DIR to your IsaacGymEnvs/isaacgymenvs checkout}"
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

GPU_A=${1:-0}
GPU_B=${2:-1}
SEED=1337
K=2
EH=64
GH=32

echo "$(date '+%F %T') ▶ START  ELU_k2_AblNoPG_seed${SEED}   (GPU=${GPU_A})"
CUDA_VISIBLE_DEVICES=${GPU_A} python -u train.py \
    task=DiabloBalanceGrasp \
    train=DiabloBalanceGraspPPO \
    headless=True \
    seed=${SEED} \
    moe_num_actors=${K} \
    moe_expert_hidden=${EH} \
    moe_gate_hidden=${GH} \
    task.env.ablation_no_phase_gate=True \
    experiment=ELU_k2_AblNoPG_seed${SEED} \
    2>&1 | tee ablation_elu_k2_no_phase_gate.log &
PID_A=$!
echo "  PID_A=${PID_A}"

echo "$(date '+%F %T') ▶ START  ELU_k2_AblNoAB_seed${SEED}   (GPU=${GPU_B})"
CUDA_VISIBLE_DEVICES=${GPU_B} python -u train.py \
    task=DiabloBalanceGrasp \
    train=DiabloBalanceGraspPPO \
    headless=True \
    seed=${SEED} \
    moe_num_actors=${K} \
    moe_expert_hidden=${EH} \
    moe_gate_hidden=${GH} \
    task.env.ablation_no_alive_bonus=True \
    experiment=ELU_k2_AblNoAB_seed${SEED} \
    2>&1 | tee ablation_elu_k2_no_alive_bonus.log &
PID_B=$!
echo "  PID_B=${PID_B}"

echo ""
echo "兩個 run 並行執行，監看進度："
echo "  tail -f ablation_elu_k2_no_phase_gate.log"
echo "  tail -f ablation_elu_k2_no_alive_bonus.log"
echo ""

wait ${PID_A}
echo "$(date '+%F %T') ✓ ELU_k2_AblNoPG_seed${SEED} 完成 (exit=$?)"
wait ${PID_B}
echo "$(date '+%F %T') ✓ ELU_k2_AblNoAB_seed${SEED} 完成 (exit=$?)"

echo ""
echo "============================================================"
echo "訓練完成，接著跑 eval："
echo "  bash shell_evaluate_generalization.sh \\"
echo "    runs/ELU_k2_AblNoPG_seed${SEED}_*/nn/ELU_k2_AblNoPG_seed${SEED}.pth \\"
echo "    2 ${EH} ${GH}"
echo ""
echo "  bash shell_evaluate_generalization.sh \\"
echo "    runs/ELU_k2_AblNoAB_seed${SEED}_*/nn/ELU_k2_AblNoAB_seed${SEED}.pth \\"
echo "    2 ${EH} ${GH}"
echo "============================================================"
