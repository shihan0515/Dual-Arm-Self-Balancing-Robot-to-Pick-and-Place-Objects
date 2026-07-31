#!/bin/bash
# 評估四個 ELU-MoE checkpoint（k=2/3 × seed=2021/1337）
# Usage: bash eval_elu_moe.sh [GPU_ID]
set -e
cd "${IGE_DIR:?set IGE_DIR to your IsaacGymEnvs/isaacgymenvs checkout}"
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

GPU=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

declare -A CKPTS
CKPTS["ELU_k2_seed2021"]="runs/DiabloBalanceGrasp_moeELU_k2_seed2021_21-13-17-49/nn/DiabloBalanceGrasp_moeELU_k2_seed2021.pth"
CKPTS["ELU_k2_seed1337"]="runs/DiabloBalanceGrasp_moeELU_k2_seed1337_21-20-54-01/nn/DiabloBalanceGrasp_moeELU_k2_seed1337.pth"
CKPTS["ELU_k3_seed2021"]="runs/DiabloBalanceGrasp_moeELU_k3_seed2021_22-04-22-21/nn/DiabloBalanceGrasp_moeELU_k3_seed2021.pth"
CKPTS["ELU_k3_seed1337"]="runs/DiabloBalanceGrasp_moeELU_k3_seed1337_22-12-03-15/nn/DiabloBalanceGrasp_moeELU_k3_seed1337.pth"

declare -A ACTORS
ACTORS["ELU_k2_seed2021"]=2
ACTORS["ELU_k2_seed1337"]=2
ACTORS["ELU_k3_seed2021"]=3
ACTORS["ELU_k3_seed1337"]=3

for MODEL in ELU_k2_seed2021 ELU_k2_seed1337 ELU_k3_seed2021 ELU_k3_seed1337; do
    CKPT="${CKPTS[$MODEL]}"
    K="${ACTORS[$MODEL]}"
    LOG_DIR="eval_${MODEL}_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"
    SUMMARY="$LOG_DIR/summary.txt"

    echo "$(date '+%F %T') ▶ $MODEL  (k=$K)"
    echo "DiabloBalanceGraspClaude (ELU-MoE) Evaluation" > "$SUMMARY"
    echo "Checkpoint: $CKPT" >> "$SUMMARY"
    echo "========================================" >> "$SUMMARY"

    for OBJ in mug drill dumbbell; do
        echo "" >> "$SUMMARY"
        echo "[ Object: $OBJ ]" >> "$SUMMARY"

        SRS=""; PSRS=""; P2RS=""; TCRS=""; PL2S=""
        for SEED in 42 7 123; do
            LOG_FILE="$LOG_DIR/${OBJ}_seed${SEED}.log"
            CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
                task=DiabloBalanceGrasp \
                train=DiabloBalanceGraspPPO \
                headless=True \
                num_envs=500 \
                seed=${SEED} \
                test=True \
                moe_num_actors=${K} \
                moe_expert_hidden=64 \
                moe_gate_hidden=32 \
                checkpoint="${CKPT}" \
                task.env.eval_mode=True \
                task.env.eval_object_name="${OBJ}" \
                task.env.success_hold_steps=10 \
                task.env.latch_hold_steps=10 \
                2>&1 | tee "$LOG_FILE"

            SR=$(grep  "final success_rate:"         "$LOG_FILE" | tail -1 | grep -oP 'final success_rate: \K[0-9.]+')
            PSR=$(grep "final partial_success_rate:" "$LOG_FILE" | tail -1 | grep -oP 'final partial_success_rate: \K[0-9.]+')
            P2R=$(grep "final phase2_rate:"          "$LOG_FILE" | tail -1 | grep -oP 'final phase2_rate: \K[0-9.]+')
            TCR=$(grep "final tight_contact_rate:"   "$LOG_FILE" | tail -1 | grep -oP 'final tight_contact_rate: \K[0-9.]+')
            PL2=$(grep "final place_L2"              "$LOG_FILE" | tail -1 | grep -oP 'place_L2\(mm\): \K[0-9.]+')
            ML2=$(grep "final place_L2"              "$LOG_FILE" | tail -1 | grep -oP 'min_L2\(mm\): \K[0-9.]+')
            echo "    seed=${SEED}  sr=${SR}  psr=${PSR}  p2r=${P2R}  tcr=${TCR}  placeL2=${PL2}mm  minL2=${ML2}mm" >> "$SUMMARY"
        done
    done
    echo "$(date '+%F %T') ✓ DONE $MODEL"
    echo ""
    cat "$SUMMARY"
    echo "---"
done

echo "全部 ELU-MoE 評估完成"
