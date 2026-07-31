#!/bin/bash
# eval_all_v2.sh
# 重跑全部 34 個 checkpoint 的 eval（加入 PSR-based fall/drop/timeout 指標）
# Usage: bash eval_all_v2.sh [GPU_ID]

set -e
cd "${IGE_DIR:?set IGE_DIR to your IsaacGymEnvs/isaacgymenvs checkout}"
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

GPU=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_SEEDS=(42 7 123)
OBJECTS=(mug drill dumbbell)

# Format: "LABEL|K|EH|GH|TRAIN_SEED|CKPT"
# EH=moe_expert_hidden, GH=moe_gate_hidden (0 = ReLU/PPO)
CHECKPOINTS=(
    # ── PPO ──────────────────────────────────────────────────────────────────
    "PPO|1|0|0|2021|runs/DiabloBalanceGrasp_ppp_6000_seed2021_10-23-10-14/nn/DiabloBalanceGrasp_ppp_6000_seed2021.pth"
    "PPO|1|0|0|1337|runs/DiabloBalanceGrasp_ppp_6000_1337_13-13-43-58/nn/DiabloBalanceGrasp_ppp_6000_1337.pth"
    "PPO|1|0|0|42|runs/PPO_seed42_26-15-21-54/nn/PPO_seed42.pth"
    "PPO|1|0|0|789|runs/PPO_seed789_27-01-16-05/nn/PPO_seed789.pth"
    "PPO|1|0|0|456|runs/PPO_seed456_26-22-26-29/nn/PPO_seed456.pth"
    "PPO|1|0|0|123|runs/PPO_seed123_26-15-21-54/nn/PPO_seed123.pth"
    # ── ReLU MoE k=2 ────────────────────────────────────────────────────────
    "ReLU_k2|2|0|0|2021|runs/DiabloBalanceGrasp_moe_2021_10-11-16-37/nn/DiabloBalanceGrasp_moe_2021.pth"
    "ReLU_k2|2|0|0|1337|runs/DiabloBalanceGrasp_13-12-38-41/nn/DiabloBalanceGrasp.pth"
    "ReLU_k2|2|0|0|789|runs/ReLU_k2_seed789_28-11-06-40/nn/ReLU_k2_seed789.pth"
    "ReLU_k2|2|0|0|123|runs/ReLU_k2_seed123_28-03-45-25/nn/ReLU_k2_seed123.pth"
    "ReLU_k2|2|0|0|42|runs/ReLU_k2_seed42_27-20-29-14/nn/ReLU_k2_seed42.pth"
    "ReLU_k2|2|0|0|456|runs/ReLU_k2_seed456_28-05-43-55/nn/ReLU_k2_seed456.pth"
    # ── ReLU MoE k=3 ────────────────────────────────────────────────────────
    "ReLU_k3|3|0|0|2021|runs/DiabloBalanceGrasp_moe_k3_2021_17-14-08-37/nn/DiabloBalanceGrasp_moe_k3_2021.pth"
    "ReLU_k3|3|0|0|1337|runs/DiabloBalanceGrasp_moe_k3_1337_17-14-08-09/nn/DiabloBalanceGrasp_moe_k3_1337.pth"
    "ReLU_k3|3|0|0|42|runs/ReLU_k3_seed42_27-05-45-29/nn/ReLU_k3_seed42.pth"
    "ReLU_k3|3|0|0|456|runs/ReLU_k3_seed456_27-13-07-05/nn/ReLU_k3_seed456.pth"
    "ReLU_k3|3|0|0|789|runs/ReLU_k3_seed789_27-20-23-26/nn/ReLU_k3_seed789.pth"
    "ReLU_k3|3|0|0|123|runs/ReLU_k3_seed123_27-10-51-51/nn/ReLU_k3_seed123.pth"
    # ── ELU MoE k=2 ─────────────────────────────────────────────────────────
    "ELU_k2|2|64|32|1337|runs/DiabloBalanceGrasp_moeELU_k2_seed1337_21-20-54-01/nn/DiabloBalanceGrasp_moeELU_k2_seed1337.pth"
    "ELU_k2|2|64|32|2021|runs/DiabloBalanceGrasp_moeELU_k2_seed2021_21-13-17-49/nn/DiabloBalanceGrasp_moeELU_k2_seed2021.pth"
    "ELU_k2|2|64|32|123|runs/ELU_k2_seed123_29-10-03-22/nn/ELU_k2_seed123.pth"
    "ELU_k2|2|64|32|777|runs/ELU_k2_seed777_30-01-12-26/nn/ELU_k2_seed777.pth"
    "ELU_k2|2|64|32|999|runs/ELU_k2_seed999_30-04-22-39/nn/ELU_k2_seed999.pth"
    "ELU_k2|2|64|32|789|runs/ELU_k2_seed789_29-18-43-45/nn/ELU_k2_seed789.pth"
    "ELU_k2|2|64|32|42|runs/ELU_k2_seed42_29-09-04-59/nn/ELU_k2_seed42.pth"
    "ELU_k2|2|64|32|456|runs/ELU_k2_seed456_29-17-36-49/nn/ELU_k2_seed456.pth"
    # ── ELU MoE k=3 ─────────────────────────────────────────────────────────
    "ELU_k3|3|64|32|2021|runs/DiabloBalanceGrasp_moeELU_k3_seed2021_22-04-22-21/nn/DiabloBalanceGrasp_moeELU_k3_seed2021.pth"
    "ELU_k3|3|64|32|1337|runs/DiabloBalanceGrasp_moeELU_k3_seed1337_22-12-03-15/nn/DiabloBalanceGrasp_moeELU_k3_seed1337.pth"
    "ELU_k3|3|64|32|789|runs/ELU_k3_seed789_29-02-17-35/nn/ELU_k3_seed789.pth"
    "ELU_k3|3|64|32|123|runs/ELU_k3_seed123_28-18-30-28/nn/ELU_k3_seed123.pth"
    "ELU_k3|3|64|32|777|runs/ELU_k3_seed777_30-08-47-23/nn/ELU_k3_seed777.pth"
    "ELU_k3|3|64|32|999|runs/ELU_k3_seed999_30-14-02-51/nn/ELU_k3_seed999.pth"
    "ELU_k3|3|64|32|42|runs/ELU_k3_seed42_28-13-27-58/nn/ELU_k3_seed42.pth"
    "ELU_k3|3|64|32|456|runs/ELU_k3_seed456_28-23-14-52/nn/ELU_k3_seed456.pth"
)

run_eval() {
    local K=$1 EH=$2 GH=$3 CKPT=$4 OBJ=$5 ESEED=$6 LOG_FILE=$7
    local extra_args=""
    if [ "$EH" -gt 0 ]; then
        extra_args="moe_expert_hidden=${EH} moe_gate_hidden=${GH}"
    fi
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        task=DiabloBalanceGrasp \
        train=DiabloBalanceGraspPPO \
        headless=True \
        num_envs=500 \
        seed=${ESEED} \
        test=True \
        moe_num_actors=${K} \
        checkpoint="${CKPT}" \
        task.env.eval_mode=True \
        task.env.eval_object_name="${OBJ}" \
        task.env.success_hold_steps=10 \
        task.env.latch_hold_steps=10 \
        ${extra_args} \
        2>&1 | tee "$LOG_FILE"
}

parse_log() {
    local f=$1
    local psr=$(grep  "final partial_success_rate:" "$f" | tail -1 | grep -oP 'final partial_success_rate: \K[0-9.]+')
    local fall=$(grep "final place_L2"              "$f" | tail -1 | grep -oP 'final fall_rate: \K[0-9.]+')
    local drop=$(grep "final place_L2"              "$f" | tail -1 | grep -oP 'final drop_rate: \K[0-9.]+')
    local tout=$(grep "final place_L2"              "$f" | tail -1 | grep -oP 'final timeout_rate: \K[0-9.]+')
    local ml2=$(grep  "final place_L2"              "$f" | tail -1 | grep -oP 'min_L2\(mm\): \K[0-9.]+')
    local pfal=$(grep "final psr_fall_rate:"        "$f" | tail -1 | grep -oP 'final psr_fall_rate: \K[0-9.]+')
    local pdro=$(grep "final psr_fall_rate:"        "$f" | tail -1 | grep -oP 'final psr_drop_rate: \K[0-9.]+')
    local ptmo=$(grep "final psr_fall_rate:"        "$f" | tail -1 | grep -oP 'final psr_timeout_rate: \K[0-9.]+')
    echo "psr=${psr:--} fall=${fall:--} drop=${drop:--} timeout=${tout:--} minL2=${ml2:--}mm | psr_fall=${pfal:--} psr_drop=${pdro:--} psr_timeout=${ptmo:--}"
}

TOTAL=${#CHECKPOINTS[@]}
IDX=0

for entry in "${CHECKPOINTS[@]}"; do
    IDX=$((IDX + 1))
    IFS='|' read MODEL K EH GH SEED CKPT <<< "$entry"
    LOG_DIR="eval_v2_${MODEL}_seed${SEED}_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"
    SUMMARY="$LOG_DIR/summary.txt"

    echo "$(date '+%F %T') [$IDX/$TOTAL] ${MODEL} seed=${SEED}"
    {
        echo "Model: ${MODEL}  Seed: ${SEED}  k=${K}  expert_hidden=${EH}"
        echo "Checkpoint: ${CKPT}"
        echo "========================================"
    } > "$SUMMARY"

    for OBJ in "${OBJECTS[@]}"; do
        echo "" >> "$SUMMARY"
        echo "[ Object: $OBJ ]" >> "$SUMMARY"
        for ESEED in "${EVAL_SEEDS[@]}"; do
            LOG_FILE="$LOG_DIR/${OBJ}_seed${ESEED}.log"
            run_eval "${K}" "${EH}" "${GH}" "${CKPT}" "${OBJ}" "${ESEED}" "${LOG_FILE}"
            echo "    seed=${ESEED}  $(parse_log "$LOG_FILE")" >> "$SUMMARY"
        done
    done

    echo "$(date '+%F %T') ✓ DONE ${MODEL} seed=${SEED}"
    cat "$SUMMARY"
    echo "---"
done

echo "$(date '+%F %T') 全部 eval v2 完成（共 ${TOTAL} 個 checkpoint）"
