#!/bin/bash
# Evaluate DiabloBalanceGraspClaude (MoE-PPO) across seeds and objects

export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

TASK=DiabloBalanceGrasp
TRAIN=DiabloBalanceGraspPPO
# 用法：bash shell_eval_claude.sh [checkpoint路徑] [輸出標籤] [expert數] [變體 simple|elu]
# 變體須與 checkpoint 訓練時一致：moe_0.2 / moe_4242 是 elu；moe_2021 / moe_t / moe_t3 是 simple
CKPT="${1:-runs/DiabloBalanceGrasp_moe_0.2_10-01-53-46/nn/DiabloBalanceGrasp_moe_0.2.pth}"
TAG="${2:-claude}"
MOE_ACTORS="${3:-2}"
VARIANT="${4:-simple}"
HOLD=10   # success/latch 條件須連續成立的步數（嚴格判定）

case "$VARIANT" in
    elu)    EXPERT_HIDDEN=64; GATE_HIDDEN=32 ;;
    simple) EXPERT_HIDDEN=0;  GATE_HIDDEN=0  ;;
    *) echo "未知變體: $VARIANT（用 simple 或 elu）"; exit 1 ;;
esac
NUM_ENVS=500
GPU_ID=0
SEEDS=(${EVAL_SEEDS:-42 7 123})   # 可用環境變數 EVAL_SEEDS 覆寫評估 seed
OBJECTS=("mug" "drill" "dumbbell")

# mean±std over the given values (sample std, n-1)
mean_std() {
    printf '%s\n' "$@" | awk '
        NF { s += $1; ss += $1*$1; n++ }
        END {
            if (n == 0) { printf "n/a"; exit }
            m = s/n; v = (n > 1) ? (ss - n*m*m)/(n-1) : 0; if (v < 0) v = 0
            printf "%.4f±%.4f", m, sqrt(v)
        }'
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="eval_${TAG}_hold${HOLD}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"

echo "DiabloBalanceGraspClaude (MoE-PPO) Evaluation" > "$SUMMARY"
echo "Checkpoint: $CKPT" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"

for obj in "${OBJECTS[@]}"; do
    echo ""
    echo "[ Object: $obj ]"
    echo "" >> "$SUMMARY"
    echo "[ Object: $obj ]" >> "$SUMMARY"

    sr_list=(); psr_list=(); p2r_list=(); tcr_list=(); pl2_list=(); ml2_list=()

    for seed in "${SEEDS[@]}"; do
        log_file="$LOG_DIR/${obj}_seed${seed}.log"
        echo "  seed=$seed ..."

        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
            task=$TASK \
            train=$TRAIN \
            headless=True \
            num_envs=$NUM_ENVS \
            seed="$seed" \
            test=True \
            task.env.eval_mode=True \
            task.env.success_hold_steps=$HOLD \
            task.env.latch_hold_steps=$HOLD \
            moe_num_actors=$MOE_ACTORS \
            moe_expert_hidden=$EXPERT_HIDDEN \
            moe_gate_hidden=$GATE_HIDDEN \
            checkpoint="$CKPT" \
            task.env.eval_object_name="$obj" \
            2>&1 | tee "$log_file"

        sr=$(grep  "final success_rate:"         "$log_file" | tail -1 | grep -oP 'final success_rate: \K[0-9.]+')
        psr=$(grep "final partial_success_rate:"  "$log_file" | tail -1 | grep -oP 'final partial_success_rate: \K[0-9.]+')
        p2r=$(grep "final phase2_rate:"           "$log_file" | tail -1 | grep -oP 'final phase2_rate: \K[0-9.]+')
        tcr=$(grep "final tight_contact_rate:"    "$log_file" | tail -1 | grep -oP 'final tight_contact_rate: \K[0-9.]+')
        pl2=$(grep "final place_L2"               "$log_file" | tail -1 | grep -oP 'final place_L2\(mm\): \K[0-9.]+')
        ml2=$(grep "final min_L2"                 "$log_file" | tail -1 | grep -oP 'final min_L2\(mm\): \K[0-9.]+')

        if [ -z "$sr" ]; then
            echo "    WARNING: seed=${seed} 抓不到 final success_rate，檢查 $log_file" | tee -a "$SUMMARY"
        fi
        [ -n "$sr"  ] && sr_list+=("$sr")
        [ -n "$psr" ] && psr_list+=("$psr")
        [ -n "$p2r" ] && p2r_list+=("$p2r")
        [ -n "$tcr" ] && tcr_list+=("$tcr")
        [ -n "$pl2" ] && pl2_list+=("$pl2")
        [ -n "$ml2" ] && ml2_list+=("$ml2")

        echo "    seed=${seed}  sr=${sr}  psr=${psr}  p2r=${p2r}  tcr=${tcr}  placeL2=${pl2}mm  minL2=${ml2}mm" >> "$SUMMARY"
        echo "    => sr=${sr}  psr=${psr}  p2r=${p2r}  tcr=${tcr}  placeL2=${pl2}mm  minL2=${ml2}mm"
    done

    avg_line="    mean±std (n=${#sr_list[@]}): sr=$(mean_std "${sr_list[@]}")  psr=$(mean_std "${psr_list[@]}")  p2r=$(mean_std "${p2r_list[@]}")  tcr=$(mean_std "${tcr_list[@]}")  placeL2=$(mean_std "${pl2_list[@]}")mm  minL2=$(mean_std "${ml2_list[@]}")mm"
    echo "$avg_line" >> "$SUMMARY"
    echo "$avg_line"
done

echo ""
echo "=========================================="
echo "Done! Summary: $SUMMARY"
echo "=========================================="
cat "$SUMMARY"
