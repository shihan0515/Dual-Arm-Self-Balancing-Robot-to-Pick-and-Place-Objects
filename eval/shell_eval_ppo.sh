#!/bin/bash
# Evaluate DiabloBalanceGrasp (PPO) across seeds and objects

export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

TASK=DiabloBalanceGrasp
TRAIN=DiabloBalanceGraspPPO
CKPT="runs/DiabloBalanceGrasp_PPO_6000_09-21-50-29/nn/DiabloBalanceGrasp_PPO_6000.pth"
MOE_ACTORS=1
NUM_ENVS=500
HOLD=10   # success/latch 條件須連續成立的步數（嚴格判定）
GPU_ID=0
SEEDS=("42" "7" "123")
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
LOG_DIR="eval_ppo_hold${HOLD}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"

echo "DiabloBalanceGrasp (PPO) Evaluation" > "$SUMMARY"
echo "Checkpoint: $CKPT" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"

for obj in "${OBJECTS[@]}"; do
    echo ""
    echo "[ Object: $obj ]"
    echo "" >> "$SUMMARY"
    echo "[ Object: $obj ]" >> "$SUMMARY"

    sr_list=(); psr_list=(); p2r_list=(); tcr_list=()

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
            checkpoint="$CKPT" \
            task.env.eval_object_name="$obj" \
            2>&1 | tee "$log_file"

        sr=$(grep  "final success_rate:"         "$log_file" | tail -1 | grep -oP 'final success_rate: \K[0-9.]+')
        psr=$(grep "final partial_success_rate:"  "$log_file" | tail -1 | grep -oP 'final partial_success_rate: \K[0-9.]+')
        p2r=$(grep "final phase2_rate:"           "$log_file" | tail -1 | grep -oP 'final phase2_rate: \K[0-9.]+')
        tcr=$(grep "final tight_contact_rate:"    "$log_file" | tail -1 | grep -oP 'final tight_contact_rate: \K[0-9.]+')

        if [ -z "$sr" ]; then
            echo "    WARNING: seed=${seed} 抓不到 final success_rate，檢查 $log_file" | tee -a "$SUMMARY"
        fi
        [ -n "$sr"  ] && sr_list+=("$sr")
        [ -n "$psr" ] && psr_list+=("$psr")
        [ -n "$p2r" ] && p2r_list+=("$p2r")
        [ -n "$tcr" ] && tcr_list+=("$tcr")

        echo "    seed=${seed}  sr=${sr}  psr=${psr}  p2r=${p2r}  tcr=${tcr}" >> "$SUMMARY"
        echo "    => sr=${sr}  psr=${psr}  p2r=${p2r}  tcr=${tcr}"
    done

    avg_line="    mean±std (n=${#sr_list[@]}): sr=$(mean_std "${sr_list[@]}")  psr=$(mean_std "${psr_list[@]}")  p2r=$(mean_std "${p2r_list[@]}")  tcr=$(mean_std "${tcr_list[@]}")"
    echo "$avg_line" >> "$SUMMARY"
    echo "$avg_line"
done

echo ""
echo "=========================================="
echo "Done! Summary: $SUMMARY"
echo "=========================================="
cat "$SUMMARY"
