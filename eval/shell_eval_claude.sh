#!/bin/bash
# Evaluate DiabloBalanceGraspClaude (MoE-PPO) across seeds and objects

export PYTHONPATH=/home/neo/Repositories/IsaacGymEnvs:$PYTHONPATH

TASK=DiabloBalanceGrasp
TRAIN=DiabloBalanceGraspPPO
CKPT="runs/DiabloBalanceGrasp_moe_0.2_10-01-53-46/nn/DiabloBalanceGrasp_moe_0.2.pth"
MOE_ACTORS=2
NUM_ENVS=500
GPU_ID=0
SEEDS=("42" "7" "123")
OBJECTS=("mug" "drill" "dumbbell")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="eval_claude_${TIMESTAMP}"
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
            moe_num_actors=$MOE_ACTORS \
            checkpoint="$CKPT" \
            task.env.eval_object_name="$obj" \
            2>&1 | tee "$log_file"

        sr=$(grep  "final success_rate:"         "$log_file" | tail -1 | grep -oP 'final success_rate: \K[0-9.]+')
        psr=$(grep "final partial_success_rate:"  "$log_file" | tail -1 | grep -oP 'final partial_success_rate: \K[0-9.]+')
        p2r=$(grep "final phase2_rate:"           "$log_file" | tail -1 | grep -oP 'final phase2_rate: \K[0-9.]+')
        tcr=$(grep "final tight_contact_rate:"    "$log_file" | tail -1 | grep -oP 'final tight_contact_rate: \K[0-9.]+')

        echo "    seed=${seed}  sr=${sr}  psr=${psr}  p2r=${p2r}  tcr=${tcr}" >> "$SUMMARY"
        echo "    => sr=${sr}  psr=${psr}  p2r=${p2r}  tcr=${tcr}"
    done
done

echo ""
echo "=========================================="
echo "Done! Summary: $SUMMARY"
echo "=========================================="
cat "$SUMMARY"
