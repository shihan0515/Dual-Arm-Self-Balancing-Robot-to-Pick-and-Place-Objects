#!/bin/bash
# Run PPO + MoE-PPO evaluation, then parse and plot each separately.
# Output:
#   eval_ppo_<ts>/          eval_moe_<ts>/          (logs)
#   eval_ppo_<ts>.xlsx      eval_moe_<ts>.xlsx      (Excel)
#   fig_ppo_<ts>_*.png      fig_moe_<ts>_*.png      (plots)

set -e
export PYTHONPATH=/home/neo/Repositories/IsaacGymEnvs:$PYTHONPATH

# ── Checkpoints ───────────────────────────────────────────────────────────────
PPO_CKPT="runs/DiabloBalanceGrasp_PPO_6000_09-21-50-29/nn/DiabloBalanceGrasp_PPO_6000.pth"
MOE_CKPT="runs/DiabloBalanceGrasp_moe_0.2_10-01-53-46/nn/DiabloBalanceGrasp_moe_0.2.pth"

# ── Shared settings ───────────────────────────────────────────────────────────
TASK=DiabloBalanceGrasp
TRAIN=DiabloBalanceGraspPPO
NUM_ENVS=500
GPU_ID=0
SEEDS=("42" "7" "123")
OBJECTS=("mug" "drill" "dumbbell")

TS=$(date +%Y%m%d_%H%M%S)

# ─────────────────────────────────────────────────────────────────────────────
run_eval() {
    local label=$1   # "PPO" or "MoE-PPO"
    local ckpt=$2
    local actors=$3
    local log_dir=$4

    mkdir -p "$log_dir"
    local summary="$log_dir/summary.txt"
    echo "${label} Evaluation"  > "$summary"
    echo "Checkpoint: $ckpt"   >> "$summary"
    echo "==============================" >> "$summary"

    for obj in "${OBJECTS[@]}"; do
        echo ""
        echo "[ ${label} | Object: ${obj} ]"
        echo "" >> "$summary"
        echo "[ Object: ${obj} ]" >> "$summary"

        for seed in "${SEEDS[@]}"; do
            local log_file="${log_dir}/${obj}_seed${seed}.log"
            echo "  seed=${seed} ..."

            CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
                task=$TASK train=$TRAIN headless=True \
                num_envs=$NUM_ENVS seed="$seed" test=True \
                moe_num_actors=$actors \
                checkpoint="$ckpt" \
                task.env.eval_object_name="$obj" \
                2>&1 | tee "$log_file"

            local sr psr p2r tcr
            sr=$(grep  "final success_rate:"        "$log_file" | tail -1 | grep -oP 'final success_rate: \K[0-9.]+')
            psr=$(grep "final partial_success_rate:" "$log_file" | tail -1 | grep -oP 'final partial_success_rate: \K[0-9.]+')
            p2r=$(grep "final phase2_rate:"          "$log_file" | tail -1 | grep -oP 'final phase2_rate: \K[0-9.]+')
            tcr=$(grep "final tight_contact_rate:"   "$log_file" | tail -1 | grep -oP 'final tight_contact_rate: \K[0-9.]+')

            echo "    seed=${seed}  sr=${sr}  psr=${psr}  p2r=${p2r}  tcr=${tcr}" >> "$summary"
            echo "    => sr=${sr}  psr=${psr}  p2r=${p2r}  tcr=${tcr}"
        done
    done

    echo ""
    echo "==============================="
    echo "Done: ${label}"
    echo "==============================="
    cat "$summary"
}

# ── Run both evaluations ──────────────────────────────────────────────────────
PPO_DIR="eval_ppo_${TS}"
MOE_DIR="eval_moe_${TS}"

echo "===== PPO evaluation ====="
run_eval "PPO"     "$PPO_CKPT" 1 "$PPO_DIR"

echo ""
echo "===== MoE-PPO evaluation ====="
run_eval "MoE-PPO" "$MOE_CKPT" 2 "$MOE_DIR"

# ── Parse logs → Excel ────────────────────────────────────────────────────────
echo ""
echo "Parsing logs to Excel..."
python parse_eval_to_excel.py "$PPO_DIR" "eval_ppo_${TS}.xlsx"
python parse_eval_to_excel.py "$MOE_DIR" "eval_moe_${TS}.xlsx"

# ── Plot ──────────────────────────────────────────────────────────────────────
echo "Plotting..."
python plot_eval_results.py "eval_ppo_${TS}.xlsx" "fig_ppo_${TS}" "PPO"
python plot_eval_results.py "eval_moe_${TS}.xlsx" "fig_moe_${TS}" "MoE-PPO"

echo ""
echo "============================================"
echo "All done!"
echo "  PPO Excel  : eval_ppo_${TS}.xlsx"
echo "  MoE Excel  : eval_moe_${TS}.xlsx"
echo "  PPO plots  : fig_ppo_${TS}_by_metric.png / _by_object.png"
echo "  MoE plots  : fig_moe_${TS}_by_metric.png / _by_object.png"
echo "============================================"
