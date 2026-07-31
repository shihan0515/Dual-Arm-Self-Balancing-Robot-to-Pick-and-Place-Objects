#!/bin/bash
# Eval reward ablation checkpoints — ELU MoE-PPO k=2 seed=1337
# NoPG on GPU 0, NoAB on GPU 1 (parallel)

set -e
cd "${IGE_DIR:?set IGE_DIR to your IsaacGymEnvs/isaacgymenvs checkout}"
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_SEEDS=(42 7 123)
OBJECTS=(mug drill dumbbell)
K=2; EH=64; GH=32

CKPT_NOPG="runs/ELU_k2_AblNoPG_seed1337_03-16-12-58/nn/ELU_k2_AblNoPG_seed1337.pth"
CKPT_NOAB="runs/ELU_k2_AblNoAB_seed1337_03-16-12-58/nn/ELU_k2_AblNoAB_seed1337.pth"

run_eval() {
    local GPU=$1 LABEL=$2 CKPT=$3 OBJ=$4 ESEED=$5
    local OUT_DIR="eval_abl_${LABEL}_obj${OBJ}_eseed${ESEED}_${TIMESTAMP}"
    mkdir -p "$OUT_DIR"
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
        moe_expert_hidden=${EH} \
        moe_gate_hidden=${GH} \
        2>&1 | tee "${OUT_DIR}/eval.log"
    echo "$(date '+%F %T') ✓ ${LABEL} obj=${OBJ} eseed=${ESEED}"
}

# GPU 0: NoPG
(
  for OBJ in "${OBJECTS[@]}"; do
    for ESEED in "${EVAL_SEEDS[@]}"; do
      run_eval 0 ELU_k2_NoPG "$CKPT_NOPG" "$OBJ" "$ESEED"
    done
  done
  echo "$(date '+%F %T') === NoPG 全部 eval 完成 ==="
) 2>&1 | tee eval_ablation_nopg.log &
PID_NOPG=$!

# GPU 1: NoAB
(
  for OBJ in "${OBJECTS[@]}"; do
    for ESEED in "${EVAL_SEEDS[@]}"; do
      run_eval 1 ELU_k2_NoAB "$CKPT_NOAB" "$OBJ" "$ESEED"
    done
  done
  echo "$(date '+%F %T') === NoAB 全部 eval 完成 ==="
) 2>&1 | tee eval_ablation_noab.log &
PID_NOAB=$!

echo "NoPG eval PID=${PID_NOPG} (GPU 0)"
echo "NoAB eval PID=${PID_NOAB} (GPU 1)"
echo "監看: tail -f eval_ablation_nopg.log  /  tail -f eval_ablation_noab.log"

wait $PID_NOPG && echo "$(date '+%F %T') ✓ NoPG eval 完成"
wait $PID_NOAB && echo "$(date '+%F %T') ✓ NoAB eval 完成"
echo "$(date '+%F %T') 全部完成"
