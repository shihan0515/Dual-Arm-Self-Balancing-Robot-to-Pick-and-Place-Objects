#!/bin/bash
# FastTD3 baseline 訓練鏈：依序訓練多個 seeds（單 GPU）。
# 用法: ./chain_fasttd3_train.sh [GPU] [NUM_UPDATES] [TOTAL_TIMESTEPS] [SEEDS...]
#   GPU             預設 0（RTX 5070 Ti 16GB；GPU1 8GB 需另加 --buffer-size 4096）
#   NUM_UPDATES     預設 8（UTD；8≈24h/seed、4≈12h、2≈6.6h @GPU0）
#   TOTAL_TIMESTEPS 預設 150000（× num_envs 1024 ≈ 154M env steps）
#   SEEDS           預設 1337 2021 789 123 42（與 PPO baseline 相同）
#
# 產出：
#   checkpoint → ${FASTTD3_DIR}/models/IGE-DiabloBalanceGrasp__FastTD3__<seed>_*.pt
#   TensorBoard → ${FASTTD3_DIR}/runs_fasttd3/
#   log → 本目錄 fasttd3_seed<seed>.log
#
# 注意：GPU0 (sm_120) 上 torch.compile 會因 Triton PTX 錯誤而失敗，
#       DiabloBalanceGraspArgs 已預設 compile=False。

GPU=${1:-0}
NUM_UPDATES=${2:-8}
TOTAL=${3:-150000}
shift 3 2>/dev/null
SEEDS=("${@:-}")
if [ ${#SEEDS[@]} -eq 0 ] || [ -z "${SEEDS[0]}" ]; then
    SEEDS=(1337 2021 789 123 42)
fi

FASTTD3_DIR=${FASTTD3_DIR}
IGE_DIR=${IGE_DIR}
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH
cd "$FASTTD3_DIR" || exit 1

for SEED in "${SEEDS[@]}"; do
    LOG=$IGE_DIR/fasttd3_seed${SEED}.log
    echo "$(date '+%F %T') 開始 FastTD3 seed=$SEED (GPU$GPU, UTD=$NUM_UPDATES, steps=$TOTAL)"
    # EXTRA_ARGS 環境變數可附加額外參數，例如 EXTRA_ARGS="--buffer-size 4096"
    CUDA_VISIBLE_DEVICES=$GPU python -u train.py \
        --env-name IGE-DiabloBalanceGrasp \
        --exp-name FastTD3 \
        --seed "$SEED" \
        --num-updates "$NUM_UPDATES" \
        --total-timesteps "$TOTAL" \
        --no-use-wandb \
        ${EXTRA_ARGS} \
        > "$LOG" 2>&1
    echo "$(date '+%F %T') 結束 seed=$SEED (exit=$?)"
done
echo "$(date '+%F %T') FastTD3 訓練鏈全部完成"
