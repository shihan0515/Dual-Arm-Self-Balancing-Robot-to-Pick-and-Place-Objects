#!/bin/bash
# FastTD3 checkpoint 評估：3 物體 × 3 eval seeds（42, 7, 123），
# 與論文 §4.5 的 PSR / minL2 / psr_* 失敗模式協定完全一致。
# 用法: ./eval_fasttd3.sh <checkpoint.pt> <label> [GPU]
#   例: ./eval_fasttd3.sh \
#         ${FASTTD3_DIR}/models/IGE-DiabloBalanceGrasp__FastTD3__1337_final.pt \
#         seed1337 0
# 每個 (物體, eval seed) 輸出到 eval_FastTD3_<label>_obj*_eseed*_<stamp>/eval.log，
# 彙整寫入 eval_FastTD3_<label>_<stamp>_summary.txt（各指標取 log 最後一行）。

CKPT=$1
LABEL=$2
GPU=${3:-0}
if [ -z "$CKPT" ] || [ -z "$LABEL" ]; then
    echo "用法: $0 <checkpoint.pt> <label> [GPU]"; exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
FASTTD3_DIR=${FASTTD3_DIR}
IGE_DIR=${IGE_DIR}
SUMMARY=$IGE_DIR/eval_FastTD3_${LABEL}_${STAMP}_summary.txt
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH
cd "$FASTTD3_DIR" || exit 1

echo "FastTD3 eval  ckpt=$CKPT  label=$LABEL  stamp=$STAMP" | tee "$SUMMARY"

for OBJ in mug drill dumbbell; do
    for ESEED in 42 7 123; do
        OUT=$IGE_DIR/eval_FastTD3_${LABEL}_obj${OBJ}_eseed${ESEED}_${STAMP}
        mkdir -p "$OUT"
        echo "$(date '+%F %T') eval object=$OBJ eval_seed=$ESEED ..."
        CUDA_VISIBLE_DEVICES=$GPU python -u eval_isaacgymenvs.py \
            --checkpoint "$CKPT" \
            --object "$OBJ" --seed "$ESEED" \
            --num-envs 500 --eval-steps 4000 \
            > "$OUT/eval.log" 2>&1
        {
            echo "--- object=$OBJ eval_seed=$ESEED"
            grep "partial_success_rate" "$OUT/eval.log" | tail -1
            grep "place_L2"             "$OUT/eval.log" | tail -1
            grep "psr_fall_rate"        "$OUT/eval.log" | tail -1
        } | tee -a "$SUMMARY"
    done
done
echo "$(date '+%F %T') 完成，彙整見 $SUMMARY"
