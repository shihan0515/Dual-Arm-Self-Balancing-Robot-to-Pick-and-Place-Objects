#!/bin/bash
# chain_train_extra_seeds.sh
# 動態 job queue：兩張 GPU 共享工作列表，哪張先空就繼續搶下一個 job
# 目前已有: seed=2021, 1337 → 新增: 42,123,456,789 → 每個 config 共 6 seeds
#
# Usage: bash chain_train_extra_seeds.sh [GPU_A] [GPU_B]
#   預設 GPU_A=0, GPU_B=1

set -e
cd "${IGE_DIR:?set IGE_DIR to your IsaacGymEnvs/isaacgymenvs checkout}"
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH

GPU_A=${1:-0}
GPU_B=${2:-1}
NEW_SEEDS=(42 123 456 789)

WORKDIR="logs_extra_seeds"
QUEUE_FILE="${WORKDIR}/job_queue.txt"
LOCK_FILE="${WORKDIR}/job_queue.lock"
DONE_FILE="${WORKDIR}/job_done.txt"

mkdir -p "${WORKDIR}"

# ─────────────────────────────────────────────────────────────────
# 建立 job queue（每行：EXP_NAME:K:EH:GH:SEED）
# 順序：重要的 baseline 先排（PPO、ReLU_k3），ELU 排後面
# ─────────────────────────────────────────────────────────────────
build_queue() {
    # PPO (k=1, linear head)
    for S in "${NEW_SEEDS[@]}"; do echo "PPO_seed${S}:1:0:0:${S}"; done
    # ReLU/Simple MoE k=3 (thesis main result)
    for S in "${NEW_SEEDS[@]}"; do echo "ReLU_k3_seed${S}:3:0:0:${S}"; done
    # ReLU/Simple MoE k=2
    for S in "${NEW_SEEDS[@]}"; do echo "ReLU_k2_seed${S}:2:0:0:${S}"; done
    # ELU MoE k=3
    for S in "${NEW_SEEDS[@]}"; do echo "ELU_k3_seed${S}:3:64:32:${S}"; done
    # ELU MoE k=2
    for S in "${NEW_SEEDS[@]}"; do echo "ELU_k2_seed${S}:2:64:32:${S}"; done
}

TOTAL_JOBS=20

build_queue > "${QUEUE_FILE}"
> "${DONE_FILE}"
> "${LOCK_FILE}"

# ─────────────────────────────────────────────────────────────────
# grab_next_job：使用 flock 原子性地從 queue 取出下一個 job
# stdout 輸出該行，queue 為空時輸出空字串
# ─────────────────────────────────────────────────────────────────
grab_next_job() {
    (
        flock -x 200
        JOB=$(head -1 "${QUEUE_FILE}" 2>/dev/null)
        if [ -n "${JOB}" ]; then
            tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp"
            mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
            echo "${JOB}"
        fi
    ) 200>"${LOCK_FILE}"
}

# ─────────────────────────────────────────────────────────────────
# worker GPU_ID：不斷搶 job 直到 queue 清空
# ─────────────────────────────────────────────────────────────────
worker() {
    local GPU=$1
    local WORKER_LOG="${WORKDIR}/_gpu${GPU}_worker.log"

    echo "$(date '+%F %T') [GPU${GPU}] worker 啟動" | tee -a "${WORKER_LOG}"

    while true; do
        JOB=$(grab_next_job)
        if [ -z "${JOB}" ]; then
            DONE_COUNT=$(wc -l < "${DONE_FILE}")
            echo "$(date '+%F %T') [GPU${GPU}] queue 已空，完成 ${DONE_COUNT}/${TOTAL_JOBS}，退出" \
                | tee -a "${WORKER_LOG}"
            break
        fi

        IFS=':' read -r EXP K EH GH SEED <<< "${JOB}"
        DONE_COUNT=$(wc -l < "${DONE_FILE}")
        REMAIN=$(( TOTAL_JOBS - DONE_COUNT ))

        echo "" | tee -a "${WORKER_LOG}"
        echo "$(date '+%F %T') [GPU${GPU}] ▶ ${EXP}  (剩餘 ${REMAIN} jobs)" \
            | tee -a "${WORKER_LOG}"
        echo "  k=${K}  expert_hidden=${EH}  gate_hidden=${GH}  seed=${SEED}" \
            | tee -a "${WORKER_LOG}"

        CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
            task=DiabloBalanceGrasp \
            train=DiabloBalanceGraspPPO \
            headless=True \
            seed=${SEED} \
            moe_num_actors=${K} \
            moe_expert_hidden=${EH} \
            moe_gate_hidden=${GH} \
            experiment=${EXP} \
            2>&1 | tee "${WORKDIR}/${EXP}.log"

        EXIT_CODE=${PIPESTATUS[0]}
        echo "${EXP}" >> "${DONE_FILE}"
        DONE_COUNT=$(wc -l < "${DONE_FILE}")

        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "$(date '+%F %T') [GPU${GPU}] ✓ DONE ${EXP}  (${DONE_COUNT}/${TOTAL_JOBS})" \
                | tee -a "${WORKER_LOG}"
        else
            echo "$(date '+%F %T') [GPU${GPU}] ✗ FAILED ${EXP} (exit=${EXIT_CODE})" \
                | tee -a "${WORKER_LOG}"
        fi
    done
}

# ─────────────────────────────────────────────────────────────────
# 啟動
# ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        chain_train_extra_seeds  動態分配模式                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  新增 seeds : ${NEW_SEEDS[*]}                                   ║"
echo "║  總計 jobs  : ${TOTAL_JOBS}  (兩張 GPU 動態搶佔)                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Job 優先順序：                                              ║"
echo "║    1. PPO          × 4  (baseline 最優先)                    ║"
echo "║    2. ReLU_k3      × 4  (論文主結果)                         ║"
echo "║    3. ReLU_k2      × 4                                       ║"
echo "║    4. ELU_k3       × 4                                       ║"
echo "║    5. ELU_k2       × 4                                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  哪張 GPU 先跑完當前 job，就自動搶下一個                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "監看進度："
echo "  watch -n 30 'echo \"完成: \$(wc -l < ${DONE_FILE})/${TOTAL_JOBS}\" && cat ${DONE_FILE}'"
echo "  tail -f ${WORKDIR}/_gpu${GPU_A}_worker.log"
echo "  tail -f ${WORKDIR}/_gpu${GPU_B}_worker.log"
echo ""
echo "TensorBoard："
echo "  tensorboard --logdir runs/"
echo ""

worker ${GPU_A} &
PID_A=$!
echo "GPU ${GPU_A} worker PID: ${PID_A}"

worker ${GPU_B} &
PID_B=$!
echo "GPU ${GPU_B} worker PID: ${PID_B}"

echo ""
echo "兩個 worker 並行搶 job..."

wait ${PID_A}
wait ${PID_B}

echo ""
DONE_COUNT=$(wc -l < "${DONE_FILE}")
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  全部完成！  ${DONE_COUNT}/${TOTAL_JOBS} jobs                             ║"
echo "║  完成清單：                                                  ║"
cat "${DONE_FILE}" | while read L; do echo "║    ✓ ${L}"; done
echo "╚══════════════════════════════════════════════════════════════╝"
