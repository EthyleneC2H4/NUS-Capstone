#!/usr/bin/env bash
# ============================================================================
# Resume experiments from P1_pe_dim16 (after the first 15 runs completed)
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="/root/miniconda3/envs/cancer-gnn/bin/python"
LOG_DIR="results"
SUMMARY="${LOG_DIR}/m5_ablation_summary.csv"
MASTER_LOG="${LOG_DIR}/master_resume.log"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

extract_metrics() {
    local log_file="$1"
    local aupr auroc
    aupr=$(grep -oP 'AUPR: \K[0-9.]+' "$log_file" | tail -1)
    auroc=$(grep -oP 'AUROC: \K[0-9.]+' "$log_file" | tail -1)
    echo "${aupr:-NA},${auroc:-NA}"
}

BASE_FLAGS=(
    --gcn 1
    --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB
    --norm_type none
    --use_residual True
    --use_net_weights True
    --lr_scheduler cosine
    --label_smoothing 0.05
    --hidden 64
    --n_layers 3
    --dropout 0.5
    --epochs 2000
    --patience 250
)

SEEDS=(72 1 2)

echo "============================================================" | tee "$MASTER_LOG"
echo "[$(timestamp)] RESUME: Starting from P1_pe_dim16" | tee -a "$MASTER_LOG"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

# ── Phase 1 (remaining): P1, P3, P4 ─────────────────────────────────────────
REMAINING=(
    "P1_pe_dim16|--pe_dim 16"
    "P3_gps_meta|--gps_meta 1 --gps_heads 4"
    "P4_cross_net_attn|--cross_network_attention 1"
)

for entry in "${REMAINING[@]}"; do
    IFS='|' read -r EXP_NAME EXTRA_FLAGS <<< "$entry"

    for SEED in "${SEEDS[@]}"; do
        RUN_LOG="${LOG_DIR}/m5_${EXP_NAME}_seed${SEED}.log"

        echo "[$(timestamp)] START: ${EXP_NAME}  seed=${SEED}" | tee -a "$MASTER_LOG"

        START=$(date +%s)

        CMD=("$PYTHON" experiments/run_improved.py
             "${BASE_FLAGS[@]}"
             --seed "$SEED")

        if [[ -n "$EXTRA_FLAGS" ]]; then
            read -ra EXTRA_ARRAY <<< "$EXTRA_FLAGS"
            CMD+=("${EXTRA_ARRAY[@]}")
        fi

        echo "  Command: ${CMD[*]}" | tee -a "$MASTER_LOG"

        set +e
        env PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
        EXIT_CODE=$?
        set -e

        END=$(date +%s)
        DURATION=$((END - START))

        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "[$(timestamp)] FAILED: ${EXP_NAME} seed=${SEED} (exit=$EXIT_CODE, ${DURATION}s)" | tee -a "$MASTER_LOG"
            echo "${EXP_NAME},${SEED},FAILED,FAILED,${DURATION}" >> "$SUMMARY"
        else
            METRICS=$(extract_metrics "$RUN_LOG")
            echo "${EXP_NAME},${SEED},${METRICS},${DURATION}" >> "$SUMMARY"
            echo "[$(timestamp)] DONE: ${EXP_NAME} seed=${SEED} -> ${METRICS} (${DURATION}s)" | tee -a "$MASTER_LOG"
        fi

        echo "" | tee -a "$MASTER_LOG"
    done
done

echo "[$(timestamp)] PHASE 1 COMPLETE" | tee -a "$MASTER_LOG"

# ── Phase 2: Extra baseline seeds for M3 ────────────────────────────────────
EXTRA_SEEDS=(42 99)

for SEED in "${EXTRA_SEEDS[@]}"; do
    RUN_LOG="${LOG_DIR}/m3_extraseed_${SEED}.log"

    echo "[$(timestamp)] START: baseline extra seed=${SEED}" | tee -a "$MASTER_LOG"
    START=$(date +%s)

    CMD=("$PYTHON" experiments/run_improved.py
         "${BASE_FLAGS[@]}"
         --seed "$SEED")

    set +e
    env PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
    EXIT_CODE=$?
    set -e

    END=$(date +%s)
    DURATION=$((END - START))

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[$(timestamp)] FAILED: extra seed=${SEED}" | tee -a "$MASTER_LOG"
    else
        METRICS=$(extract_metrics "$RUN_LOG")
        echo "[$(timestamp)] DONE: extra seed=${SEED} -> ${METRICS} (${DURATION}s)" | tee -a "$MASTER_LOG"
    fi
done

echo "[$(timestamp)] PHASE 2 COMPLETE" | tee -a "$MASTER_LOG"

# ── Phase 3: GNNExplainer on best model ─────────────────────────────────────
echo "[$(timestamp)] PHASE 3: GNNExplainer" | tee -a "$MASTER_LOG"

BEST_LINE=$(tail -n +2 "$SUMMARY" | grep -v FAILED | sort -t',' -k3 -nr | head -1)
BEST_EXP=$(echo "$BEST_LINE" | cut -d',' -f1)
BEST_SEED=$(echo "$BEST_LINE" | cut -d',' -f2)
BEST_AUPR=$(echo "$BEST_LINE" | cut -d',' -f3)
echo "Best: ${BEST_EXP} seed=${BEST_SEED} AUPR=${BEST_AUPR}" | tee -a "$MASTER_LOG"

BEST_MODEL_DIR=$(ls -dt results/my_models/GCN_* 2>/dev/null | head -1)
if [[ -z "$BEST_MODEL_DIR" ]]; then
    BEST_MODEL_DIR=$(ls -dt results/my_models/*/ 2>/dev/null | head -1)
fi

if [[ -n "$BEST_MODEL_DIR" ]]; then
    echo "Model dir: ${BEST_MODEL_DIR}" | tee -a "$MASTER_LOG"
    set +e
    env PYTHONUNBUFFERED=1 "$PYTHON" experiments/run_gnn_explain.py \
        --model_dir "$BEST_MODEL_DIR" \
        --top_k 20 \
        --algorithm gnnexplainer 2>&1 | tee "${LOG_DIR}/gnnexplainer.log"
    GNN_EXIT=$?
    set -e
    echo "[$(timestamp)] GNNExplainer exit=$GNN_EXIT" | tee -a "$MASTER_LOG"
fi

echo "[$(timestamp)] PHASE 3 COMPLETE" | tee -a "$MASTER_LOG"

# ── Phase 4: Archive ────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
ARCHIVE_NAME="experiment_results_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "" | tee -a "$MASTER_LOG"
echo "=== M5 ABLATION SUMMARY ===" | tee -a "$MASTER_LOG"
column -t -s',' "$SUMMARY" | tee -a "$MASTER_LOG"

echo "[$(timestamp)] Creating archive: ${ARCHIVE_NAME}" | tee -a "$MASTER_LOG"
tar -czf "$ARCHIVE_NAME" \
    results/m5_ablation_summary.csv \
    results/master_run.log \
    results/master_resume.log \
    results/m5_*.log \
    results/m3_*.log \
    results/gnnexplainer.log \
    results/results_improved.txt \
    results/my_models/ \
    2>/dev/null

ARCHIVE_SIZE=$(ls -lh "$ARCHIVE_NAME" | awk '{print $5}')
echo "[$(timestamp)] Archive: ${ARCHIVE_NAME} (${ARCHIVE_SIZE})" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] ALL DONE. Shutting down in 2 min..." | tee -a "$MASTER_LOG"

sleep 120
shutdown now
