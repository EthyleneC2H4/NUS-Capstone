#!/usr/bin/env bash
# ============================================================================
# Parallel experiment runner — 4 concurrent runs on RTX 5090 (7GB each, 28GB/32GB)
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="/root/miniconda3/envs/cancer-gnn/bin/python"
LOG_DIR="results"
SUMMARY="${LOG_DIR}/m5_ablation_summary.csv"
MASTER_LOG="${LOG_DIR}/master_parallel.log"

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

echo "============================================================" | tee "$MASTER_LOG"
echo "[$(timestamp)] PARALLEL RUN: 4 concurrent experiments" | tee -a "$MASTER_LOG"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

# ── Batch 1: 4 parallel runs ────────────────────────────────────────────────
JOBS=(
    "P4_cross_net_attn|72|--cross_network_attention 1"
    "P4_cross_net_attn|1|--cross_network_attention 1"
    "P4_cross_net_attn|2|--cross_network_attention 1"
    "baseline_extra|42|"
)

PIDS=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r EXP_NAME SEED EXTRA_FLAGS <<< "$job"

    RUN_LOG="${LOG_DIR}/m5_${EXP_NAME}_seed${SEED}.log"

    echo "[$(timestamp)] LAUNCH: ${EXP_NAME} seed=${SEED}" | tee -a "$MASTER_LOG"

    CMD=("$PYTHON" experiments/run_improved.py
         "${BASE_FLAGS[@]}"
         --seed "$SEED")

    if [[ -n "$EXTRA_FLAGS" ]]; then
        read -ra EXTRA_ARRAY <<< "$EXTRA_FLAGS"
        CMD+=("${EXTRA_ARRAY[@]}")
    fi

    env PYTHONUNBUFFERED=1 "${CMD[@]}" > "$RUN_LOG" 2>&1 &
    PIDS+=($!)
    sleep 2  # stagger to avoid timestamp collision on model dirs
done

echo "[$(timestamp)] Waiting for 4 jobs: ${PIDS[*]}" | tee -a "$MASTER_LOG"

# Wait for all and record results
for i in "${!JOBS[@]}"; do
    IFS='|' read -r EXP_NAME SEED EXTRA_FLAGS <<< "${JOBS[$i]}"
    PID=${PIDS[$i]}
    RUN_LOG="${LOG_DIR}/m5_${EXP_NAME}_seed${SEED}.log"

    wait "$PID" 2>/dev/null || true
    EXIT_CODE=${PIPESTATUS[0]:-$?}

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[$(timestamp)] FAILED: ${EXP_NAME} seed=${SEED} (exit=$EXIT_CODE)" | tee -a "$MASTER_LOG"
        echo "${EXP_NAME},${SEED},FAILED,FAILED,0" >> "$SUMMARY"
    else
        METRICS=$(extract_metrics "$RUN_LOG")
        echo "${EXP_NAME},${SEED},${METRICS},0" >> "$SUMMARY"
        echo "[$(timestamp)] DONE: ${EXP_NAME} seed=${SEED} -> ${METRICS}" | tee -a "$MASTER_LOG"
    fi
done

echo "[$(timestamp)] BATCH 1 COMPLETE" | tee -a "$MASTER_LOG"

# ── Batch 2: last extra seed ────────────────────────────────────────────────
RUN_LOG="${LOG_DIR}/m3_extraseed_99.log"
echo "[$(timestamp)] START: baseline extra seed=99" | tee -a "$MASTER_LOG"

CMD=("$PYTHON" experiments/run_improved.py "${BASE_FLAGS[@]}" --seed 99)
set +e
env PYTHONUNBUFFERED=1 "${CMD[@]}" > "$RUN_LOG" 2>&1
set -e

METRICS=$(extract_metrics "$RUN_LOG")
echo "[$(timestamp)] DONE: extra seed=99 -> ${METRICS}" | tee -a "$MASTER_LOG"

echo "[$(timestamp)] ALL RUNS COMPLETE" | tee -a "$MASTER_LOG"

# ── GNNExplainer on best model ──────────────────────────────────────────────
echo "[$(timestamp)] GNNExplainer..." | tee -a "$MASTER_LOG"

BEST_LINE=$(tail -n +2 "$SUMMARY" | grep -v FAILED | sort -t',' -k3 -nr | head -1)
BEST_EXP=$(echo "$BEST_LINE" | cut -d',' -f1)
BEST_AUPR=$(echo "$BEST_LINE" | cut -d',' -f3)
echo "Best: ${BEST_EXP} AUPR=${BEST_AUPR}" | tee -a "$MASTER_LOG"

BEST_MODEL_DIR=$(ls -dt results/my_models/GCN_* 2>/dev/null | head -1)

if [[ -n "$BEST_MODEL_DIR" ]]; then
    echo "Model dir: ${BEST_MODEL_DIR}" | tee -a "$MASTER_LOG"
    MISSING=""
    for f in model.pkl args.pkl batch.pkl node2idx.pkl meta_x.pkl meta_edge_index.pkl; do
        [[ -f "${BEST_MODEL_DIR}/${f}" ]] || MISSING="${MISSING} ${f}"
    done
    if [[ -z "$MISSING" ]]; then
        set +e
        env PYTHONUNBUFFERED=1 "$PYTHON" experiments/run_gnn_explain.py \
            --model_dir "$BEST_MODEL_DIR" --top_k 20 --algorithm gnnexplainer 2>&1 | tee "${LOG_DIR}/gnnexplainer.log"
        set -e
        echo "[$(timestamp)] GNNExplainer done" | tee -a "$MASTER_LOG"
    else
        echo "WARNING: Missing files:${MISSING}" | tee -a "$MASTER_LOG"
    fi
fi

# ── Archive & shutdown ──────────────────────────────────────────────────────
ARCHIVE_NAME="experiment_results_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "" | tee -a "$MASTER_LOG"
echo "=== M5 ABLATION SUMMARY ===" | tee -a "$MASTER_LOG"
column -t -s',' "$SUMMARY" | tee -a "$MASTER_LOG"

echo "[$(timestamp)] Creating archive: ${ARCHIVE_NAME}" | tee -a "$MASTER_LOG"
tar -czf "$ARCHIVE_NAME" \
    results/m5_ablation_summary.csv \
    results/*.log \
    results/results_improved.txt \
    results/my_models/ 2>/dev/null

ARCHIVE_SIZE=$(ls -lh "$ARCHIVE_NAME" | awk '{print $5}')
echo "[$(timestamp)] Archive: ${ARCHIVE_NAME} (${ARCHIVE_SIZE})" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] ALL DONE. Shutting down in 2 min..." | tee -a "$MASTER_LOG"

sleep 120
shutdown now
