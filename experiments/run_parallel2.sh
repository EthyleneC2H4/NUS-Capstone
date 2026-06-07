#!/usr/bin/env bash
# ============================================================================
# 2× parallel runner — avoids GPU deadlock with 4 concurrent processes
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="/root/miniconda3/envs/cancer-gnn/bin/python"
LOG_DIR="results"
SUMMARY="${LOG_DIR}/m5_ablation_summary.csv"
MASTER_LOG="${LOG_DIR}/master_parallel2.log"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

extract_metrics() {
    local log_file="$1"
    local aupr auroc
    aupr=$(grep -oP 'AUPR: \K[0-9.]+' "$log_file" | tail -1)
    auroc=$(grep -oP 'AUROC: \K[0-9.]+' "$log_file" | tail -1)
    echo "${aupr:-NA},${auroc:-NA}"
}

BASE_FLAGS=(
    --gcn 1 --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB
    --norm_type none --use_residual True --use_net_weights True
    --lr_scheduler cosine --label_smoothing 0.05
    --hidden 64 --n_layers 3 --dropout 0.5 --epochs 2000 --patience 250
)

echo "============================================================" | tee "$MASTER_LOG"
echo "[$(timestamp)] 2× PARALLEL RUN" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

run_job() {
    local EXP_NAME="$1" SEED="$2" EXTRA_FLAGS="$3"
    local RUN_LOG="${LOG_DIR}/m5_${EXP_NAME}_seed${SEED}.log"
    echo "[$(timestamp)] START: ${EXP_NAME} seed=${SEED}" | tee -a "$MASTER_LOG"
    local CMD=("$PYTHON" experiments/run_improved.py "${BASE_FLAGS[@]}" --seed "$SEED")
    if [[ -n "$EXTRA_FLAGS" ]]; then
        read -ra EXTRA <<< "$EXTRA_FLAGS"
        CMD+=("${EXTRA[@]}")
    fi
    env PYTHONUNBUFFERED=1 "${CMD[@]}" > "$RUN_LOG" 2>&1
    local EXIT=$?
    if [[ $EXIT -ne 0 ]]; then
        echo "[$(timestamp)] FAILED: ${EXP_NAME} seed=${SEED}" | tee -a "$MASTER_LOG"
        echo "${EXP_NAME},${SEED},FAILED,FAILED,0" >> "$SUMMARY"
    else
        local M=$(extract_metrics "$RUN_LOG")
        echo "${EXP_NAME},${SEED},${M},0" >> "$SUMMARY"
        echo "[$(timestamp)] DONE: ${EXP_NAME} seed=${SEED} -> ${M}" | tee -a "$MASTER_LOG"
    fi
}

# ── Batch 1: P4×2 ──────────────────────────────────────────────────────────
run_job "P4_cross_net_attn" 72 "--cross_network_attention 1" &
PID1=$!
sleep 3
run_job "P4_cross_net_attn" 1 "--cross_network_attention 1" &
PID2=$!
wait $PID1 $PID2
echo "[$(timestamp)] BATCH 1 DONE" | tee -a "$MASTER_LOG"

# ── Batch 2: P4 + baseline_extra ────────────────────────────────────────────
run_job "P4_cross_net_attn" 2 "--cross_network_attention 1" &
PID1=$!
sleep 3
run_job "baseline_extra" 42 "" &
PID2=$!
wait $PID1 $PID2
echo "[$(timestamp)] BATCH 2 DONE" | tee -a "$MASTER_LOG"

# ── Batch 3: last extra seed ────────────────────────────────────────────────
run_job "baseline_extra" 99 ""
echo "[$(timestamp)] BATCH 3 DONE" | tee -a "$MASTER_LOG"

# ── GNNExplainer ────────────────────────────────────────────────────────────
echo "[$(timestamp)] GNNExplainer..." | tee -a "$MASTER_LOG"
BEST_LINE=$(tail -n +2 "$SUMMARY" | grep -v FAILED | sort -t',' -k3 -nr | head -1)
echo "Best: $(echo "$BEST_LINE" | cut -d',' -f1,3)" | tee -a "$MASTER_LOG"
BEST_DIR=$(ls -dt results/my_models/GCN_* 2>/dev/null | head -1)
if [[ -n "$BEST_DIR" ]]; then
    MISSING=""
    for f in model.pkl args.pkl batch.pkl node2idx.pkl meta_x.pkl meta_edge_index.pkl; do
        [[ -f "${BEST_DIR}/${f}" ]] || MISSING="${MISSING} ${f}"
    done
    if [[ -z "$MISSING" ]]; then
        env PYTHONUNBUFFERED=1 "$PYTHON" experiments/run_gnn_explain.py \
            --model_dir "$BEST_DIR" --top_k 20 --algorithm gnnexplainer 2>&1 | tee "${LOG_DIR}/gnnexplainer.log"
        echo "[$(timestamp)] GNNExplainer done" | tee -a "$MASTER_LOG"
    fi
fi

# ── Archive ─────────────────────────────────────────────────────────────────
ARCHIVE_NAME="experiment_results_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "" | tee -a "$MASTER_LOG"
echo "=== M5 ABLATION SUMMARY ===" | tee -a "$MASTER_LOG"
column -t -s',' "$SUMMARY" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] Archive: ${ARCHIVE_NAME}" | tee -a "$MASTER_LOG"
tar -czf "$ARCHIVE_NAME" results/m5_ablation_summary.csv results/*.log results/results_improved.txt results/my_models/ 2>/dev/null
echo "[$(timestamp)] Archive: $(ls -lh "$ARCHIVE_NAME" | awk '{print $5}')" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] DONE. Shutting down in 2 min..." | tee -a "$MASTER_LOG"
sleep 120
shutdown now
