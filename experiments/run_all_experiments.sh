#!/usr/bin/env bash
# ============================================================================
# Master Experiment Runner — M5 Ablation + Multi-seed + GNNExplainer
# ============================================================================
# Total estimated time: ~9-10 hours on RTX 5090
#
# Phase 1: M5 ablation (8 experiments × 3 seeds)  →  ~8-9h
# Phase 2: Extra baseline seeds for M3 validation  →  ~40 min
# Phase 3: GNNExplainer on best model              →  ~30 min
# Phase 4: Create results archive & shutdown
#
# Usage:
#   nohup bash experiments/run_all_experiments.sh > results/master_run.log 2>&1 &
#   tail -f results/master_run.log
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="/root/miniconda3/envs/cancer-gnn/bin/python"
LOG_DIR="results"
SUMMARY="${LOG_DIR}/m5_ablation_summary.csv"
MASTER_LOG="${LOG_DIR}/master_run.log"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] MASTER EXPERIMENT RUNNER STARTED" | tee -a "$MASTER_LOG"
echo "  Python: $PYTHON" | tee -a "$MASTER_LOG"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

# ── Base config ──────────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: M5 Ablation (8 experiments × 3 seeds)
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] PHASE 1: M5 ABLATION" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

SEEDS=(72 1 2)

EXPERIMENTS=(
    "baseline|"
    "P0_focal_loss|--focal_gamma 2.0 --focal_alpha 0.75"
    "P9_dropedge|--drop_edge_rate 0.1"
    "P7_heterophily|--heterophily_aware 1"
    "P2_graphmae|--pretrain_graphmae 1 --pretrain_epochs 200"
    "P1_pe_dim16|--pe_dim 16"
    "P3_gps_meta|--gps_meta 1 --gps_heads 4"
    "P4_cross_net_attn|--cross_network_attention 1"
)

extract_metrics() {
    local log_file="$1"
    local aupr auroc
    aupr=$(grep -oP 'AUPR: \K[0-9.]+' "$log_file" | tail -1)
    auroc=$(grep -oP 'AUROC: \K[0-9.]+' "$log_file" | tail -1)
    echo "${aupr:-NA},${auroc:-NA}"
}

echo "experiment,seed,aupr,auroc,duration_sec" > "$SUMMARY"

PHASE1_START=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
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

PHASE1_END=$(date +%s)
PHASE1_DURATION=$(( (PHASE1_END - PHASE1_START) / 60 ))
echo "[$(timestamp)] PHASE 1 COMPLETE (${PHASE1_DURATION} min)" | tee -a "$MASTER_LOG"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Extra baseline seeds for M3 multi-seed validation
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] PHASE 2: EXTRA MULTI-SEED BASELINES" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

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
        echo "[$(timestamp)] FAILED: extra seed=${SEED} (exit=$EXIT_CODE)" | tee -a "$MASTER_LOG"
    else
        METRICS=$(extract_metrics "$RUN_LOG")
        echo "[$(timestamp)] DONE: extra seed=${SEED} -> ${METRICS} (${DURATION}s)" | tee -a "$MASTER_LOG"
    fi
done

echo "[$(timestamp)] PHASE 2 COMPLETE" | tee -a "$MASTER_LOG"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: GNNExplainer on best model
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] PHASE 3: GNNEXPLAINER" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

# Find the best model by AUPR from the summary CSV
BEST_LINE=$(tail -n +2 "$SUMMARY" | grep -v FAILED | sort -t',' -k3 -nr | head -1)
BEST_EXP=$(echo "$BEST_LINE" | cut -d',' -f1)
BEST_SEED=$(echo "$BEST_LINE" | cut -d',' -f2)
BEST_AUPR=$(echo "$BEST_LINE" | cut -d',' -f3)

echo "Best experiment: ${BEST_EXP} seed=${BEST_SEED} AUPR=${BEST_AUPR}" | tee -a "$MASTER_LOG"

# Find the model directory for the best run (most recently created matching dir)
BEST_MODEL_DIR=$(ls -dt results/my_models/GCN_* 2>/dev/null | head -1)

if [[ -z "$BEST_MODEL_DIR" ]]; then
    echo "WARNING: No model directory found. Looking for any EMGNNImproved dir..." | tee -a "$MASTER_LOG"
    BEST_MODEL_DIR=$(ls -dt results/my_models/*/ 2>/dev/null | head -1)
fi

if [[ -n "$BEST_MODEL_DIR" ]]; then
    echo "Model dir: ${BEST_MODEL_DIR}" | tee -a "$MASTER_LOG"

    # Verify required files
    MISSING=""
    for f in model.pkl args.pkl batch.pkl node2idx.pkl meta_x.pkl meta_edge_index.pkl; do
        if [[ ! -f "${BEST_MODEL_DIR}/${f}" ]]; then
            MISSING="${MISSING} ${f}"
        fi
    done

    if [[ -z "$MISSING" ]]; then
        echo "[$(timestamp)] Running GNNExplainer on ${BEST_MODEL_DIR}..." | tee -a "$MASTER_LOG"
        set +e
        env PYTHONUNBUFFERED=1 "$PYTHON" experiments/run_gnn_explain.py \
            --model_dir "$BEST_MODEL_DIR" \
            --top_k 20 \
            --algorithm gnnexplainer 2>&1 | tee "${LOG_DIR}/gnnexplainer.log"
        GNN_EXIT=$?
        set -e

        if [[ $GNN_EXIT -eq 0 ]]; then
            echo "[$(timestamp)] GNNExplainer complete" | tee -a "$MASTER_LOG"
        else
            echo "[$(timestamp)] GNNExplainer FAILED (exit=$GNN_EXIT)" | tee -a "$MASTER_LOG"
        fi
    else
        echo "WARNING: Missing files in model dir:${MISSING}" | tee -a "$MASTER_LOG"
        echo "Skipping GNNExplainer." | tee -a "$MASTER_LOG"
    fi
else
    echo "WARNING: No model directory found. Skipping GNNExplainer." | tee -a "$MASTER_LOG"
fi

echo "[$(timestamp)] PHASE 3 COMPLETE" | tee -a "$MASTER_LOG"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Results archive & shutdown
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] PHASE 4: ARCHIVE & SHUTDOWN" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( (TOTAL_END - PHASE1_START) / 60 ))

# Print final summary
echo "" | tee -a "$MASTER_LOG"
echo "=== M5 ABLATION SUMMARY ===" | tee -a "$MASTER_LOG"
column -t -s',' "$SUMMARY" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Total wall time: ${TOTAL_DURATION} min" | tee -a "$MASTER_LOG"

# Create archive of all results
ARCHIVE_NAME="experiment_results_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "[$(timestamp)] Creating archive: ${ARCHIVE_NAME}" | tee -a "$MASTER_LOG"
tar -czf "$ARCHIVE_NAME" \
    results/m5_ablation_summary.csv \
    results/master_run.log \
    results/m5_*.log \
    results/m3_*.log \
    results/gnnexplainer.log \
    results/results_improved.txt \
    results/my_models/ \
    2>/dev/null || tar -czf "$ARCHIVE_NAME" \
    results/m5_ablation_summary.csv \
    results/master_run.log \
    results/results_improved.txt \
    2>/dev/null

ARCHIVE_SIZE=$(ls -lh "$ARCHIVE_NAME" | awk '{print $5}')
echo "[$(timestamp)] Archive created: ${ARCHIVE_NAME} (${ARCHIVE_SIZE})" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] Archive path: $(pwd)/${ARCHIVE_NAME}" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "[$(timestamp)] ALL EXPERIMENTS COMPLETE" | tee -a "$MASTER_LOG"
echo "  Total wall time: ${TOTAL_DURATION} min" | tee -a "$MASTER_LOG"
echo "  Summary:  ${SUMMARY}" | tee -a "$MASTER_LOG"
echo "  Archive:  ${ARCHIVE_NAME}" | tee -a "$MASTER_LOG"
echo "  Shutting down in 2 minutes..." | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

# Wait 2 min to allow log flush, then shutdown
sleep 120
shutdown now
