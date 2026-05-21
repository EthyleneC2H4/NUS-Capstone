#!/usr/bin/env bash
# ============================================================================
# M5 Ablation Experiments — Advanced GNN Techniques
# ============================================================================
# Run on RTX 5090 server. Total estimated time: ~2.5-3 hours (single seed)
# or ~8-9 hours (3 seeds per experiment).
#
# Usage:
#   chmod +x experiments/run_m5_ablation.sh
#   nohup bash experiments/run_m5_ablation.sh > results/m5_ablation.log 2>&1 &
#
# Each experiment adds ONE technique on top of the 6-network baseline.
# Results are appended to results/results_improved.txt automatically.
# A summary CSV is written to results/m5_ablation_summary.csv at the end.
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."   # project root

PYTHON="${PYTHON:-python}"
LOG_DIR="results"
SUMMARY="${LOG_DIR}/m5_ablation_summary.csv"
mkdir -p "$LOG_DIR"

# ── Common baseline flags (6-network best config, no M5 features) ──────────
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

# ── Seeds to run (change to "72" for quick single-seed, "72 1 2" for 3-seed) ─
SEEDS=(72 1 2)

# ── Experiments ────────────────────────────────────────────────────────────────
# Format: "NAME|EXTRA_FLAGS"
# Each experiment = baseline + EXTRA_FLAGS
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

# ── Helpers ────────────────────────────────────────────────────────────────────

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

extract_metrics() {
    # Parse the last "Test Results" line from stdout
    # Expected format: Test Results  |  AUPR: 0.8067  |  AUROC: 0.9170
    local log_file="$1"
    local aupr auroc
    aupr=$(grep -oP 'AUPR: \K[0-9.]+' "$log_file" | tail -1)
    auroc=$(grep -oP 'AUROC: \K[0-9.]+' "$log_file" | tail -1)
    echo "${aupr:-NA},${auroc:-NA}"
}

# ── Write CSV header ──────────────────────────────────────────────────────────
echo "experiment,seed,aupr,auroc,duration_sec" > "$SUMMARY"

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL_START=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r EXP_NAME EXTRA_FLAGS <<< "$entry"

    for SEED in "${SEEDS[@]}"; do
        RUN_LOG="${LOG_DIR}/m5_${EXP_NAME}_seed${SEED}.log"

        echo "============================================================"
        echo "[$(timestamp)] START: ${EXP_NAME}  seed=${SEED}"
        echo "============================================================"

        START=$(date +%s)

        # Build command
        CMD=("$PYTHON" experiments/run_improved.py
             "${BASE_FLAGS[@]}"
             --seed "$SEED")

        # Append extra flags (word-split intentional)
        if [[ -n "$EXTRA_FLAGS" ]]; then
            read -ra EXTRA_ARRAY <<< "$EXTRA_FLAGS"
            CMD+=("${EXTRA_ARRAY[@]}")
        fi

        # Run
        echo "  Command: ${CMD[*]}"
        set +e
        env PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
        EXIT_CODE=$?
        set -e

        END=$(date +%s)
        DURATION=$((END - START))

        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "[$(timestamp)] FAILED: ${EXP_NAME} seed=${SEED} (exit=$EXIT_CODE, ${DURATION}s)"
            echo "${EXP_NAME},${SEED},FAILED,FAILED,${DURATION}" >> "$SUMMARY"
        else
            METRICS=$(extract_metrics "$RUN_LOG")
            echo "${EXP_NAME},${SEED},${METRICS},${DURATION}" >> "$SUMMARY"
            echo "[$(timestamp)] DONE: ${EXP_NAME} seed=${SEED} -> ${METRICS} (${DURATION}s)"
        fi

        echo ""
    done
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo "============================================================"
echo "[$(timestamp)] ALL M5 EXPERIMENTS COMPLETE"
echo "  Total time: ${TOTAL_DURATION} min"
echo "  Summary:    ${SUMMARY}"
echo "============================================================"

# ── Print summary table ───────────────────────────────────────────────────────
echo ""
echo "=== M5 ABLATION SUMMARY ==="
echo ""
column -t -s',' "$SUMMARY"
echo ""
