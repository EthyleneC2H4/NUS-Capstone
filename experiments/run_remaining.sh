#!/usr/bin/env bash
# Sequential run of remaining experiments (P4 + extra seeds + GNNExplainer)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/root/miniconda3/envs/cancer-gnn/bin/python"
LOG_DIR="results"
SUMMARY="${LOG_DIR}/m5_ablation_summary.csv"
MLOG="${LOG_DIR}/master_remaining.log"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
extract_metrics() {
    local f="$1"
    local a=$(grep -oP 'AUPR: \K[0-9.]+' "$f" | tail -1)
    local r=$(grep -oP 'AUROC: \K[0-9.]+' "$f" | tail -1)
    echo "${a:-NA},${r:-NA}"
}

B=(--gcn 1 --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB --norm_type none --use_residual True --use_net_weights True --lr_scheduler cosine --label_smoothing 0.05 --hidden 64 --n_layers 3 --dropout 0.5 --epochs 2000 --patience 250)

echo "[$(timestamp)] REMAINING EXPERIMENTS (sequential)" | tee "$MLOG"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "$MLOG"

JOBS=(
    "P4_cross_net_attn|72|--cross_network_attention 1"
    "P4_cross_net_attn|1|--cross_network_attention 1"
    "P4_cross_net_attn|2|--cross_network_attention 1"
    "baseline_extra|42|"
    "baseline_extra|99|"
)

for job in "${JOBS[@]}"; do
    IFS='|' read -r NAME SEED EXTRA <<< "$job"
    LOG="${LOG_DIR}/m5_${NAME}_seed${SEED}.log"
    echo "[$(timestamp)] START: ${NAME} seed=${SEED}" | tee -a "$MLOG"
    CMD=("$PYTHON" experiments/run_improved.py "${B[@]}" --seed "$SEED")
    [[ -n "$EXTRA" ]] && { read -ra E <<< "$EXTRA"; CMD+=("${E[@]}"); }
    set +e; env PYTHONUNBUFFERED=1 "${CMD[@]}" > "$LOG" 2>&1; RC=$?; set -e
    if [[ $RC -ne 0 ]]; then
        echo "[$(timestamp)] FAILED: ${NAME} seed=${SEED}" | tee -a "$MLOG"
        echo "${NAME},${SEED},FAILED,FAILED,0" >> "$SUMMARY"
    else
        M=$(extract_metrics "$LOG")
        echo "${NAME},${SEED},${M},0" >> "$SUMMARY"
        echo "[$(timestamp)] DONE: ${NAME} seed=${SEED} -> ${M}" | tee -a "$MLOG"
    fi
done

echo "[$(timestamp)] ALL RUNS DONE" | tee -a "$MLOG"

# GNNExplainer
echo "[$(timestamp)] GNNExplainer..." | tee -a "$MLOG"
BEST=$(tail -n +2 "$SUMMARY" | grep -v FAILED | sort -t',' -k3 -nr | head -1)
echo "Best: $(echo "$BEST" | cut -d',' -f1,3)" | tee -a "$MLOG"
DIR=$(ls -dt results/my_models/GCN_* 2>/dev/null | head -1)
if [[ -n "$DIR" ]]; then
    OK=true
    for f in model.pkl args.pkl batch.pkl node2idx.pkl meta_x.pkl meta_edge_index.pkl; do
        [[ -f "${DIR}/${f}" ]] || { echo "MISSING: ${f}"; OK=false; }
    done
    if $OK; then
        env PYTHONUNBUFFERED=1 "$PYTHON" experiments/run_gnn_explain.py \
            --model_dir "$DIR" --top_k 20 --algorithm gnnexplainer 2>&1 | tee "${LOG_DIR}/gnnexplainer.log"
    fi
fi

# Archive & shutdown
ARCHIVE="experiment_results_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "" | tee -a "$MLOG"
echo "=== SUMMARY ===" | tee -a "$MLOG"
column -t -s',' "$SUMMARY" | tee -a "$MLOG"
echo "[$(timestamp)] Archive: ${ARCHIVE}" | tee -a "$MLOG"
tar -czf "$ARCHIVE" results/m5_ablation_summary.csv results/*.log results/results_improved.txt results/my_models/ 2>/dev/null
ls -lh "$ARCHIVE" | tee -a "$MLOG"
echo "[$(timestamp)] DONE. Shutdown in 2 min..." | tee -a "$MLOG"
sleep 120
shutdown now
