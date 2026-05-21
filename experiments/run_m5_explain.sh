#!/usr/bin/env bash
# ============================================================================
# P10: GNNExplainer — run after M5 ablation to explain best model
# ============================================================================
# Usage:
#   bash experiments/run_m5_explain.sh <model_dir>
#
# Example:
#   bash experiments/run_m5_explain.sh results/my_models/EMGNNImproved_GCN_CPDB_2026_05_22_12_00_00
#
# Estimated time: ~10 min (top 20 genes × 200 GNNExplainer epochs)
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_dir>"
    echo ""
    echo "Pick the best model from the M5 ablation:"
    echo "  ls results/my_models/ | grep EMGNNImproved"
    exit 1
fi

MODEL_DIR="$1"

echo "============================================================"
echo "P10: GNNExplainer on ${MODEL_DIR}"
echo "============================================================"

# Verify required files exist
for f in model.pkl args.pkl batch.pkl node2idx.pkl meta_x.pkl meta_edge_index.pkl; do
    if [[ ! -f "${MODEL_DIR}/${f}" ]]; then
        echo "ERROR: Missing ${MODEL_DIR}/${f}"
        echo "  This model may not have saved all required artefacts."
        echo "  Re-run training with the latest code to regenerate."
        exit 1
    fi
done

env PYTHONUNBUFFERED=1 "$PYTHON" experiments/run_gnn_explain.py \
    --model_dir "$MODEL_DIR" \
    --top_k 20 \
    --algorithm gnnexplainer

echo ""
echo "Done. Results saved to:"
echo "  ${MODEL_DIR}/gnn_explanations.pkl"
echo "  ${MODEL_DIR}/gnn_explanations.csv"
