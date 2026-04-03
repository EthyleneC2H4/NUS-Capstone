"""
Gene set enrichment analysis on EMGNN predictions (Methodology 4).

Reads predictions.tsv from a trained model directory and runs:
  1. Enrichr ORA on top-N predicted novel genes
  2. Pre-ranked GSEA using continuous prediction scores
  3. Hallmark overlap (no internet required, uses local .gmt file)

Usage
-----
    # Enrichr (requires internet)
    python experiments/run_gsea.py \\
        --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \\
        --mode enrichr --top_n 200

    # Pre-ranked GSEA
    python experiments/run_gsea.py \\
        --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \\
        --mode preranked

    # Local Hallmark overlap (no internet)
    python experiments/run_gsea.py \\
        --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \\
        --mode hallmark \\
        --gmt_path ./data/h.all.v2023.1.Hs.symbols.gmt
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.explainability.gsea import GSEAAnalyzer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True,
                   help='Path to trained model directory')
    p.add_argument('--mode', default='enrichr',
                   choices=['enrichr', 'preranked', 'hallmark'])
    p.add_argument('--top_n',      type=int, default=200,
                   help='Top-N predicted genes for Enrichr ORA')
    p.add_argument('--fdr',        type=float, default=0.05)
    p.add_argument('--gmt_path',   type=str, default=None,
                   help='Path to local .gmt file (required for hallmark mode)')
    p.add_argument('--gene_set_lib', type=str,
                   default='MSigDB_Hallmark_2020')
    p.add_argument('--out_dir',    type=str, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.model_dir, 'gsea')
    os.makedirs(out_dir, exist_ok=True)

    # Load predictions
    pred_path = os.path.join(args.model_dir, 'predictions.tsv')
    if not os.path.exists(pred_path):
        print(f"ERROR: predictions.tsv not found at {pred_path}")
        sys.exit(1)

    df = pd.read_csv(pred_path, sep='\t')
    print(f"Loaded {len(df)} gene predictions from {pred_path}")

    # Sort by cancer probability descending
    df = df.sort_values('Prob_pos', ascending=False).reset_index(drop=True)

    gsea = GSEAAnalyzer(
        gene_set_library=args.gene_set_lib,
        fdr_threshold=args.fdr,
    )

    # ── Mode: Enrichr ─────────────────────────────────────────────────────────
    if args.mode == 'enrichr':
        top_genes = df['Name'].head(args.top_n).tolist()
        print(f"\nRunning Enrichr ORA on top-{args.top_n} predicted genes …")
        results = gsea.run_enrichr(top_genes, output_dir=out_dir)
        if len(results):
            out_csv = os.path.join(out_dir, 'enrichr_results.csv')
            results.to_csv(out_csv, index=False)
            print(f"  {len(results)} significant terms → {out_csv}")
            fig = gsea.plot_enrichment_barplot(
                results,
                title=f'Enrichr: Top-{args.top_n} Predicted Cancer Genes',
                save_path=os.path.join(out_dir, 'enrichr_barplot.pdf'),
            )
        else:
            print("  No significant terms found.")

    # ── Mode: Pre-ranked ──────────────────────────────────────────────────────
    elif args.mode == 'preranked':
        gene_scores = dict(zip(df['Name'], df['Prob_pos']))
        print(f"\nRunning pre-ranked GSEA on {len(gene_scores)} genes …")
        results = gsea.run_preranked(gene_scores, output_dir=out_dir)
        out_csv = os.path.join(out_dir, 'preranked_results.csv')
        results.to_csv(out_csv, index=False)
        print(f"  Results → {out_csv}")

    # ── Mode: Hallmark overlap (local) ────────────────────────────────────────
    elif args.mode == 'hallmark':
        if not args.gmt_path:
            print("ERROR: --gmt_path required for hallmark mode.")
            sys.exit(1)
        top_genes = df['Name'].head(args.top_n).tolist()
        print(f"\nComputing Hallmark overlap for top-{args.top_n} genes …")
        results = gsea.hallmark_overlap(top_genes, hallmark_gmt_path=args.gmt_path)
        out_csv = os.path.join(out_dir, 'hallmark_overlap.csv')
        results.to_csv(out_csv, index=False)
        print(f"  Results → {out_csv}")
        print(results.head(10).to_string(index=False))

    print(f"\nGSEA analysis complete. Outputs in: {out_dir}")


if __name__ == '__main__':
    main()
