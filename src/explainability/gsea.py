"""
Gene Set Enrichment Analysis (GSEA) for predicted cancer genes.

Two modes
---------
1. Enrichr-based (ORA) – given a discrete gene list (top-N predicted genes),
   tests for over-representation in MSigDB / GO / KEGG gene sets.
   Requires an internet connection (Enrichr API).

2. Pre-ranked GSEA – uses continuous prediction scores as the ranking metric,
   which captures graded confidence rather than a hard threshold.
   Relies on a locally stored .gmt file or a named MSigDB library.

Dependencies
------------
    pip install gseapy matplotlib pandas

Usage
-----
    from src.explainability.gsea import GSEAAnalyzer

    # Build score dict from model output
    scores = {gene: prob for gene, prob in zip(gene_names, cancer_probs)}

    gsea = GSEAAnalyzer()
    enr_results = gsea.run_enrichr(top_genes)
    gsea.plot_enrichment_barplot(enr_results, save_path='gsea_bar.pdf')

    preranked = gsea.run_preranked(scores)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False


# ── Curated cancer-relevant Hallmark gene sets ─────────────────────────────────

CANCER_HALLMARKS: List[str] = [
    'HALLMARK_E2F_TARGETS',
    'HALLMARK_G2M_CHECKPOINT',
    'HALLMARK_MYC_TARGETS_V1',
    'HALLMARK_MYC_TARGETS_V2',
    'HALLMARK_APOPTOSIS',
    'HALLMARK_DNA_REPAIR',
    'HALLMARK_P53_PATHWAY',
    'HALLMARK_PI3K_AKT_MTOR_SIGNALING',
    'HALLMARK_HYPOXIA',
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
    'HALLMARK_INFLAMMATORY_RESPONSE',
    'HALLMARK_WNT_BETA_CATENIN_SIGNALING',
    'HALLMARK_HEDGEHOG_SIGNALING',
    'HALLMARK_NOTCH_SIGNALING',
    'HALLMARK_ANGIOGENESIS',
    'HALLMARK_KRAS_SIGNALING_UP',
]


class GSEAAnalyzer:
    """
    Wraps gseapy to provide GSEA for predicted cancer genes.

    Parameters
    ----------
    gene_set_library : str
        Enrichr library name or path to a .gmt file.
    organism : str
        'Human' or 'Mouse'.
    fdr_threshold : float
        Adjusted p-value cutoff for significance.
    """

    def __init__(
        self,
        gene_set_library: str = 'MSigDB_Hallmark_2020',
        organism: str = 'human',
        fdr_threshold: float = 0.05,
    ):
        self.gene_set_library = gene_set_library
        self.organism = organism
        self.fdr_threshold = fdr_threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Enrichr (over-representation analysis)
    # ──────────────────────────────────────────────────────────────────────────

    def run_enrichr(
        self,
        gene_list: List[str],
        output_dir: str = './gsea_results/enrichr',
    ) -> pd.DataFrame:
        """
        Run Enrichr over-representation analysis.

        Parameters
        ----------
        gene_list : list of gene symbols (HGNC)
        output_dir : directory for gseapy output files

        Returns
        -------
        DataFrame sorted by adjusted p-value (significant hits only).
        """
        self._check_gseapy()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=self.gene_set_library,
            organism=self.organism,
            outdir=output_dir,
            no_plot=True,
            verbose=False,
        )
        results = enr.results.copy()
        results = results.rename(columns={'Adjusted P-value': 'FDR',
                                          'P-value': 'pvalue'})
        significant = (results[results['FDR'] < self.fdr_threshold]
                       .sort_values('FDR')
                       .reset_index(drop=True))
        return significant

    # ──────────────────────────────────────────────────────────────────────────
    # Pre-ranked GSEA
    # ──────────────────────────────────────────────────────────────────────────

    def run_preranked(
        self,
        gene_scores: Dict[str, float],
        gene_set_library: Optional[str] = None,
        permutation_num: int = 1000,
        output_dir: str = './gsea_results/preranked',
    ) -> pd.DataFrame:
        """
        Run pre-ranked GSEA using continuous prediction scores as ranking.

        Parameters
        ----------
        gene_scores : {gene_symbol: prediction_score}
        gene_set_library : override library (path to .gmt or Enrichr name)
        permutation_num : number of permutations for p-value estimation
        output_dir : output directory

        Returns
        -------
        DataFrame of enrichment results from gseapy.
        """
        self._check_gseapy()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        ranked = (pd.Series(gene_scores)
                    .sort_values(ascending=False)
                    .reset_index())
        ranked.columns = ['gene', 'score']

        library = gene_set_library or self.gene_set_library

        pre_res = gp.prerank(
            rnk=ranked,
            gene_sets=library,
            processes=4,
            permutation_num=permutation_num,
            outdir=output_dir,
            no_plot=True,
            seed=42,
            verbose=False,
        )
        return pre_res.res2d

    # ──────────────────────────────────────────────────────────────────────────
    # Hallmark overlap (lightweight, no network call required)
    # ──────────────────────────────────────────────────────────────────────────

    def hallmark_overlap(
        self,
        gene_list: List[str],
        hallmark_gmt_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute simple overlap statistics between *gene_list* and cancer
        Hallmark gene sets loaded from a local .gmt file.

        Returns a DataFrame with columns:
            Term | GeneSet_Size | Overlap | Overlap_Ratio | Genes_Overlapping
        """
        if hallmark_gmt_path is None:
            raise ValueError(
                "Provide hallmark_gmt_path (download h.all.v*.symbols.gmt "
                "from https://www.gsea-msigdb.org/gsea/msigdb/)"
            )
        gene_set = set(g.upper() for g in gene_list)
        gmt = self._parse_gmt(hallmark_gmt_path)

        rows = []
        for term, genes in gmt.items():
            overlap = gene_set & set(g.upper() for g in genes)
            rows.append({
                'Term': term,
                'GeneSet_Size': len(genes),
                'Overlap': len(overlap),
                'Overlap_Ratio': len(overlap) / max(len(genes), 1),
                'Genes_Overlapping': ';'.join(sorted(overlap)),
            })
        df = pd.DataFrame(rows).sort_values('Overlap_Ratio', ascending=False)
        return df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def plot_enrichment_barplot(
        self,
        results: pd.DataFrame,
        fdr_col: str = 'FDR',
        term_col: str = 'Term',
        top_n: int = 20,
        title: str = 'Gene Set Enrichment',
        save_path: Optional[str] = None,
    ):
        """Bar plot of top enriched gene sets (-log10 FDR)."""
        import matplotlib.pyplot as plt

        top = results.head(top_n).copy()
        top['-log10(FDR)'] = -np.log10(top[fdr_col].clip(lower=1e-10))

        fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.4 + 1)))
        ax.barh(range(len(top)), top['-log10(FDR)'], color='steelblue',
                edgecolor='white')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(
            top[term_col].str.replace('HALLMARK_', '').str.replace('_', ' '),
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel('-log₁₀(FDR-adjusted p-value)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--',
                   alpha=0.7, label='FDR = 0.05')
        ax.legend(fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved enrichment plot → {save_path}")
        return fig

    def plot_dotplot(
        self,
        results: pd.DataFrame,
        fdr_col: str = 'FDR',
        term_col: str = 'Term',
        overlap_col: str = 'Overlap',
        top_n: int = 20,
        save_path: Optional[str] = None,
    ):
        """Dot plot: x = -log10(FDR), dot size = overlap count."""
        import matplotlib.pyplot as plt

        top = results.head(top_n).copy()
        top['-log10(FDR)'] = -np.log10(top[fdr_col].clip(lower=1e-10))

        fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.45 + 1)))
        sc = ax.scatter(
            top['-log10(FDR)'],
            range(len(top)),
            s=top[overlap_col] * 10,
            c=top['-log10(FDR)'],
            cmap='RdYlGn',
            edgecolors='grey',
            linewidths=0.5,
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(
            top[term_col].str.replace('HALLMARK_', '').str.replace('_', ' '),
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel('-log₁₀(FDR)', fontsize=11)
        ax.set_title('GSEA Dot Plot', fontsize=13, fontweight='bold')
        plt.colorbar(sc, ax=ax, label='-log₁₀(FDR)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_gmt(path: str) -> Dict[str, List[str]]:
        gmt = {}
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                term  = parts[0]
                genes = [g for g in parts[2:] if g]
                gmt[term] = genes
        return gmt

    @staticmethod
    def _check_gseapy():
        if not GSEAPY_AVAILABLE:
            raise ImportError(
                "gseapy is required for this method. "
                "Install with: pip install gseapy"
            )
