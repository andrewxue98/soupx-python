"""
SoupX Python Implementation

A Python port of the SoupX R package for removing ambient RNA contamination
from droplet-based single-cell RNA sequencing data.

This implementation mirrors the R package behavior exactly.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from .core import SoupChannel
from .estimation import (
    autoEstCont,
    estimateNonExpressingCells,
    quickMarkers
)
from .correction import adjustCounts

__version__ = "0.3.0"

# R-compatible naming (primary interface)
__all__ = [
    "SoupChannel",
    "adjustCounts",
    "autoEstCont",
    "calculateContaminationFraction",
    "estimateNonExpressingCells",
    "quickMarkers",
]


# Convenience functions for Python users (snake_case aliases)
def adjust_counts(*args, **kwargs):
    """Python-style alias for adjustCounts."""
    return adjustCounts(*args, **kwargs)


def auto_est_cont(*args, **kwargs):
    """Python-style alias for autoEstCont."""
    return autoEstCont(*args, **kwargs)


def calculate_contamination_fraction(*args, **kwargs):
    """Python-style alias for calculateContaminationFraction."""
    return calculateContaminationFraction(*args, **kwargs)


def estimate_non_expressing_cells(*args, **kwargs):
    """Python-style alias for estimateNonExpressingCells."""
    return estimateNonExpressingCells(*args, **kwargs)


def quick_markers(*args, **kwargs):
    """Python-style alias for quickMarkers."""
    return quickMarkers(*args, **kwargs)


def load10X(dataDir, **kwargs):
    """
    Load 10X data from cellranger output directory.
    Mimics R's load10X function.

    Parameters
    ----------
    dataDir : str
        Path to cellranger outs folder
    **kwargs
        Additional arguments passed to SoupChannel

    Returns
    -------
    SoupChannel
        Initialized SoupChannel object
    """
    import scanpy as sc
    import os
    from pathlib import Path

    data_path = Path(dataDir)

    # Check for different cellranger output structures
    if (data_path / "filtered_feature_bc_matrix").exists():
        # cellranger v3+
        toc = sc.read_10x_mtx(data_path / "filtered_feature_bc_matrix")
        tod = sc.read_10x_mtx(data_path / "raw_feature_bc_matrix")
    elif (data_path / "filtered_gene_bc_matrices").exists():
        # cellranger v2
        # Find genome folder
        genome_dir = list((data_path / "filtered_gene_bc_matrices").iterdir())[0]
        toc = sc.read_10x_mtx(data_path / "filtered_gene_bc_matrices" / genome_dir.name)
        tod = sc.read_10x_mtx(data_path / "raw_gene_bc_matrices" / genome_dir.name)
    else:
        raise ValueError(f"Could not find 10X data in {dataDir}")

    # Convert to sparse CSR matrices
    toc_sparse = toc.X.T.tocsr()  # Transpose to genes x cells
    tod_sparse = tod.X.T.tocsr()

    # Create metadata
    metaData = pd.DataFrame({
        'nUMIs': np.array(toc_sparse.sum(axis=0)).flatten()
    }, index=toc.obs_names)

    # Try to load clusters if available
    clusters_path = data_path / "analysis" / "clustering" / "graphclust" / "clusters.csv"
    if clusters_path.exists():
        clusters_df = pd.read_csv(clusters_path)
        # Match barcodes
        if 'Barcode' in clusters_df.columns and 'Cluster' in clusters_df.columns:
            cluster_dict = dict(zip(clusters_df['Barcode'], clusters_df['Cluster']))
            metaData['clusters'] = [str(cluster_dict.get(bc, '0')) for bc in toc.obs_names]

    # Create SoupChannel
    sc_obj = SoupChannel(
        tod=tod_sparse,
        toc=toc_sparse,
        metaData=metaData,
        **kwargs
    )

    # Store gene names if available
    if hasattr(toc, 'var_names'):
        sc_obj.gene_names = toc.var_names.tolist()
        # Set soup profile index to gene names
        if sc_obj.soupProfile is not None:
            sc_obj.soupProfile.index = sc_obj.gene_names

    return sc_obj



def calculateContaminationFraction(
        sc,
        nonExpressedGeneList,
        useToEst,
        verbose=True,
        forceAccept=False
):
    """
    Estimate a global contamination fraction from user-supplied gene sets.

    This mirrors SoupX's manual marker-based estimation path: aggregate counts
    from cells known not to express a gene set, fit a Poisson GLM with expected
    soup counts as an offset, and store the resulting global rho on the channel.
    """
    import statsmodels.api as sm

    if not isinstance(sc, SoupChannel):
        raise TypeError("sc must be a SoupChannel object")

    if isinstance(nonExpressedGeneList, dict):
        gene_set_items = []
        for name, genes in nonExpressedGeneList.items():
            if isinstance(genes, (str, int, np.integer)):
                gene_set_items.append((str(name), [genes]))
            else:
                gene_set_items.append((str(name), list(genes)))
    elif isinstance(nonExpressedGeneList, list):
        if not nonExpressedGeneList:
            raise ValueError("nonExpressedGeneList must contain at least one gene set.")

        first_item = nonExpressedGeneList[0]
        if isinstance(first_item, (str, int, np.integer)):
            gene_set_items = [("set_1", list(nonExpressedGeneList))]
        else:
            gene_set_items = [
                (f"set_{i + 1}", list(gene_set))
                for i, gene_set in enumerate(nonExpressedGeneList)
            ]
    else:
        raise ValueError(
            "nonExpressedGeneList must be a list of gene sets or a dict of named gene sets."
        )

    invalid_sets = []
    for name, genes in gene_set_items:
        if not genes:
            invalid_sets.append(f"{name} is empty")
            continue

        missing_genes = [gene for gene in genes if gene not in sc.soupProfile.index]
        if missing_genes:
            invalid_sets.append(
                f"{name} contains genes not found in data: {missing_genes}"
            )

    if invalid_sets:
        raise ValueError(
            "Invalid nonExpressedGeneList: " + "; ".join(invalid_sets)
        )

    useToEst = np.asarray(useToEst, dtype=bool)
    if useToEst.ndim == 1:
        useToEst = useToEst[:, np.newaxis]

    expected_shape = (sc.n_cells, len(gene_set_items))
    if useToEst.shape != expected_shape:
        raise ValueError(
            f"useToEst must have shape {expected_shape}, got {useToEst.shape}"
        )
    if not useToEst.any():
        raise ValueError(
            "No cells specified as acceptable for estimation. useToEst must not be all FALSE"
        )

    gene_index_map = {
        gene: sc.soupProfile.index.get_loc(gene)
        for _, genes in gene_set_items
        for gene in genes
    }

    df_parts = []
    for set_idx, (name, genes) in enumerate(gene_set_items):
        cell_mask = useToEst[:, set_idx]
        if not np.any(cell_mask):
            continue

        gene_indices = [gene_index_map[gene] for gene in genes]
        soup_frac = sc.soupProfile.loc[genes, 'est'].sum()
        counts = np.asarray(
            sc.toc[gene_indices, :][:, cell_mask].sum(axis=0)
        ).ravel()

        df_parts.append(pd.DataFrame({
            'cells': sc.metaData.index[cell_mask],
            'geneSet': name,
            'soupFrac': soup_frac,
            'counts': counts
        }))

    if not df_parts:
        raise ValueError(
            "No cells specified as acceptable for estimation. useToEst must not be all FALSE"
        )

    df = pd.concat(df_parts, ignore_index=True)
    df['nUMIs'] = sc.metaData.loc[df['cells'], 'nUMIs'].to_numpy()
    df['expSoupCnts'] = df['nUMIs'] * df['soupFrac']

    zero_exp = df['expSoupCnts'] <= 0
    if np.any(zero_exp & (df['counts'] > 0)):
        raise ValueError(
            "Some selected cells have observed marker counts but zero expected soup counts."
        )
    df = df.loc[~zero_exp].copy()
    if df.empty:
        raise ValueError("No expected soup counts available for contamination estimation.")

    fit = sm.GLM(
        df['counts'].to_numpy(),
        np.ones((len(df), 1)),
        family=sm.families.Poisson(),
        offset=np.log(df['expSoupCnts'].to_numpy())
    ).fit()

    rho = float(np.exp(fit.params[0]))
    sc.set_contamination_fraction(rho, forceAccept=forceAccept)

    conf_int = np.asarray(fit.conf_int())
    sc.metaData['rhoLow'] = float(np.exp(conf_int[0, 0]))
    sc.metaData['rhoHigh'] = float(np.exp(conf_int[0, 1]))
    sc.fit = {
        'model': fit,
        'data': df,
        'rhoEst': rho
    }

    if verbose:
        print(f"Estimated global contamination fraction of {100 * rho:0.2f}%")

    return sc
