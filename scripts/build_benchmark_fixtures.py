#!/usr/bin/env python3
"""
Build small benchmark fixtures from example 10x H5 matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "data"
OUT_DIR = ROOT / "benchmarks" / "fixtures"

RAW_H5 = DATA_DIR / "raw_feature_bc_matrix.h5"
FILTERED_H5 = DATA_DIR / "filtered_feature_bc_matrix.h5"

FIXTURE_SPECS = {
    "very_small": {"cells": 64, "genes": 512, "empty_droplets": 256},
    "small": {"cells": 256, "genes": 1500, "empty_droplets": 1024},
    "medium": {"cells": 512, "genes": 2500, "empty_droplets": 1536},
}


def _load_filtered() -> sc.AnnData:
    adata = sc.read_10x_h5(str(FILTERED_H5))
    adata.var_names_make_unique()
    return adata


def _compute_clusters(adata: sc.AnnData) -> np.ndarray:
    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    sc.pp.highly_variable_genes(work, n_top_genes=min(2000, work.n_vars))
    work = work[:, work.var["highly_variable"]].copy()
    n_components = min(30, work.n_obs - 1, work.n_vars - 1)
    embedding = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(work.X)
    n_clusters = 6
    labels = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=2048,
        n_init=10,
    ).fit_predict(embedding)
    return labels.astype(str)


def _balanced_cell_selection(clusters: np.ndarray, n_cells: int) -> np.ndarray:
    cluster_labels, counts = np.unique(clusters, return_counts=True)
    order = np.argsort(cluster_labels)
    cluster_labels = cluster_labels[order]
    counts = counts[order]

    selected = []
    per_cluster = {label: np.flatnonzero(clusters == label) for label in cluster_labels}

    min_take = max(1, n_cells // len(cluster_labels))
    for label in cluster_labels:
        take = min(min_take, len(per_cluster[label]))
        selected.extend(per_cluster[label][:take].tolist())

    if len(selected) < n_cells:
        remaining = []
        for label in cluster_labels:
            remaining.extend(per_cluster[label][min_take:].tolist())
        selected.extend(remaining[: max(0, n_cells - len(selected))])

    return np.array(sorted(selected[:n_cells]), dtype=int)


def _choose_gene_indices(toc: sparse.csr_matrix, n_genes: int) -> np.ndarray:
    gene_totals = np.asarray(toc.sum(axis=1)).ravel()
    ranked = np.argsort(-gene_totals, kind="stable")
    return np.sort(ranked[:n_genes])


def _sample_empty_droplets(
    raw_h5: Path,
    n_needed: int,
) -> np.ndarray:
    with h5py.File(raw_h5, "r") as handle:
        data = handle["matrix"]["data"]
        indptr = handle["matrix"]["indptr"][:]
        n_cols = len(indptr) - 1
        chunk_cols = 20000
        empty_indices = []

        for start_col in range(0, n_cols, chunk_cols):
            end_col = min(start_col + chunk_cols, n_cols)
            start_ptr = indptr[start_col]
            end_ptr = indptr[end_col]
            chunk_data = data[start_ptr:end_ptr]
            chunk_ptrs = indptr[start_col : end_col + 1] - start_ptr
            if chunk_data.size == 0:
                continue
            prefix = np.concatenate([[0], np.cumsum(chunk_data, dtype=np.int64)])
            umi_counts = prefix[chunk_ptrs[1:]] - prefix[chunk_ptrs[:-1]]
            chunk_cols_idx = np.arange(start_col, end_col)
            chunk_empty = chunk_cols_idx[
                (umi_counts > 0)
                & (umi_counts < 100)
            ]
            empty_indices.extend(chunk_empty.tolist())
            if len(empty_indices) >= n_needed:
                return np.array(empty_indices[:n_needed], dtype=np.int64)

    if len(empty_indices) < n_needed:
        raise ValueError(f"Only found {len(empty_indices)} empty droplets, need {n_needed}")
    return np.array(empty_indices[:n_needed], dtype=np.int64)


def _extract_h5_columns(
    h5_path: Path,
    column_indices: np.ndarray,
    gene_indices: np.ndarray,
) -> sparse.csr_matrix:
    sorted_columns = np.sort(column_indices.astype(np.int64))

    with h5py.File(h5_path, "r") as handle:
        data_ds = handle["matrix"]["data"]
        indices_ds = handle["matrix"]["indices"]
        indptr = handle["matrix"]["indptr"][:]

        gene_map = np.full(int(handle["matrix"]["shape"][0]), -1, dtype=np.int64)
        gene_map[gene_indices] = np.arange(len(gene_indices), dtype=np.int64)

        out_data = []
        out_indices = []
        out_indptr = [0]

        for col_idx in sorted_columns:
            start = indptr[col_idx]
            end = indptr[col_idx + 1]
            rows = indices_ds[start:end]
            values = data_ds[start:end]
            mapped_rows = gene_map[rows]
            keep = mapped_rows >= 0

            if np.any(keep):
                out_indices.append(mapped_rows[keep])
                out_data.append(values[keep])
                out_indptr.append(out_indptr[-1] + int(np.sum(keep)))
            else:
                out_indptr.append(out_indptr[-1])

    data = np.concatenate(out_data) if out_data else np.array([], dtype=np.int32)
    indices = np.concatenate(out_indices) if out_indices else np.array([], dtype=np.int32)
    indptr = np.array(out_indptr, dtype=np.int64)

    matrix = sparse.csc_matrix(
        (data, indices, indptr),
        shape=(len(gene_indices), len(sorted_columns)),
    )
    return matrix.tocsr()


def _marker_gene_sets(
    toc: sparse.csr_matrix,
    gene_names: list[str],
    clusters: np.ndarray,
) -> dict[str, list[str]]:
    cluster_labels = np.unique(clusters)
    cluster_means = {}
    for label in cluster_labels:
        cluster_mask = clusters == label
        cluster_means[label] = np.asarray(toc[:, cluster_mask].mean(axis=1)).ravel()

    overall_mean = np.asarray(toc.mean(axis=1)).ravel() + 1e-6
    marker_sets = {}
    for label in cluster_labels:
        score = cluster_means[label] / overall_mean
        ranked = np.argsort(-score, kind="stable")
        marker_sets[str(label)] = [gene_names[idx] for idx in ranked[:5]]
    return marker_sets


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    filtered = _load_filtered()
    clusters = _compute_clusters(filtered)
    filtered.obs["fixture_clusters"] = clusters
    filtered_barcodes = filtered.obs_names.to_numpy()

    manifest = {
        "source_raw_h5": str(RAW_H5),
        "source_filtered_h5": str(FILTERED_H5),
        "fixtures": {},
    }

    for name, spec in FIXTURE_SPECS.items():
        fixture_dir = OUT_DIR / name
        fixture_dir.mkdir(parents=True, exist_ok=True)

        cell_idx = _balanced_cell_selection(clusters, spec["cells"])
        cell_barcodes = filtered_barcodes[cell_idx]
        toc_cells = filtered.X[cell_idx, :].tocsr()
        gene_idx = _choose_gene_indices(toc_cells.T.tocsr(), spec["genes"])

        raw_subset_idx = _sample_empty_droplets(RAW_H5, spec["empty_droplets"])

        toc = filtered.X[cell_idx, :][:, gene_idx].T.tocsr()
        tod = _extract_h5_columns(RAW_H5, raw_subset_idx, gene_idx)

        gene_names = filtered.var_names.to_numpy()[gene_idx].tolist()
        clusters_subset = clusters[cell_idx]
        metadata = pd.DataFrame(
            {
                "nUMIs": np.asarray(toc.sum(axis=0)).ravel(),
                "clusters": clusters_subset,
            },
            index=cell_barcodes,
        )

        marker_sets = _marker_gene_sets(toc, gene_names, clusters_subset)

        sparse.save_npz(fixture_dir / "toc.npz", toc)
        sparse.save_npz(fixture_dir / "tod.npz", tod)
        metadata.to_csv(fixture_dir / "metadata.csv")
        pd.Series(gene_names, name="gene").to_csv(
            fixture_dir / "gene_names.csv",
            index=False,
        )

        with (fixture_dir / "fixture.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "name": name,
                    "cells": spec["cells"],
                    "genes": spec["genes"],
                    "empty_droplets": spec["empty_droplets"],
                    "marker_sets": marker_sets,
                },
                handle,
                indent=2,
            )

        manifest["fixtures"][name] = {
            "cells": spec["cells"],
            "genes": spec["genes"],
            "empty_droplets": spec["empty_droplets"],
            "raw_columns": len(raw_subset_idx),
        }

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
