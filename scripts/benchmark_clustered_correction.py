#!/usr/bin/env python3
"""
Benchmark clustered adjustCounts paths and compare sparse outputs numerically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import soupx
from soupx.core import SoupChannel


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "benchmarks" / "fixtures"


def load_fixture(name: str) -> SoupChannel:
    fixture_dir = FIXTURE_ROOT / name
    toc = sparse.load_npz(fixture_dir / "toc.npz").tocsr()
    tod = sparse.load_npz(fixture_dir / "tod.npz").tocsr()
    metadata = pd.read_csv(fixture_dir / "metadata.csv", index_col=0)
    gene_names = pd.read_csv(fixture_dir / "gene_names.csv")["gene"].tolist()
    channel = SoupChannel(
        tod=tod,
        toc=toc,
        metaData=metadata.copy(),
        gene_names=gene_names,
    )
    channel.setClusters(metadata["clusters"].astype(str).to_numpy())
    channel.set_contamination_fraction(0.05)
    return channel


def matrix_digest(matrix: sparse.spmatrix) -> str:
    csr = matrix.tocsr().copy()
    csr.eliminate_zeros()
    digest = hashlib.sha1()
    digest.update(np.asarray(csr.data).tobytes())
    digest.update(np.asarray(csr.indices).tobytes())
    digest.update(np.asarray(csr.indptr).tobytes())
    return digest.hexdigest()


def run_method(fixture: str, method: str, output_dir: Path | None) -> dict:
    sc_obj = load_fixture(fixture)
    np.random.seed(0)
    start = time.perf_counter()
    corrected = soupx.adjustCounts(
        sc_obj,
        method=method,
        clusters=sc_obj.metaData["clusters"].to_numpy(),
        roundToInt=False,
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    corrected = corrected.tocsr()
    corrected.eliminate_zeros()

    if output_dir is not None:
        fixture_dir = output_dir / fixture
        fixture_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(fixture_dir / f"{method}.npz", corrected)

    return {
        "seconds": elapsed,
        "sum": float(corrected.sum()),
        "nnz": int(corrected.nnz),
        "digest": matrix_digest(corrected),
    }


def compare_outputs(reference_dir: Path, candidate_dir: Path, fixture: str, method: str) -> dict:
    ref = sparse.load_npz(reference_dir / fixture / f"{method}.npz").tocsr()
    cand = sparse.load_npz(candidate_dir / fixture / f"{method}.npz").tocsr()
    diff = (cand - ref).tocsr()
    diff.eliminate_zeros()
    if diff.nnz == 0:
        return {"max_abs_diff": 0.0, "sum_abs_diff": 0.0}
    abs_data = np.abs(diff.data)
    return {
        "max_abs_diff": float(abs_data.max()),
        "sum_abs_diff": float(abs_data.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=["very_small", "small", "medium"],
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["subtraction", "multinomial", "soupOnly"],
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--save-matrices-dir", type=Path)
    parser.add_argument("--compare-matrices-dir", type=Path)
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for fixture in args.fixtures:
        fixture_result = {}
        for method in args.methods:
            fixture_result[method] = run_method(fixture, method, args.save_matrices_dir)
            if args.compare_matrices_dir is not None and args.save_matrices_dir is not None:
                fixture_result[method]["compare"] = compare_outputs(
                    args.compare_matrices_dir,
                    args.save_matrices_dir,
                    fixture,
                    method,
                )
        results[fixture] = fixture_result

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
