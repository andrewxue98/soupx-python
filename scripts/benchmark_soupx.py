#!/usr/bin/env python3
"""
Benchmark and verify SoupX hotspots on generated fixtures.
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


def load_fixture(name: str) -> tuple[SoupChannel, dict]:
    fixture_dir = FIXTURE_ROOT / name
    toc = sparse.load_npz(fixture_dir / "toc.npz").tocsr()
    tod = sparse.load_npz(fixture_dir / "tod.npz").tocsr()
    metadata = pd.read_csv(fixture_dir / "metadata.csv", index_col=0)
    gene_names = pd.read_csv(fixture_dir / "gene_names.csv")["gene"].tolist()
    with (fixture_dir / "fixture.json").open("r", encoding="utf-8") as handle:
        fixture_meta = json.load(handle)

    channel = SoupChannel(
        tod=tod,
        toc=toc,
        metaData=metadata.copy(),
        gene_names=gene_names,
    )
    channel.setClusters(metadata["clusters"].astype(str).to_numpy())
    return channel, fixture_meta


def matrix_digest(matrix: sparse.spmatrix) -> str:
    csr = matrix.tocsr()
    digest = hashlib.sha1()
    digest.update(np.asarray(csr.data).tobytes())
    digest.update(np.asarray(csr.indices).tobytes())
    digest.update(np.asarray(csr.indptr).tobytes())
    return digest.hexdigest()


def frame_digest(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    canonical = df.sort_values(list(df.columns)).reset_index(drop=True)
    return hashlib.sha1(
        canonical.to_json(orient="split", double_precision=12).encode("utf-8")
    ).hexdigest()


def run_quick_markers(sc_obj: SoupChannel, fixture_meta: dict) -> dict:
    start = time.perf_counter()
    markers = soupx.quickMarkers(
        sc_obj.toc,
        sc_obj.metaData["clusters"].to_numpy(),
        N=None,
        verbose=False,
        gene_names=sc_obj.gene_names,
    )
    elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "rows": int(len(markers)),
        "digest": frame_digest(markers),
        "tfidf_sum": float(markers["tfidf"].sum()) if not markers.empty else 0.0,
    }


def run_estimate_non_expressing(sc_obj: SoupChannel, fixture_meta: dict) -> dict:
    gene_sets = list(fixture_meta["marker_sets"].values())
    start = time.perf_counter()
    ute = soupx.estimateNonExpressingCells(
        sc_obj,
        gene_sets,
        maximumContamination=0.2,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    ute_arr = np.asarray(ute)
    return {
        "seconds": elapsed,
        "shape": list(ute_arr.shape),
        "true_count": int(ute_arr.sum()),
        "digest": hashlib.sha1(ute_arr.tobytes()).hexdigest(),
    }


def run_auto_est_cont(sc_obj: SoupChannel, fixture_meta: dict) -> dict:
    start = time.perf_counter()
    result = soupx.autoEstCont(
        sc_obj,
        verbose=False,
        tfidfMin=0.5,
        soupQuantile=0.8,
        maxMarkers=30,
    )
    elapsed = time.perf_counter() - start
    fit = result.fit
    estimates = fit.get("estimates", pd.DataFrame())
    return {
        "seconds": elapsed,
        "rho": float(result.contamination_fraction),
        "n_estimates": int(fit.get("n_estimates", len(estimates))),
        "digest": frame_digest(estimates),
    }


def run_adjust_counts(sc_obj: SoupChannel, method: str) -> dict:
    sc_obj.set_contamination_fraction(0.05)
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
    return {
        "seconds": elapsed,
        "sum": float(corrected.sum()),
        "nnz": int(corrected.nnz),
        "digest": matrix_digest(corrected),
    }


def run_fixture(name: str) -> dict:
    results = {"fixture": name}

    sc_obj, fixture_meta = load_fixture(name)
    results["quickMarkers"] = run_quick_markers(sc_obj, fixture_meta)

    sc_obj, fixture_meta = load_fixture(name)
    results["estimateNonExpressingCells"] = run_estimate_non_expressing(sc_obj, fixture_meta)

    sc_obj, fixture_meta = load_fixture(name)
    results["autoEstCont"] = run_auto_est_cont(sc_obj, fixture_meta)

    adjust_results = {}
    for method in ("subtraction", "multinomial", "soupOnly"):
        sc_obj, _ = load_fixture(name)
        adjust_results[method] = run_adjust_counts(sc_obj, method)
    results["adjustCounts"] = adjust_results
    return results


def compare_results(reference: dict, candidate: dict) -> list[str]:
    mismatches = []
    for fixture_name, ref_fixture in reference.items():
        cand_fixture = candidate[fixture_name]
        for section in ("quickMarkers", "estimateNonExpressingCells", "autoEstCont"):
            for key in ref_fixture[section]:
                if key == "seconds":
                    continue
                ref_value = ref_fixture[section][key]
                cand_value = cand_fixture[section][key]
                if isinstance(ref_value, float):
                    if not np.isclose(ref_value, cand_value, atol=1e-10, rtol=1e-10):
                        mismatches.append(f"{fixture_name}.{section}.{key}: {ref_value} != {cand_value}")
                elif ref_value != cand_value:
                    mismatches.append(f"{fixture_name}.{section}.{key}: {ref_value} != {cand_value}")

        for method in ref_fixture["adjustCounts"]:
            for key in ref_fixture["adjustCounts"][method]:
                if key == "seconds":
                    continue
                ref_value = ref_fixture["adjustCounts"][method][key]
                cand_value = cand_fixture["adjustCounts"][method][key]
                if isinstance(ref_value, float):
                    if not np.isclose(ref_value, cand_value, atol=1e-10, rtol=1e-10):
                        mismatches.append(
                            f"{fixture_name}.adjustCounts.{method}.{key}: {ref_value} != {cand_value}"
                        )
                elif ref_value != cand_value:
                    mismatches.append(
                        f"{fixture_name}.adjustCounts.{method}.{key}: {ref_value} != {cand_value}"
                    )

    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=["very_small", "small", "medium"],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()

    results = {fixture: run_fixture(fixture) for fixture in args.fixtures}

    if args.output:
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    else:
        print(json.dumps(results, indent=2))

    if args.compare:
        with args.compare.open("r", encoding="utf-8") as handle:
            reference = json.load(handle)
        mismatches = compare_results(reference, results)
        if mismatches:
            print(json.dumps({"mismatches": mismatches}, indent=2))
            raise SystemExit(1)
        print(json.dumps({"compare": "ok"}, indent=2))


if __name__ == "__main__":
    main()
