#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Tests for the evaluation harness.

Covers:
* metrics.py — micro / macro P/R/F1, exact-match, Jaccard, per-category
* code_bins.py — head/torso/tail/unseen binning by training frequency
* calibration.py — per-code threshold tuning + average precision
* constraint_metrics.py — NCCI violation-rate aggregation
* rank_metrics.py — recall@B, coverage@B, family MRR
* harness.py — end-to-end CSV evaluation

Run from project root::

    PYTHONPATH=src python3 tests/test_evaluation.py

The unit tests run on synthetic data and need no corpus.  The "full" tests
additionally need a normalized note corpus and the CMS NCCI tables, and skip
when those are absent — see docs/DATA.md.
"""

from __future__ import annotations

from _skip import skip

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Corpus and data paths.  This repository distributes no clinical notes and no
# licensed code descriptors, so every path below is overridable and the tests
# that need real data skip when it is absent.  See docs/DATA.md.
#
#   CPT_REC_KB_CSV          knowledge base   (default: the shipped public KB)
#   CPT_REC_SAMPLE_CSV      raw note corpus
#   CPT_REC_SAMPLE_NORM_CSV normalized note corpus
#   CPT_REC_NCCI_DIR        CMS NCCI tables  (scripts/download_ncci.py)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
PUBLIC_KB = _REPO / "data" / "kb" / "hcpcs_level2_public.csv"
SAMPLE_NORM_CSV = Path(os.environ.get("CPT_REC_SAMPLE_NORM_CSV",
                                       "outputs/notes/sample_notes_normalized.csv"))
TRAIN_STATS_CSV = Path(os.environ.get("CPT_REC_TRAIN_STATS_CSV",
                                      "outputs/splits/code_frequency_stats.csv"))
NCCI_DIR = Path(os.environ.get("CPT_REC_NCCI_DIR", "data/ncci"))


# ===================================================================
# Test 1.1: metrics.py
# ===================================================================

def test_classify_code():
    from cpt_rec.common.evaluation.metrics import classify_code
    assert classify_code("43235") == "CPT_I"
    assert classify_code("2029F") == "CPT_II"
    assert classify_code("0479T") == "CPT_III"
    assert classify_code("0001U") == "PLA"
    assert classify_code("0002M") == "MAAA"
    assert classify_code("J0585") == "HCPCS"
    assert classify_code("PBHANDEXT") == "UNKNOWN"
    print("PASS: test_classify_code")


def test_compute_set_metrics_toy():
    from cpt_rec.common.evaluation.metrics import compute_set_metrics

    # Note 1: gold={A,B,C}, pred={A,B,D} -> TP=2, FP=1, FN=1
    # Note 2: gold={A},     pred={A}      -> TP=1, FP=0, FN=0 (exact match)
    # Note 3: gold={E},     pred={}       -> TP=0, FP=0, FN=1
    pairs = [
        ({"A", "B", "C"}, {"A", "B", "D"}),
        ({"A"}, {"A"}),
        ({"E"}, set()),
    ]
    m = compute_set_metrics(pairs, compute_per_category=False)

    # Micro: TP=3, FP=1, FN=2
    assert m.n_tp == 3
    assert m.n_fp == 1
    assert m.n_fn == 2
    assert abs(m.micro_precision - 3 / 4) < 1e-9
    assert abs(m.micro_recall - 3 / 5) < 1e-9
    expected_micro_f1 = 2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5))
    assert abs(m.micro_f1 - expected_micro_f1) < 1e-9

    # Exact match: 1/3
    assert abs(m.exact_match - 1 / 3) < 1e-9

    # Jaccard: (2/4 + 1/1 + 0/1) / 3 = (0.5 + 1 + 0) / 3
    assert abs(m.jaccard_mean - (0.5 + 1.0 + 0.0) / 3) < 1e-9

    # Per-code: A appears in gold 2x, pred 2x, both hits -> TP=2
    assert m.per_code["A"] == (2, 0, 0)
    assert m.per_code["D"] == (0, 1, 0)
    assert m.per_code["E"] == (0, 0, 1)

    print(
        f"PASS: test_compute_set_metrics_toy  (micro-F1={m.micro_f1:.4f}, "
        f"macro-F1={m.macro_f1:.4f})"
    )


def test_compute_set_metrics_per_category():
    from cpt_rec.common.evaluation.metrics import compute_set_metrics

    pairs = [
        # CPT_I only
        ({"43235", "45378"}, {"43235", "45380"}),
        # mixed: HCPCS + CPT_I
        ({"J0585", "43235"}, {"J0585"}),
    ]
    m = compute_set_metrics(pairs, compute_per_category=True)
    assert "CPT_I" in m.per_category
    assert "HCPCS" in m.per_category
    cpt_i = m.per_category["CPT_I"]
    # CPT_I: TP=1 (43235 hit in note 1), 43235 hit in note 1 once? Actually:
    # Note1 CPT_I: gold={43235,45378}, pred={43235,45380} -> TP=1, FP=1, FN=1
    # Note2 CPT_I: gold={43235},       pred={}            -> FN=1
    # -> total TP=1, FP=1, FN=2
    assert cpt_i.n_tp == 1
    assert cpt_i.n_fp == 1
    assert cpt_i.n_fn == 2

    hcpcs = m.per_category["HCPCS"]
    # Note1 HCPCS: empty on both -> no contribution
    # Note2 HCPCS: gold={J0585}, pred={J0585} -> TP=1
    assert hcpcs.n_tp == 1
    assert hcpcs.n_fp == 0
    assert hcpcs.n_fn == 0
    print("PASS: test_compute_set_metrics_per_category")


# ===================================================================
# Test 1.2: drift_analysis.py
# ===================================================================

def test_compute_bin_metrics_toy():
    from cpt_rec.common.evaluation.code_bins import compute_bin_metrics

    # A=head, B=torso, C=tail, D=unseen (not in code_bin)
    code_bin = {"A": "head", "B": "torso", "C": "tail"}
    pairs = [
        ({"A", "B", "D"}, {"A", "B", "C"}),  # TP: A,B  FP: C  FN: D
        ({"A"}, {"A"}),                       # TP: A
        ({"C"}, set()),                       # FN: C
    ]
    bins = compute_bin_metrics(pairs, code_bin)
    # A (head): TP=2, FP=0, FN=0
    # B (torso): TP=1, FP=0, FN=0
    # C (tail): TP=0, FP=1, FN=1
    # D (unseen): TP=0, FP=0, FN=1
    assert "head" in bins and bins["head"].n_tp == 2
    assert "torso" in bins and bins["torso"].n_tp == 1
    assert "tail" in bins and bins["tail"].n_tp == 0
    assert bins["tail"].n_fn == 1
    assert "unseen" in bins and bins["unseen"].n_fn == 1
    print("PASS: test_compute_bin_metrics_toy")


def test_load_code_bins_sample():
    from cpt_rec.common.evaluation.code_bins import load_code_bins
    if not TRAIN_STATS_CSV.exists():
        skip(f"needs training-frequency stats at {TRAIN_STATS_CSV}")
        return
    code_bin = load_code_bins(TRAIN_STATS_CSV)
    assert len(code_bin) > 1000, "Expected >1000 codes in training stats"
    # 45378 is typically the most frequent CPT in a GI-heavy corpus
    assert code_bin.get("45378") == "head", f"45378 bin: {code_bin.get('45378')}"
    print(
        f"PASS: test_load_code_bins_sample ({len(code_bin)} codes, "
        f"45378={code_bin.get('45378')})"
    )


# ===================================================================
# Test 1.3: calibration.py
# ===================================================================

def test_best_threshold_and_ap():
    from cpt_rec.common.evaluation.calibration import (
        _average_precision,
        _best_threshold_for_code,
    )
    rng = np.random.default_rng(42)
    # Synthetic code with separable positives and negatives
    pos_scores = rng.uniform(0.6, 1.0, size=50)
    neg_scores = rng.uniform(0.0, 0.4, size=200)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(50), np.zeros(200)]).astype(int)

    t, f1 = _best_threshold_for_code(scores, labels, n_thresholds=100)
    assert 0.4 <= t <= 0.7, f"Threshold out of expected band: {t}"
    assert f1 > 0.95, f"Expected near-perfect F1 on separable data, got {f1}"

    ap = _average_precision(scores, labels)
    assert ap > 0.95, f"Expected near-perfect AP, got {ap}"
    print(
        f"PASS: test_best_threshold_and_ap (t={t:.3f}, f1={f1:.4f}, ap={ap:.4f})"
    )


def test_calibrate_thresholds_pipeline():
    from cpt_rec.common.evaluation.calibration import (
        CalibrationResult,
        calibrate_thresholds,
    )
    rng = np.random.default_rng(0)
    data = {}
    for code in ["A", "B", "C", "RARE"]:
        n = 500
        labels = rng.binomial(1, 0.3 if code != "RARE" else 0.02, size=n)
        # positive scores drawn higher
        scores = np.where(
            labels == 1,
            rng.uniform(0.5, 1.0, size=n),
            rng.uniform(0.0, 0.5, size=n),
        )
        data[code] = (scores, labels)

    cal = calibrate_thresholds(data, min_positives=10, n_thresholds=50)

    # A, B, C tuned; RARE likely falls back (positives probably < 10)
    assert cal.n_codes_tuned >= 3
    for code in ["A", "B", "C"]:
        assert code in cal.per_code, f"{code} should be tuned"
        assert 0.0 < cal.per_code[code] < 1.0

    # Round-trip save/load
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cal.json"
        cal.save(p)
        loaded = CalibrationResult.load(p)
        assert loaded.global_threshold == cal.global_threshold
        assert loaded.per_code == cal.per_code

    print(
        f"PASS: test_calibrate_thresholds_pipeline  "
        f"(tuned={cal.n_codes_tuned}, fallback={cal.n_codes_fallback}, "
        f"macro_ap={cal.macro_ap:.4f})"
    )


# ===================================================================
# Test 1.4: constraint_metrics.py
# ===================================================================

def test_constraint_metrics_with_fake_checker():
    """Use a mock checker to avoid needing the full NCCI data dir."""
    from dataclasses import dataclass
    from cpt_rec.common.evaluation.constraint_metrics import (
        compute_constraint_metrics,
    )
    from cpt_rec.common.ncci.rule_checker import (
        AOCViolation,
        CheckResult,
        PTPViolation,
    )

    class MockChecker:
        def check(self, code_set, code_units=None):
            hard, cont, aoc, mue = [], [], [], []
            # Hard PTP if both 'A' and 'B' present
            if "A" in code_set and "B" in code_set:
                hard.append(PTPViolation("A", "B", 0, "mock"))
            # AOC: 'ADDON' present without 'PRIM'
            if "ADDON" in code_set and "PRIM" not in code_set:
                aoc.append(AOCViolation("ADDON", {"PRIM"}, False))
            return CheckResult(
                code_set=code_set,
                hard_ptp_violations=hard,
                modifier_contingent_ptp=cont,
                aoc_violations=aoc,
                mue_violations=mue,
            )

    pred_sets = [
        {"A", "B"},           # hard PTP
        {"A", "C"},           # clean
        {"ADDON"},            # AOC orphan
        {"ADDON", "PRIM"},    # clean
        {"A", "B", "ADDON"},  # hard PTP + AOC orphan
    ]
    cm = compute_constraint_metrics(pred_sets, MockChecker())
    assert cm.n_notes == 5
    assert cm.n_notes_hard_ptp == 2  # sets 0 and 4
    assert cm.n_notes_aoc_hard == 2   # sets 2 and 4
    assert cm.rate_hard_ptp == 2 / 5
    assert cm.rate_aoc_hard == 2 / 5
    # Hard-valid: sets with no hard_ptp AND no hard aoc -> sets 1, 3
    assert cm.n_notes_hard_valid == 2
    print(
        f"PASS: test_constraint_metrics_with_fake_checker "
        f"(rate_hard_ptp={cm.rate_hard_ptp:.2f}, rate_aoc={cm.rate_aoc_hard:.2f})"
    )


def test_constraint_metrics_real_checker():
    """Real NCCI checker on synthetic code sets — smoke test."""
    from cpt_rec.common.evaluation.constraint_metrics import (
        compute_constraint_metrics,
    )
    from cpt_rec.common.ncci.rule_checker import NCCIRuleChecker

    if not NCCI_DIR.exists():
        skip(f"needs the CMS NCCI tables at {NCCI_DIR}")
        return

    checker = NCCIRuleChecker.from_data_dir(NCCI_DIR)
    # Use a handful of common real codes.  We don't assert specific rates
    # here because NCCI data changes; we only verify the pipeline runs.
    pred_sets = [
        {"43235", "45378"},
        {"66984", "69436"},
        {"43235"},
    ]
    cm = compute_constraint_metrics(pred_sets, checker)
    assert cm.n_notes == 3
    assert 0 <= cm.rate_hard_ptp <= 1
    assert 0 <= cm.rate_hard_valid <= 1
    print(
        f"PASS: test_constraint_metrics_real_checker "
        f"(n={cm.n_notes}, rate_hard_ptp={cm.rate_hard_ptp:.3f}, "
        f"rate_hard_valid={cm.rate_hard_valid:.3f})"
    )


# ===================================================================
# Test 1.5: harness.py — end-to-end
# ===================================================================

def test_harness_end_to_end_sample():
    """
    End-to-end harness test using the DEID sample as both gold AND predictions
    (perfect predictor).  Should yield micro-F1 == 1.0.
    """
    from cpt_rec.common.evaluation.harness import run_evaluation

    if not SAMPLE_NORM_CSV.exists():
        skip(f"needs a normalized note corpus at {SAMPLE_NORM_CSV}")
        return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Build a "predictions" CSV that copies gold proc_codes verbatim
        df = pd.read_csv(SAMPLE_NORM_CSV, dtype=str)
        id_col = "NOTE_ID" if "NOTE_ID" in df.columns else "note_id"
        pred = pd.DataFrame({
            "note_id": df[id_col].values,
            "pred_codes": df["proc_codes"].values,
        })
        pred_path = td / "perfect_predictions.csv"
        pred.to_csv(pred_path, index=False)

        out_dir = td / "eval"
        agg = run_evaluation(
            predictions_csv=pred_path,
            gold_csv=SAMPLE_NORM_CSV,
            out_dir=out_dir,
            train_stats_csv=TRAIN_STATS_CSV if TRAIN_STATS_CSV.exists() else None,
            ncci_dir=None,
        )

        s = agg["set"]
        assert abs(s["micro_f1"] - 1.0) < 1e-9, f"Expected perfect micro-F1, got {s['micro_f1']}"
        assert abs(s["exact_match"] - 1.0) < 1e-9

        # Output files exist
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "per_code_metrics.csv").exists()
        assert (out_dir / "summary.txt").exists()
        if TRAIN_STATS_CSV.exists():
            assert (out_dir / "per_bin_metrics.csv").exists()
            assert "per_bin" in agg
            # All bins present should have micro_f1=1.0 under perfect prediction
            for b, bm in agg["per_bin"].items():
                assert bm["micro_f1"] == 1.0, f"Bin {b} not perfect: {bm}"

    print("PASS: test_harness_end_to_end_sample")


def test_harness_degraded_predictions_sample():
    """
    Degrade predictions by dropping 1 random code per note, verify metrics
    reflect lower F1 but still sane.
    """
    from cpt_rec.common.evaluation.harness import run_evaluation

    if not SAMPLE_NORM_CSV.exists():
        skip(f"needs a normalized note corpus at {SAMPLE_NORM_CSV}")
        return

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = pd.read_csv(SAMPLE_NORM_CSV, dtype=str)
        id_col = "NOTE_ID" if "NOTE_ID" in df.columns else "note_id"

        degraded = []
        for _, row in df.iterrows():
            codes = [c for c in str(row["proc_codes"]).split("|") if c]
            if len(codes) > 1:
                drop_idx = rng.integers(0, len(codes))
                codes = [c for i, c in enumerate(codes) if i != drop_idx]
            degraded.append("|".join(codes))
        pred = pd.DataFrame({
            "note_id": df[id_col].values,
            "pred_codes": degraded,
        })
        pred_path = td / "degraded_predictions.csv"
        pred.to_csv(pred_path, index=False)

        out_dir = td / "eval"
        agg = run_evaluation(
            predictions_csv=pred_path,
            gold_csv=SAMPLE_NORM_CSV,
            out_dir=out_dir,
        )
        s = agg["set"]
        # Recall should be below 1 (we dropped codes); precision should be 1 (no FPs).
        assert s["micro_recall"] < 1.0
        assert abs(s["micro_precision"] - 1.0) < 1e-9
        assert s["micro_f1"] < 1.0
        print(
            f"PASS: test_harness_degraded_predictions_sample  "
            f"(micro-P={s['micro_precision']:.4f}, "
            f"micro-R={s['micro_recall']:.4f}, "
            f"micro-F1={s['micro_f1']:.4f})"
        )


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Evaluation harness — test suite")
    print("=" * 60)
    print()

    print("--- Step 1.1: metrics.py ---")
    test_classify_code()
    test_compute_set_metrics_toy()
    test_compute_set_metrics_per_category()
    print()

    print("--- Step 1.2: drift_analysis.py ---")
    test_compute_bin_metrics_toy()
    test_load_code_bins_sample()
    print()

    print("--- Step 1.3: calibration.py ---")
    test_best_threshold_and_ap()
    test_calibrate_thresholds_pipeline()
    print()

    print("--- Step 1.4: constraint_metrics.py ---")
    test_constraint_metrics_with_fake_checker()
    test_constraint_metrics_real_checker()
    print()

    print("--- Step 1.5: harness.py (end-to-end) ---")
    test_harness_end_to_end_sample()
    test_harness_degraded_predictions_sample()
    print()

    print("=" * 60)
    print("ALL EVALUATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
