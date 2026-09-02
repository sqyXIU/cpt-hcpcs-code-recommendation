# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Unit tests for example-F1 and the note-level bootstrap CIs."""

from __future__ import annotations

from cpt_rec.common.evaluation.metrics import (
    bootstrap_set_metrics,
    compute_set_metrics,
)


PAIRS = [
    ({"A", "B"}, {"A", "B"}),       # exact
    ({"A", "C"}, {"A"}),            # 1 FN
    ({"D"}, {"D", "E"}),            # 1 FP
    ({"F"}, {"G"}),                 # miss
]


def test_example_f1_computed():
    m = compute_set_metrics(PAIRS, compute_per_category=False)
    # per-note F1: 1.0, 2*1/3, 2*1/3, 0.0 -> mean = (1 + 2/3 + 2/3 + 0) / 4
    expected = (1.0 + 2 / 3 + 2 / 3 + 0.0) / 4
    assert abs(m.example_f1 - expected) < 1e-9
    assert "example_f1" in m.to_dict()


def test_example_f1_both_empty_is_perfect():
    m = compute_set_metrics([(set(), set())], compute_per_category=False)
    assert m.example_f1 == 1.0


def test_bootstrap_shapes_and_coverage():
    b = bootstrap_set_metrics(PAIRS, n_boot=200, seed=7)
    for name in ("micro_f1", "macro_f1", "example_f1",
                 "exact_match", "jaccard_mean"):
        assert name in b
        s = b[name]
        assert s["ci_lo"] <= s["mean"] <= s["ci_hi"]
        assert s["std"] >= 0.0
    assert b["_meta"]["n_boot"] == 200

    # The point estimate must sit inside (or extremely near) the 95% CI.
    m = compute_set_metrics(PAIRS, compute_per_category=False)
    assert b["micro_f1"]["ci_lo"] - 1e-9 <= m.micro_f1 <= b["micro_f1"]["ci_hi"] + 1e-9


def test_bootstrap_deterministic_by_seed():
    b1 = bootstrap_set_metrics(PAIRS, n_boot=50, seed=123)
    b2 = bootstrap_set_metrics(PAIRS, n_boot=50, seed=123)
    assert b1["micro_f1"] == b2["micro_f1"]
    assert b1["macro_f1"] == b2["macro_f1"]


def test_bootstrap_degenerate_perfect_predictions():
    pairs = [({"A"}, {"A"}), ({"B"}, {"B"})]
    b = bootstrap_set_metrics(pairs, n_boot=25, seed=1)
    assert b["micro_f1"]["mean"] == 1.0
    assert b["micro_f1"]["std"] == 0.0


def test_bootstrap_off_returns_empty():
    assert bootstrap_set_metrics(PAIRS, n_boot=0) == {}
    assert bootstrap_set_metrics([], n_boot=10) == {}
