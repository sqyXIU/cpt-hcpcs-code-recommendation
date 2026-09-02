# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Unit tests for the shortlist-review metrics (R@B and companions).

Every expectation below is hand-computed from ``RANKINGS`` / ``GOLD`` rather
than snapshotted from the implementation, so a regression in the metric
definitions fails loudly instead of silently re-baselining.
"""

from __future__ import annotations

import math

from cpt_rec.common.evaluation.rank_metrics import (
    bootstrap_rank_metrics,
    compute_rank_metrics,
    order_by_score,
)

# Note A: gold {X, Y}; pool ranks X(1) Z(2) Y(3) W(4)
# Note B: gold {P};    pool ranks Q(1) P(2)
RANKINGS = {
    "A": (["X", "Z", "Y", "W"], [0.9, 0.8, 0.7, 0.6]),
    "B": (["Q", "P"], [0.9, 0.5]),
}
GOLD = {"A": {"X", "Y"}, "B": {"P"}}
FAMILY = {"X": "F1", "Z": "F1", "Y": "F2", "W": "F2", "P": "F3", "Q": "F3"}
BINS = {"X": "head", "Y": "tail", "P": "head"}
BUDGETS = [1, 2, 3]


def _m(**kw):
    return compute_rank_metrics(RANKINGS, GOLD, budgets=BUDGETS, **kw)


def test_order_by_score_sorts_desc_and_is_tie_stable():
    codes, scores = order_by_score(["a", "b", "c"], [0.1, 0.9, 0.9])
    assert codes == ["B", "C", "A"]          # ties keep input order; codes upcased
    assert scores == [0.9, 0.9, 0.1]


def test_recall_at_budget():
    m = _m()
    # 3 gold total; top-1 catches X only; top-2 adds P; top-3 adds Y
    assert abs(m.recall_at[1] - 1 / 3) < 1e-9
    assert abs(m.recall_at[2] - 2 / 3) < 1e-9
    assert abs(m.recall_at[3] - 1.0) < 1e-9
    assert abs(m.pool_ceiling - 1.0) < 1e-9


def test_precision_at_budget_uses_codes_actually_shown():
    m = _m()
    # B's pool is only 2 long, so at budget 3 it shows 2, not 3: 5 shown, 3 TP.
    assert abs(m.precision_at[3] - 0.6) < 1e-9
    assert abs(m.shown_at[3] - 2.5) < 1e-9


def test_precision_is_a_rescaling_of_recall_at_fixed_budget():
    """P@B = R@B * n_gold / n_shown — the property that lets R@B stand alone."""
    m = _m()
    for b in BUDGETS:
        rhs = m.recall_at[b] * m.n_gold_codes / (m.shown_at[b] * m.n_notes)
        assert abs(m.precision_at[b] - rhs) < 1e-12


def test_coverage_is_note_level_and_all_or_nothing():
    m = _m()
    assert m.coverage_at[1] == 0.0            # neither note fully covered
    assert abs(m.coverage_at[2] - 0.5) < 1e-9  # only B
    assert abs(m.coverage_at[3] - 1.0) < 1e-9


def test_map_and_ndcg():
    m = _m()
    ap_a = (1 / 1 + 2 / 3) / 2                 # hits at ranks 1 and 3, denom min(2,3)
    ap_b = (1 / 2) / 1                         # hit at rank 2
    assert abs(m.map_at[3] - (ap_a + ap_b) / 2) < 1e-9

    ndcg_a = (1 + 0.5) / (1 + 1 / math.log2(3))
    ndcg_b = (1 / math.log2(3)) / 1
    assert abs(m.ndcg_at[3] - (ndcg_a + ndcg_b) / 2) < 1e-9


def test_burden_at_recall_interpolates_and_reports_unreachable():
    m = compute_rank_metrics(
        RANKINGS, GOLD, budgets=BUDGETS, recall_targets=[0.80, 0.90]
    )
    # curve: (1.0, 1/3) (2.0, 2/3) (2.5, 1.0) -> interpolate inside the last leg
    assert abs(m.burden_at_recall[0.80] - 2.2) < 1e-9
    assert abs(m.burden_at_recall[0.90] - 2.35) < 1e-9

    unreachable = compute_rank_metrics(
        {"A": (["Z"], [1.0])}, {"A": {"X"}}, budgets=[1], recall_targets=[0.5]
    )
    assert unreachable.burden_at_recall[0.5] is None


def test_family_companions_see_sibling_substitution_that_recall_hides():
    m = _m(family_of=FAMILY)
    # X and Y each top their family; P sits below its sibling Q.
    assert abs(m.family_mrr - (1 + 1 + 0.5) / 3) < 1e-9
    assert abs(m.top1_in_family - 2 / 3) < 1e-9
    assert m.n_family_scored == 3
    # R@2 already counts P as found, but FamilyMRR still charges for the swap.
    assert m.recall_at[2] > 0 and m.family_mrr < 1.0


def test_per_bin_recall_is_gold_side_and_bounded_by_its_ceiling():
    m = _m(code_bin=BINS)
    assert m.per_bin["head"]["n_gold_codes"] == 2      # X (A) and P (B)
    assert m.per_bin["tail"]["n_gold_codes"] == 1      # Y (A)
    assert abs(m.per_bin["head"]["recall_at"]["1"] - 0.5) < 1e-9
    assert m.per_bin["tail"]["recall_at"]["1"] == 0.0
    assert abs(m.per_bin["tail"]["recall_at"]["3"] - 1.0) < 1e-9
    for bm in m.per_bin.values():
        for v in bm["recall_at"].values():
            assert v <= bm["pool_ceiling"] + 1e-12


def test_auc_pr():
    m = _m()
    # ranks over all 6 scored pairs: hits at 1, 4, 6 of 3 positives
    expected = (1 / 1) * (1 / 3) + (2 / 4) * (1 / 3) + (3 / 6) * (1 / 3)
    assert abs(m.auc_pr - expected) < 1e-9


def test_missing_ranking_counts_as_all_misses():
    """Matches harness.align: a gold note with no prediction is not skipped."""
    m = compute_rank_metrics({}, {"A": {"X"}}, budgets=[5])
    assert m.recall_at[5] == 0.0
    assert m.pool_ceiling == 0.0
    assert m.n_notes == 1


def test_empty_gold_note_is_vacuously_covered_and_has_no_auc():
    m = compute_rank_metrics({"A": (["X"], [1.0])}, {"A": set()}, budgets=[5])
    assert m.n_notes_no_gold == 1
    assert m.coverage_at[5] == 1.0
    assert m.auc_pr is None


def test_budget_beyond_pool_saturates_at_the_ceiling():
    m = compute_rank_metrics(RANKINGS, GOLD, budgets=[4, 50])
    assert m.recall_at[50] == m.recall_at[4] == m.pool_ceiling
    assert m.shown_at[50] == 3.0        # (4 + 2) / 2 notes — never more than the pool


def test_bootstrap_shapes_and_coverage():
    b = bootstrap_rank_metrics(RANKINGS, GOLD, budgets=[2], n_boot=200, seed=7)
    for name in ("recall_at_2", "coverage_at_2"):
        s = b[name]
        assert s["ci_lo"] <= s["mean"] <= s["ci_hi"]
        assert s["std"] >= 0.0
    assert b["_meta"]["n_boot"] == 200


def test_to_dict_is_json_shaped():
    import json

    d = _m(family_of=FAMILY, code_bin=BINS).to_dict()
    json.dumps(d, sort_keys=True)                # must not raise
    assert set(d["recall_at"]) == {"1", "2", "3"}
    assert d["family"]["n_gold_scored"] == 3
