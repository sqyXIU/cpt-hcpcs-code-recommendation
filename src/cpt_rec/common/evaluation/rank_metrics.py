#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Ranking metrics for CPT/HCPCS **shortlist review**.

The deployment target is a physician-review tool: the system proposes a short,
ranked list of candidate codes and a clinician selects the correct ones.  Under
that workflow a false negative (a manual search of a ~20k-code space, or a
missed charge) and a false positive (a few seconds spent reading and rejecting
one line) are wildly asymmetric, so micro-F1 on a thresholded set — which
prices them equally — is the wrong decision metric.  See
``docs/decisions/2026-08-18_decision_metric_change.md``.

This module computes the replacement suite from a **ranked candidate list per
note** (the ``--save-npz`` companion written by ``cptrec-verifier-predict``):

* ``R@B``   — micro-recall at a review budget of ``B`` codes per note.
  **The primary decision metric** (B = 5 primary, B = 10 secondary).
* ``P@B``   — micro-precision at the same budget.  At a fixed budget this is a
  monotone rescaling of ``R@B`` (``P@B = R@B · n_gold / n_shown``), so the
  budget internalises precision and one number summarises shortlist quality.
* ``Coverage@B`` — share of notes whose **every** gold code is inside the top-B
  ("the coder never has to leave the tool").  The workflow gate.
* ``B@R``   — the dual: mean codes/note needed to reach recall ``R``.  The
  clinician-facing number.
* ``FamilyMRR`` / ``Top1-in-family`` — **required companions.**  ``R@B`` is too
  generous to sibling confusion: with median gold family-rank 3, a shortlist
  containing both siblings scores as a success even though the discrimination
  work simply moved to the clinician.  These keep it measurable.
* ``MAP@B`` / ``nDCG@B`` — within-list scan order.  Reported, not deciding.
* ``AUC-PR`` — threshold-independent summary over all scored pairs.
* ``pool_ceiling`` — recall over the *whole* candidate list, i.e. what the
  ranking is working against.  The gap ``pool_ceiling − R@B`` is the ranking
  cost; reporting R@B without it invites over-reading.

Inputs are plain Python objects — ``note_id -> (codes, scores)`` rankings plus
``note_id -> gold set`` — so nothing here depends on a file format.  See
:mod:`harness` for the CSV/NPZ I/O wrappers.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger(__name__)

#: Pre-registered review budgets (codes shown per note).  B=5 is the primary
#: decision metric, B=10 the secondary; the rest trace the curve.
DEFAULT_BUDGETS: Tuple[int, ...] = (3, 5, 8, 10, 15, 20)

#: Recall targets for the ``B@R`` dual.
DEFAULT_RECALL_TARGETS: Tuple[float, ...] = (0.80, 0.90)

#: A ranked candidate list: codes ordered best-first, with aligned scores.
Ranking = Tuple[List[str], List[float]]


# ---------------------------------------------------------------------------
# Ranking construction
# ---------------------------------------------------------------------------

def order_by_score(codes: Sequence[str], scores: Sequence[float]) -> Ranking:
    """Sort ``(codes, scores)`` best-first.

    Ties keep their original relative order (stable), so a caller that already
    emitted a meaningful pool order does not have it scrambled by ties.
    """
    idx = sorted(range(len(codes)), key=lambda i: -float(scores[i]))
    return ([str(codes[i]).strip().upper() for i in idx],
            [float(scores[i]) for i in idx])


#: Per-note ranking depth kept when loading a *dense* (full-label-space)
#: score matrix.  R@B for the pre-registered budgets is unaffected; it only
#: bounds ``pool_ceiling``, which for such a system must be read as
#: "recall@\ :data:`DENSE_TOPN`" rather than as a retrieval ceiling.
DENSE_TOPN: int = 1000


def load_scores_npz(
    path: Path | str,
    topn: Optional[int] = None,
) -> Dict[str, Ranking]:
    """Load ``note_id -> (codes, scores)`` from a full-pool score NPZ.

    Two on-disk layouts are accepted, so that retrieve-then-rank systems and
    full-label-space classifiers can be scored by the **same** code path — a
    per-system metric definition would make the comparison table unreadable:

    **Ragged** (``note_ids`` / ``candidate_codes`` / ``probs``) — written by
    ``cptrec-verifier-predict --save-npz`` and by ``cptrec-m1-bm25-knn|cptrec-build-kb-index predict
    --dump-scores-npz``.  Each note carries its own retrieved candidate pool
    with scores, before any thresholding, so every metric here is computable
    from one inference pass with no re-training.

    **Dense** (``note_ids`` / ``labels`` / ``scores`` as ``n_notes ×
    n_labels``) — written by ``cptrec-m2-longformer dump-scores``.  The candidate
    pool is the whole training label space, identical for every note; the top
    ``topn`` labels per note are kept (default :data:`DENSE_TOPN`).

    Parameters
    ----------
    topn
        Ranking depth.  ``None`` keeps everything for the ragged layout and
        :data:`DENSE_TOPN` for the dense one.
    """
    import numpy as np

    with np.load(Path(path), allow_pickle=True) as z:
        keys = set(z.files)
        if "candidate_codes" in keys:
            note_ids = z["note_ids"]
            cand = z["candidate_codes"]
            probs = z["probs"]
            layout = "ragged"
        elif "labels" in keys and "scores" in keys:
            note_ids = z["note_ids"]
            labels = np.asarray(z["labels"], dtype=object)
            mat = np.asarray(z["scores"])
            if mat.ndim != 2 or mat.shape[1] != labels.shape[0]:
                raise ValueError(
                    f"{path}: dense layout expects scores of shape "
                    f"(n_notes, n_labels)={len(note_ids)}x{labels.shape[0]}, "
                    f"got {mat.shape}"
                )
            depth = min(int(topn or DENSE_TOPN), labels.shape[0])
            idx = np.argpartition(-mat, depth - 1, axis=1)[:, :depth]
            cand = [labels[row] for row in idx]
            probs = [mat[i, row] for i, row in enumerate(idx)]
            layout = f"dense (top-{depth} of {labels.shape[0]} labels)"
        else:
            raise ValueError(
                f"{path}: unrecognised score NPZ; expected either "
                f"`candidate_codes`+`probs` or `labels`+`scores`, "
                f"got {sorted(keys)}"
            )

    out: Dict[str, Ranking] = {}
    for nid, codes, scores in zip(note_ids, cand, probs):
        ranked = order_by_score(list(codes), list(scores))
        if topn is not None and layout == "ragged":
            ranked = (ranked[0][:topn], ranked[1][:topn])
        out[str(nid).strip()] = ranked
    LOGGER.info(
        "Loaded rankings: %d notes, layout=%s, from %s", len(out), layout, path
    )
    return out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RankMetrics:
    """Shortlist-review metrics over a corpus."""

    n_notes: int
    n_gold_codes: int
    n_notes_no_gold: int
    n_candidates_mean: float

    #: Micro-recall over the entire candidate list — the ranking's ceiling.
    pool_ceiling: float

    #: budget -> metric value
    recall_at: Dict[int, float] = field(default_factory=dict)
    precision_at: Dict[int, float] = field(default_factory=dict)
    coverage_at: Dict[int, float] = field(default_factory=dict)
    map_at: Dict[int, float] = field(default_factory=dict)
    ndcg_at: Dict[int, float] = field(default_factory=dict)
    shown_at: Dict[int, float] = field(default_factory=dict)   # mean codes/note actually shown

    #: recall target -> mean codes/note required (None if unreachable)
    burden_at_recall: Dict[float, Optional[float]] = field(default_factory=dict)

    #: Sibling-discrimination companions (whole-pool variants)
    family_mrr: Optional[float] = None
    top1_in_family: Optional[float] = None
    n_family_scored: int = 0
    #: budget -> FamilyMRR restricted to the shown shortlist
    family_mrr_at: Dict[int, Optional[float]] = field(default_factory=dict)

    auc_pr: Optional[float] = None

    #: bin name -> {"recall_at": {...}, "n_gold_codes": int, "pool_ceiling": float}
    per_bin: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _r(x):
            return None if x is None else round(float(x), 6)

        d = {
            "n_notes": self.n_notes,
            "n_gold_codes": self.n_gold_codes,
            "n_notes_no_gold": self.n_notes_no_gold,
            "n_candidates_mean": _r(self.n_candidates_mean),
            "pool_ceiling": _r(self.pool_ceiling),
            "recall_at": {str(k): _r(v) for k, v in sorted(self.recall_at.items())},
            "precision_at": {str(k): _r(v) for k, v in sorted(self.precision_at.items())},
            "coverage_at": {str(k): _r(v) for k, v in sorted(self.coverage_at.items())},
            "map_at": {str(k): _r(v) for k, v in sorted(self.map_at.items())},
            "ndcg_at": {str(k): _r(v) for k, v in sorted(self.ndcg_at.items())},
            "shown_at": {str(k): _r(v) for k, v in sorted(self.shown_at.items())},
            "burden_at_recall": {
                f"{k:g}": _r(v) for k, v in sorted(self.burden_at_recall.items())
            },
        }
        if self.auc_pr is not None:
            d["auc_pr"] = _r(self.auc_pr)
        if self.n_family_scored:
            d["family"] = {
                "family_mrr": _r(self.family_mrr),
                "top1_in_family": _r(self.top1_in_family),
                "n_gold_scored": self.n_family_scored,
                "family_mrr_at": {
                    str(k): _r(v) for k, v in sorted(self.family_mrr_at.items())
                },
            }
        if self.per_bin:
            d["per_bin"] = self.per_bin
        return d


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _ap_at(ranked: Sequence[str], gold: Set[str], budget: int) -> float:
    """Average precision at cutoff, binary relevance."""
    denom = min(len(gold), budget)
    if denom == 0:
        return 0.0
    hits = 0
    acc = 0.0
    for i, code in enumerate(ranked[:budget], start=1):
        if code in gold:
            hits += 1
            acc += hits / i
    return acc / denom


def _ndcg_at(ranked: Sequence[str], gold: Set[str], budget: int) -> float:
    """Normalized DCG at cutoff, binary relevance, log2 discount."""
    ideal_n = min(len(gold), budget)
    if ideal_n == 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, code in enumerate(ranked[:budget], start=1)
        if code in gold
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _auc_pr(pairs: Sequence[Tuple[float, bool]]) -> Optional[float]:
    """Average precision over all (score, is_relevant) pairs, descending score.

    This is the interpolation-free "average precision" estimator of the area
    under the precision-recall curve — the same convention as
    ``sklearn.metrics.average_precision_score`` — computed here to avoid taking
    a hard dependency for one number.
    """
    total_pos = sum(1 for _s, rel in pairs if rel)
    if total_pos == 0:
        return None
    ordered = sorted(pairs, key=lambda t: -t[0])
    tp = 0
    seen = 0
    prev_recall = 0.0
    ap = 0.0
    for score, rel in ordered:
        seen += 1
        if rel:
            tp += 1
        recall = tp / total_pos
        if rel:
            precision = tp / seen
            ap += precision * (recall - prev_recall)
            prev_recall = recall
    return ap


def _interp_burden(
    curve: Sequence[Tuple[float, float]], target: float
) -> Optional[float]:
    """Smallest burden reaching ``target`` recall, linearly interpolated.

    ``curve`` is ``[(shown_per_note, recall), ...]`` sorted by burden ascending.
    Returns ``None`` when the target is never reached, which is the honest
    answer for a target above the pool ceiling.
    """
    prev_b = prev_r = 0.0
    for burden, recall in curve:
        if recall >= target:
            if recall == prev_r:
                return burden
            frac = (target - prev_r) / (recall - prev_r)
            return prev_b + frac * (burden - prev_b)
        prev_b, prev_r = burden, recall
    return None


def compute_rank_metrics(
    rankings: Dict[str, Ranking],
    gold: Dict[str, Set[str]],
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    recall_targets: Sequence[float] = DEFAULT_RECALL_TARGETS,
    family_of: Optional[Dict[str, Optional[str]]] = None,
    code_bin: Optional[Dict[str, str]] = None,
    compute_auc_pr: bool = True,
) -> RankMetrics:
    """Compute the shortlist-review suite.

    Parameters
    ----------
    rankings
        ``note_id -> (codes, scores)``, best-first.  A gold note absent here is
        scored as an empty candidate list (all its gold codes are misses),
        matching :func:`harness.align`.
    gold
        ``note_id -> set(gold codes)``.  Gold drives the note universe.
    budgets
        Review budgets (codes shown per note) to evaluate.
    recall_targets
        Recall levels for the ``B@R`` dual.
    family_of
        Optional ``code -> family key`` for the sibling companions.  Codes
        mapping to ``None`` are skipped.
    code_bin
        Optional ``code -> head|torso|tail|unseen`` for the per-bin breakdown.
    """
    budgets = sorted({int(b) for b in budgets if int(b) > 0})
    max_budget = max(budgets) if budgets else 0

    n_notes = 0
    n_notes_no_gold = 0
    n_gold_total = 0
    n_cand_total = 0

    tp_at: Dict[int, int] = {b: 0 for b in budgets}
    shown_at: Dict[int, int] = {b: 0 for b in budgets}
    cover_at: Dict[int, int] = {b: 0 for b in budgets}
    ap_sum: Dict[int, float] = {b: 0.0 for b in budgets}
    ndcg_sum: Dict[int, float] = {b: 0.0 for b in budgets}
    n_scored_notes = 0            # notes with >=1 gold code (MAP/nDCG universe)

    pool_tp = 0

    # Per-bin gold accounting: bin -> {"gold": n, "tp@B": {...}, "pool": n}
    bin_gold: Dict[str, int] = defaultdict(int)
    bin_tp: Dict[str, Dict[int, int]] = defaultdict(lambda: {b: 0 for b in budgets})
    bin_pool_tp: Dict[str, int] = defaultdict(int)

    # Family companions
    fam_rr_sum = 0.0
    fam_top1 = 0
    fam_n = 0
    fam_rr_at_sum: Dict[int, float] = {b: 0.0 for b in budgets}
    fam_n_at: Dict[int, int] = {b: 0 for b in budgets}

    auc_pairs: List[Tuple[float, bool]] = []

    # Full curve for B@R: every integer budget up to the longest pool.
    curve_tp: Dict[int, int] = defaultdict(int)
    curve_shown: Dict[int, int] = defaultdict(int)
    max_pool_len = 0

    for nid, gset in gold.items():
        n_notes += 1
        ranked, scores = rankings.get(nid, ([], []))
        n_cand_total += len(ranked)
        max_pool_len = max(max_pool_len, len(ranked))
        if not gset:
            n_notes_no_gold += 1
        else:
            n_scored_notes += 1
        n_gold_total += len(gset)

        rank_of = {}
        for i, code in enumerate(ranked):
            rank_of.setdefault(code, i)

        pool_hits = gset & set(ranked)
        pool_tp += len(pool_hits)

        for b in budgets:
            top = ranked[:b]
            hits = gset & set(top)
            tp_at[b] += len(hits)
            shown_at[b] += min(b, len(ranked))
            if gset <= set(top):
                cover_at[b] += 1
            if gset:
                ap_sum[b] += _ap_at(ranked, gset, b)
                ndcg_sum[b] += _ndcg_at(ranked, gset, b)

        # Dense curve for the B@R dual
        for b in range(1, max(len(ranked), max_budget) + 1):
            curve_shown[b] += min(b, len(ranked))
            if b <= len(ranked) and ranked[b - 1] in gset:
                curve_tp[b] += 1

        if code_bin is not None:
            for g in gset:
                bn = code_bin.get(g, "unseen")
                bin_gold[bn] += 1
                if g in rank_of:
                    bin_pool_tp[bn] += 1
                    for b in budgets:
                        if rank_of[g] < b:
                            bin_tp[bn][b] += 1

        if family_of is not None:
            for g in gset:
                gf = family_of.get(g)
                if gf is None or g not in rank_of:
                    continue
                # Family members among *candidates*, in rank order.
                sibs = [c for c in ranked if family_of.get(c) == gf]
                pos = sibs.index(g) + 1
                fam_rr_sum += 1.0 / pos
                fam_top1 += int(pos == 1)
                fam_n += 1
                for b in budgets:
                    if rank_of[g] >= b:
                        continue
                    sibs_b = [c for c in ranked[:b] if family_of.get(c) == gf]
                    fam_rr_at_sum[b] += 1.0 / (sibs_b.index(g) + 1)
                    fam_n_at[b] += 1

        if compute_auc_pr:
            for code, sc in zip(ranked, scores):
                auc_pairs.append((sc, code in gset))

    # ---- assemble ----
    def _rate(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    recall_at = {b: _rate(tp_at[b], n_gold_total) for b in budgets}
    precision_at = {b: _rate(tp_at[b], shown_at[b]) for b in budgets}
    coverage_at = {b: _rate(cover_at[b], n_notes) for b in budgets}
    map_at = {b: _rate(ap_sum[b], n_scored_notes) for b in budgets}
    ndcg_at = {b: _rate(ndcg_sum[b], n_scored_notes) for b in budgets}
    shown_mean = {b: _rate(shown_at[b], n_notes) for b in budgets}

    # Cumulative curve -> (mean shown per note, cumulative recall)
    curve: List[Tuple[float, float]] = []
    cum_tp = 0
    for b in range(1, max(max_pool_len, max_budget) + 1):
        cum_tp += curve_tp.get(b, 0)
        curve.append((_rate(curve_shown.get(b, 0), n_notes),
                      _rate(cum_tp, n_gold_total)))
    burden_at_recall = {
        float(t): _interp_burden(curve, float(t)) for t in recall_targets
    }

    per_bin: Dict[str, dict] = {}
    if code_bin is not None:
        for bn in sorted(bin_gold):
            per_bin[bn] = {
                "n_gold_codes": bin_gold[bn],
                "pool_ceiling": round(_rate(bin_pool_tp[bn], bin_gold[bn]), 6),
                "recall_at": {
                    str(b): round(_rate(bin_tp[bn][b], bin_gold[bn]), 6)
                    for b in budgets
                },
            }

    result = RankMetrics(
        n_notes=n_notes,
        n_gold_codes=n_gold_total,
        n_notes_no_gold=n_notes_no_gold,
        n_candidates_mean=_rate(n_cand_total, n_notes),
        pool_ceiling=_rate(pool_tp, n_gold_total),
        recall_at=recall_at,
        precision_at=precision_at,
        coverage_at=coverage_at,
        map_at=map_at,
        ndcg_at=ndcg_at,
        shown_at=shown_mean,
        burden_at_recall=burden_at_recall,
        per_bin=per_bin,
    )
    if family_of is not None and fam_n:
        result.family_mrr = fam_rr_sum / fam_n
        result.top1_in_family = fam_top1 / fam_n
        result.n_family_scored = fam_n
        result.family_mrr_at = {
            b: (fam_rr_at_sum[b] / fam_n_at[b]) if fam_n_at[b] else None
            for b in budgets
        }
    if compute_auc_pr and auc_pairs:
        result.auc_pr = _auc_pr(auc_pairs)

    return result


# ---------------------------------------------------------------------------
# Note-level bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_rank_metrics(
    rankings: Dict[str, Ranking],
    gold: Dict[str, Set[str]],
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    n_boot: int = 1000,
    seed: int = 12345,
    ci: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap over *notes* for ``R@B`` and ``Coverage@B``.

    Notes — not codes — are the exchangeable unit, matching
    :func:`metrics.bootstrap_set_metrics` so the two CI blocks are comparable.
    """
    import numpy as np

    budgets = sorted({int(b) for b in budgets if int(b) > 0})
    nids = list(gold.keys())
    n = len(nids)
    if n == 0 or n_boot <= 0 or not budgets:
        return {}

    n_gold = np.zeros(n)
    tp = {b: np.zeros(n) for b in budgets}
    cov = {b: np.zeros(n) for b in budgets}
    for i, nid in enumerate(nids):
        gset = gold[nid]
        ranked, _ = rankings.get(nid, ([], []))
        n_gold[i] = len(gset)
        for b in budgets:
            top = set(ranked[:b])
            tp[b][i] = len(gset & top)
            cov[b][i] = float(gset <= top)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    out: Dict[str, Dict[str, float]] = {}
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    for b in budgets:
        r_samples = np.empty(n_boot)
        c_samples = np.empty(n_boot)
        for k in range(n_boot):
            sel = idx[k]
            denom = n_gold[sel].sum()
            r_samples[k] = tp[b][sel].sum() / denom if denom > 0 else 0.0
            c_samples[k] = cov[b][sel].mean()
        for name, arr in ((f"recall_at_{b}", r_samples), (f"coverage_at_{b}", c_samples)):
            out[name] = {
                "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std(ddof=1)), 6),
                "ci_lo": round(float(np.percentile(arr, lo_q)), 6),
                "ci_hi": round(float(np.percentile(arr, hi_q)), 6),
            }
    out["_meta"] = {"n_boot": float(n_boot), "seed": float(seed), "ci": float(ci)}
    return out
