# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Set-level evaluation metrics for modifier-agnostic CPT/HCPCS prediction.

Every baseline and the proposed model produces a *set* of codes per note.
This module computes the core metrics reported in the paper:

* Micro / macro precision, recall, F1
* Exact-set match, Jaccard
* Per-category breakdown (CPT Cat I / II / III / PLA / MAAA / HCPCS)

Inputs are always iterables of ``(gold_set, pred_set)`` pairs — no dependence
on any particular file format.  See :mod:`harness` for CSV I/O wrappers.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category classification (uses format only; does not require a KB)
# ---------------------------------------------------------------------------

_CAT_I_RE = re.compile(r"^\d{5}$")
_CAT_II_RE = re.compile(r"^\d{4}F$")
_CAT_III_RE = re.compile(r"^\d{4}T$")
_PLA_RE = re.compile(r"^\d{4}U$")
_MAAA_RE = re.compile(r"^\d{4}M$")
_HCPCS_RE = re.compile(r"^[A-Z]\d{4}$")


def classify_code(code: str) -> str:
    """Return one of: 'CPT_I', 'CPT_II', 'CPT_III', 'PLA', 'MAAA', 'HCPCS', 'UNKNOWN'."""
    c = code.strip().upper()
    if _CAT_I_RE.match(c):
        return "CPT_I"
    if _CAT_II_RE.match(c):
        return "CPT_II"
    if _CAT_III_RE.match(c):
        return "CPT_III"
    if _PLA_RE.match(c):
        return "PLA"
    if _MAAA_RE.match(c):
        return "MAAA"
    if _HCPCS_RE.match(c):
        return "HCPCS"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SetMetrics:
    """Aggregate set-level metrics over a corpus."""

    n_notes: int
    n_gold_codes: int
    n_pred_codes: int
    n_tp: int
    n_fp: int
    n_fn: int

    micro_precision: float
    micro_recall: float
    micro_f1: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    exact_match: float
    jaccard_mean: float

    # Example-based (per-note) F1, the standard multi-label per-example metric:
    # mean over notes of 2|G∩P| / (|G|+|P|), with both-empty = 1.0.
    example_f1: float = 0.0

    # Optional per-category and per-code breakdowns
    per_category: Dict[str, "SetMetrics"] = field(default_factory=dict)
    per_code: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)  # code -> (tp, fp, fn)

    def to_dict(self, include_per_code: bool = False) -> dict:
        d = {
            "n_notes": self.n_notes,
            "n_gold_codes": self.n_gold_codes,
            "n_pred_codes": self.n_pred_codes,
            "n_tp": self.n_tp,
            "n_fp": self.n_fp,
            "n_fn": self.n_fn,
            "micro_precision": round(self.micro_precision, 6),
            "micro_recall": round(self.micro_recall, 6),
            "micro_f1": round(self.micro_f1, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "macro_f1": round(self.macro_f1, 6),
            "exact_match": round(self.exact_match, 6),
            "jaccard_mean": round(self.jaccard_mean, 6),
            "example_f1": round(self.example_f1, 6),
        }
        if self.per_category:
            d["per_category"] = {
                k: v.to_dict(include_per_code=False) for k, v in self.per_category.items()
            }
        if include_per_code and self.per_code:
            d["per_code"] = {c: list(v) for c, v in self.per_code.items()}
        return d


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall, F1 from counts — safe on zeros."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f


def _jaccard(gold: Set[str], pred: Set[str]) -> float:
    if not gold and not pred:
        return 1.0
    union = gold | pred
    return len(gold & pred) / len(union) if union else 0.0


def _example_f1(gold: Set[str], pred: Set[str]) -> float:
    """Per-note F1 = 2|G∩P| / (|G|+|P|); both-empty counts as perfect."""
    denom = len(gold) + len(pred)
    if denom == 0:
        return 1.0
    return 2 * len(gold & pred) / denom


def compute_set_metrics(
    pairs: Iterable[Tuple[Set[str], Set[str]]],
    code_universe: Optional[Set[str]] = None,
    compute_per_category: bool = True,
) -> SetMetrics:
    """
    Compute micro / macro P/R/F1, exact-match, Jaccard, per-category metrics.

    Parameters
    ----------
    pairs
        Iterable of ``(gold_set, pred_set)`` per note.
    code_universe
        Optional set of codes that macro averaging should iterate over.
        Defaults to the union of codes observed in gold across the corpus
        (standard multi-label macro).
    compute_per_category
        If True, also produce per-category sub-metrics.
    """
    # Per-code counters
    per_code: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])  # [tp, fp, fn]

    # Corpus totals
    n_notes = 0
    n_gold_codes = 0
    n_pred_codes = 0
    n_exact = 0
    jaccard_sum = 0.0
    example_f1_sum = 0.0

    # Buffer pairs so we can also do per-category later
    buffered: List[Tuple[Set[str], Set[str]]] = []

    for gold, pred in pairs:
        gold = set(gold)
        pred = set(pred)
        buffered.append((gold, pred))

        n_notes += 1
        n_gold_codes += len(gold)
        n_pred_codes += len(pred)

        tp_codes = gold & pred
        fp_codes = pred - gold
        fn_codes = gold - pred

        for c in tp_codes:
            per_code[c][0] += 1
        for c in fp_codes:
            per_code[c][1] += 1
        for c in fn_codes:
            per_code[c][2] += 1

        if gold == pred:
            n_exact += 1
        jaccard_sum += _jaccard(gold, pred)
        example_f1_sum += _example_f1(gold, pred)

    # Determine macro universe
    if code_universe is None:
        code_universe = {c for c, (tp, fp, fn) in per_code.items() if (tp + fn) > 0}

    # Micro counts
    micro_tp = sum(v[0] for v in per_code.values())
    micro_fp = sum(v[1] for v in per_code.values())
    micro_fn = sum(v[2] for v in per_code.values())
    micro_p, micro_r, micro_f = _prf(micro_tp, micro_fp, micro_fn)

    # Macro: average P/R/F1 across codes in universe (standard multi-label macro)
    macro_ps, macro_rs, macro_fs = [], [], []
    for code in code_universe:
        tp, fp, fn = per_code.get(code, (0, 0, 0))
        p, r, f = _prf(tp, fp, fn)
        macro_ps.append(p)
        macro_rs.append(r)
        macro_fs.append(f)
    macro_p = sum(macro_ps) / len(macro_ps) if macro_ps else 0.0
    macro_r = sum(macro_rs) / len(macro_rs) if macro_rs else 0.0
    macro_f = sum(macro_fs) / len(macro_fs) if macro_fs else 0.0

    exact_match = n_exact / n_notes if n_notes > 0 else 0.0
    jaccard_mean = jaccard_sum / n_notes if n_notes > 0 else 0.0
    example_f1 = example_f1_sum / n_notes if n_notes > 0 else 0.0

    result = SetMetrics(
        n_notes=n_notes,
        n_gold_codes=n_gold_codes,
        n_pred_codes=n_pred_codes,
        n_tp=micro_tp,
        n_fp=micro_fp,
        n_fn=micro_fn,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
        exact_match=exact_match,
        jaccard_mean=jaccard_mean,
        example_f1=example_f1,
        per_code={c: tuple(v) for c, v in per_code.items()},
    )

    # Per-category breakdown
    if compute_per_category:
        # Group per-note codes by category and recompute from the *same* buffered pairs
        categories = ("CPT_I", "CPT_II", "CPT_III", "PLA", "MAAA", "HCPCS")
        per_cat: Dict[str, SetMetrics] = {}
        for cat in categories:
            cat_pairs = [
                (
                    {c for c in gold if classify_code(c) == cat},
                    {c for c in pred if classify_code(c) == cat},
                )
                for gold, pred in buffered
            ]
            # Skip categories that never appear in gold (no meaningful macro)
            if not any(g for g, _ in cat_pairs):
                continue
            per_cat[cat] = compute_set_metrics(cat_pairs, compute_per_category=False)
        result.per_category = per_cat

    return result


# ---------------------------------------------------------------------------
# Note-level bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_set_metrics(
    pairs: Sequence[Tuple[Set[str], Set[str]]],
    n_boot: int = 1000,
    seed: int = 12345,
    ci: float = 0.95,
    include_macro: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap over *notes* for the headline set metrics.

    Resamples the corpus (notes with replacement) ``n_boot`` times and returns
    ``metric -> {mean, std, ci_lo, ci_hi}`` for micro P/R/F1, example-F1,
    exact-match, mean Jaccard, and (optionally) macro-F1. Notes — not codes —
    are the exchangeable unit, matching the per-note prediction task.

    The macro universe is recomputed per resample (codes with gold present in
    that resample), mirroring :func:`compute_set_metrics`' default.
    """
    import numpy as np

    pairs = [(set(g), set(p)) for g, p in pairs]
    n_notes = len(pairs)
    if n_notes == 0 or n_boot <= 0:
        return {}

    # ---- per-note scalars (micro + per-example metrics) ----
    tp = np.zeros(n_notes)
    fp = np.zeros(n_notes)
    fn = np.zeros(n_notes)
    jac = np.zeros(n_notes)
    exact = np.zeros(n_notes)
    exf1 = np.zeros(n_notes)
    # ---- per-(note, code) events for the macro metric ----
    code_ids: Dict[str, int] = {}
    ev_note: List[int] = []
    ev_code: List[int] = []
    ev_type: List[int] = []          # 0=tp, 1=fp, 2=fn
    for i, (gold, pred) in enumerate(pairs):
        tps, fps, fns = gold & pred, pred - gold, gold - pred
        tp[i], fp[i], fn[i] = len(tps), len(fps), len(fns)
        jac[i] = _jaccard(gold, pred)
        exact[i] = float(gold == pred)
        exf1[i] = _example_f1(gold, pred)
        if include_macro:
            for t, codes in enumerate((tps, fps, fns)):
                for c in codes:
                    ev_note.append(i)
                    ev_code.append(code_ids.setdefault(c, len(code_ids)))
                    ev_type.append(t)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_notes, size=(n_boot, n_notes))

    samples: Dict[str, List[float]] = defaultdict(list)
    if include_macro:
        ev_note_a = np.asarray(ev_note)
        ev_code_a = np.asarray(ev_code)
        ev_type_a = np.asarray(ev_type)
        n_codes = len(code_ids)

    for b in range(n_boot):
        sel = idx[b]
        s_tp, s_fp, s_fn = tp[sel].sum(), fp[sel].sum(), fn[sel].sum()
        p, r, f = _prf(int(s_tp), int(s_fp), int(s_fn))
        samples["micro_precision"].append(p)
        samples["micro_recall"].append(r)
        samples["micro_f1"].append(f)
        samples["example_f1"].append(float(exf1[sel].mean()))
        samples["exact_match"].append(float(exact[sel].mean()))
        samples["jaccard_mean"].append(float(jac[sel].mean()))
        if include_macro and n_codes:
            mult = np.bincount(sel, minlength=n_notes).astype(float)
            w = mult[ev_note_a]
            c_tp = np.bincount(ev_code_a[ev_type_a == 0],
                               weights=w[ev_type_a == 0], minlength=n_codes)
            c_fp = np.bincount(ev_code_a[ev_type_a == 1],
                               weights=w[ev_type_a == 1], minlength=n_codes)
            c_fn = np.bincount(ev_code_a[ev_type_a == 2],
                               weights=w[ev_type_a == 2], minlength=n_codes)
            universe = (c_tp + c_fn) > 0
            denom = 2 * c_tp + c_fp + c_fn
            f1_c = np.divide(2 * c_tp, denom,
                             out=np.zeros_like(denom), where=denom > 0)
            samples["macro_f1"].append(
                float(f1_c[universe].mean()) if universe.any() else 0.0
            )

    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out: Dict[str, Dict[str, float]] = {}
    for name, vals in samples.items():
        arr = np.asarray(vals)
        out[name] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)), 6),
            "ci_lo": round(float(np.percentile(arr, lo_q)), 6),
            "ci_hi": round(float(np.percentile(arr, hi_q)), 6),
        }
    out["_meta"] = {"n_boot": float(n_boot), "seed": float(seed), "ci": float(ci)}
    return out
