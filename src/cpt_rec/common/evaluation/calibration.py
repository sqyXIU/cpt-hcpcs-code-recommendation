# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Per-code probability-threshold calibration on a validation split.

A probabilistic multi-label classifier produces (note, code, score) triples.
For set-valued metrics we need per-code thresholds.  Picking a single
corpus-wide threshold is fine as a baseline, but **per-code** thresholds
typically recover several F1 points on the long tail.

This module:
1. Sweeps candidate thresholds per code on a val-set score file.
2. Picks the threshold maximizing F1 for that code.
3. Falls back to a global prior for codes with too few positives.
4. Also reports Average Precision (AP) per code and macro-AP.

Output schema (JSON):

.. code-block:: json

    {
      "global_threshold": 0.35,
      "per_code": {"45378": 0.41, "43235": 0.29, ...},
      "per_code_ap": {"45378": 0.82, ...},
      "macro_ap": 0.71,
      "n_codes_tuned": 4211,
      "n_codes_fallback": 1063
    }
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    global_threshold: float
    per_code: Dict[str, float]
    per_code_ap: Dict[str, float]
    macro_ap: float
    n_codes_tuned: int
    n_codes_fallback: int

    def to_dict(self) -> dict:
        return {
            "global_threshold": round(self.global_threshold, 6),
            "per_code": {k: round(v, 6) for k, v in self.per_code.items()},
            "per_code_ap": {k: round(v, 6) for k, v in self.per_code_ap.items()},
            "macro_ap": round(self.macro_ap, 6),
            "n_codes_tuned": self.n_codes_tuned,
            "n_codes_fallback": self.n_codes_fallback,
        }

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationResult":
        with open(path) as f:
            d = json.load(f)
        return cls(
            global_threshold=d["global_threshold"],
            per_code=d["per_code"],
            per_code_ap=d.get("per_code_ap", {}),
            macro_ap=d.get("macro_ap", 0.0),
            n_codes_tuned=d.get("n_codes_tuned", 0),
            n_codes_fallback=d.get("n_codes_fallback", 0),
        )

    def threshold_for(self, code: str) -> float:
        """Return the per-code threshold, or the global fallback."""
        return self.per_code.get(code, self.global_threshold)


# ---------------------------------------------------------------------------
# Per-code tuning
# ---------------------------------------------------------------------------

def _best_threshold_for_code(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Pick the threshold in ``(0, 1)`` maximizing F1 for a single code.

    Parameters
    ----------
    scores : (N,) float
    labels : (N,) int in {0, 1}
    n_thresholds : int

    Returns
    -------
    (best_threshold, best_f1)
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5, 0.0

    # Candidate thresholds: include the actual score values (the F1-optimal
    # threshold is always at a score boundary) plus a coarse grid.
    uniq_scores = np.unique(scores)
    if len(uniq_scores) > n_thresholds:
        idx = np.linspace(0, len(uniq_scores) - 1, n_thresholds).astype(int)
        candidates = uniq_scores[idx]
    else:
        candidates = uniq_scores
    candidates = np.concatenate([candidates, [0.5]])  # always consider 0.5

    best_f1 = -1.0
    best_t = 0.5
    for t in candidates:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        denom_p = tp + fp
        denom_r = tp + fn
        if denom_p == 0 or denom_r == 0:
            continue
        p = tp / denom_p
        r = tp / denom_r
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t, max(best_f1, 0.0)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary Average Precision (area under PR curve)."""
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    cum_tp = np.cumsum(labels_sorted)
    precisions = cum_tp / (np.arange(len(labels_sorted)) + 1)
    recall_changes = labels_sorted  # 1 when rank step adds a TP
    total_positives = labels_sorted.sum()
    if total_positives == 0:
        return 0.0
    return float((precisions * recall_changes).sum() / total_positives)


def calibrate_thresholds(
    code_to_scores_labels: Dict[str, Tuple[np.ndarray, np.ndarray]],
    min_positives: int = 10,
    default_threshold: float = 0.5,
    n_thresholds: int = 100,
) -> CalibrationResult:
    """
    Fit per-code thresholds on validation data.

    Parameters
    ----------
    code_to_scores_labels
        ``{code: (scores, labels)}`` — per code, the aligned arrays of
        predicted scores and binary gold labels across the val corpus.
    min_positives
        Codes with fewer than this many positives in val fall back to the
        global threshold.
    default_threshold
        Used when a code has no signal at all.
    """
    per_code_thresholds: Dict[str, float] = {}
    per_code_ap: Dict[str, float] = {}
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    tuned = 0
    fallback = 0

    for code, (scores, labels) in code_to_scores_labels.items():
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        all_scores.append(scores)
        all_labels.append(labels)
        ap = _average_precision(scores, labels)
        per_code_ap[code] = ap

        if int(labels.sum()) < min_positives:
            fallback += 1
            continue
        t, _f1 = _best_threshold_for_code(scores, labels, n_thresholds=n_thresholds)
        per_code_thresholds[code] = t
        tuned += 1

    # Pick global threshold by tuning on concatenated labels (one-vs-rest pool)
    if all_scores:
        cat_scores = np.concatenate(all_scores)
        cat_labels = np.concatenate(all_labels)
        global_t, _ = _best_threshold_for_code(
            cat_scores, cat_labels, n_thresholds=n_thresholds
        )
    else:
        global_t = default_threshold

    macro_ap = float(np.mean(list(per_code_ap.values()))) if per_code_ap else 0.0

    return CalibrationResult(
        global_threshold=global_t,
        per_code=per_code_thresholds,
        per_code_ap=per_code_ap,
        macro_ap=macro_ap,
        n_codes_tuned=tuned,
        n_codes_fallback=fallback,
    )


def apply_thresholds(
    note_to_code_scores: Dict[str, Dict[str, float]],
    calibration: CalibrationResult,
) -> Dict[str, set]:
    """
    Apply per-code thresholds to score dicts, returning per-note predicted sets.

    Parameters
    ----------
    note_to_code_scores : {note_id: {code: score}}
    """
    out: Dict[str, set] = {}
    for nid, code_scores in note_to_code_scores.items():
        pred = {
            code
            for code, score in code_scores.items()
            if score >= calibration.threshold_for(code)
        }
        out[nid] = pred
    return out
