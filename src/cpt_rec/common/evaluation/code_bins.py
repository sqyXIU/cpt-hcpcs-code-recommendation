# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Drift-aware and frequency-bin evaluation for CPT/HCPCS predictions.

Reads:
- **training code-frequency stats** (``code_frequency_stats.csv``) for
  head / torso / tail binning,
- a **drift annotation CSV** (``*_drift.csv``) for seen / unseen labels,
- **predictions** and **gold** per note.

Produces a slice report: macro-F1 broken down by
``train_bin × {seen, unseen}`` — the flagship table of the paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from cpt_rec.common.evaluation.metrics import _prf

LOGGER = logging.getLogger(__name__)

BINS = ("head", "torso", "tail", "unseen")


@dataclass
class BinMetrics:
    """Metrics aggregated over a bin of codes."""

    bin_label: str
    n_codes: int
    n_tp: int
    n_fp: int
    n_fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def to_dict(self) -> dict:
        return {
            "bin": self.bin_label,
            "n_codes": self.n_codes,
            "n_tp": self.n_tp,
            "n_fp": self.n_fp,
            "n_fn": self.n_fn,
            "micro_precision": round(self.micro_precision, 6),
            "micro_recall": round(self.micro_recall, 6),
            "micro_f1": round(self.micro_f1, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "macro_f1": round(self.macro_f1, 6),
        }


def load_code_bins(train_stats_csv: str) -> Dict[str, str]:
    """Load ``code -> bin`` mapping from the training stats CSV."""
    df = pd.read_csv(train_stats_csv, dtype={"code": str})
    return dict(
        zip(df["code"].str.strip().str.upper(), df["bin"].str.strip().str.lower())
    )


def compute_bin_metrics(
    pairs: Iterable[Tuple[Set[str], Set[str]]],
    code_bin: Dict[str, str],
    unseen_bin: str = "unseen",
) -> Dict[str, BinMetrics]:
    """
    Compute per-bin micro/macro metrics.

    A code is assigned to a bin using ``code_bin``; unknown codes (i.e.
    unseen in training) are assigned to ``unseen_bin``.

    Parameters
    ----------
    pairs
        Iterable of ``(gold_set, pred_set)`` per note.
    code_bin
        Mapping from code -> 'head' / 'torso' / 'tail'.
    unseen_bin
        Bin label for codes not in ``code_bin``.  Default 'unseen'.

    Returns
    -------
    dict bin_label -> BinMetrics
    """
    # Per-code counts, then group by bin for micro, plus keep per-code for macro.
    per_code_counts: Dict[str, List[int]] = {}  # code -> [tp, fp, fn]

    def _bin_of(code: str) -> str:
        return code_bin.get(code.strip().upper(), unseen_bin)

    for gold, pred in pairs:
        gold = set(gold)
        pred = set(pred)
        for c in gold & pred:
            per_code_counts.setdefault(c, [0, 0, 0])[0] += 1
        for c in pred - gold:
            per_code_counts.setdefault(c, [0, 0, 0])[1] += 1
        for c in gold - pred:
            per_code_counts.setdefault(c, [0, 0, 0])[2] += 1

    # Group by bin
    bin_to_codes: Dict[str, List[str]] = {b: [] for b in BINS}
    for code in per_code_counts:
        b = _bin_of(code)
        bin_to_codes.setdefault(b, []).append(code)

    results: Dict[str, BinMetrics] = {}
    for b, codes in bin_to_codes.items():
        if not codes:
            continue
        tp = sum(per_code_counts[c][0] for c in codes)
        fp = sum(per_code_counts[c][1] for c in codes)
        fn = sum(per_code_counts[c][2] for c in codes)
        micro_p, micro_r, micro_f = _prf(tp, fp, fn)

        ps, rs, fs = [], [], []
        for c in codes:
            tpc, fpc, fnc = per_code_counts[c]
            # Only include codes that appear in gold (tp+fn > 0) for macro
            if (tpc + fnc) == 0:
                continue
            p, r, f = _prf(tpc, fpc, fnc)
            ps.append(p)
            rs.append(r)
            fs.append(f)
        macro_p = sum(ps) / len(ps) if ps else 0.0
        macro_r = sum(rs) / len(rs) if rs else 0.0
        macro_f = sum(fs) / len(fs) if fs else 0.0

        results[b] = BinMetrics(
            bin_label=b,
            n_codes=len(codes),
            n_tp=tp,
            n_fp=fp,
            n_fn=fn,
            micro_precision=micro_p,
            micro_recall=micro_r,
            micro_f1=micro_f,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f,
        )
    return results


def bin_metrics_table(bin_metrics: Dict[str, BinMetrics]) -> pd.DataFrame:
    """Flatten per-bin metrics into a DataFrame (one row per bin) in a stable order."""
    order = [b for b in BINS if b in bin_metrics]
    rows = [bin_metrics[b].to_dict() for b in order]
    return pd.DataFrame(rows)
