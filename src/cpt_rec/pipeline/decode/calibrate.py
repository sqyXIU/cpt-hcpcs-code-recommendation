#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Threshold calibration for the candidate scorer.

Reads a ``--val-npz`` (produced by ``cptrec-verifier-predict`` with
``--out-npz``) and a ``--val-csv`` (the same notes file used at predict
time, with the gold ``proc_codes`` column).  Sweeps a sigmoid threshold
grid and writes a calibration JSON of the form

::

    {
      "default_threshold": 0.18,
      "default_metrics": {"micro_p": ..., "micro_r": ..., "micro_f1": ...},
      "per_code": { "12345": 0.22, "20680": 0.10, ... },
      "search_grid": [0.05, 0.06, ..., 0.95],
      "objective": "micro_f1"
    }

The JSON plugs straight into ``cptrec-verifier-predict --calibration``.

Per-code thresholds are tuned **only** for codes with at least
``--min-pos-per-code`` positive val examples; codes below that are left
to the default threshold.  This avoids over-fitting to noisy thresholds
on rare codes that happen to land at p≈1.0 on a single val example.

CLI
---

::

    cptrec-calibrate \\
        --val-csv outputs/datasets/vumc/val_eval_sectioned.csv \\
        --val-npz outputs/verifier/wholenote320_control/val_predictions.npz \\
        --out outputs/verifier/wholenote320_control/calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cpt_rec.common.preprocess.code_utils import (
    pipe_split_codes as _pipe_split,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_val_gold(
    val_csv: Path,
    gold_code_col: str = "proc_codes",
    note_id_col: Optional[str] = None,
) -> Dict[str, set]:
    """Return ``{note_id: {gold_code, ...}}`` from a notes CSV."""
    from cpt_rec.baselines.common import (
        _ID_CANDIDATES,
        _pick_col,
    )

    df = pd.read_csv(val_csv, dtype=str)
    id_col = note_id_col or _pick_col(df, _ID_CANDIDATES, "note-id")
    if gold_code_col not in df.columns:
        raise ValueError(f"val CSV missing gold column {gold_code_col!r}")
    return {
        str(nid): set(_pipe_split(codes))
        for nid, codes in zip(df[id_col], df[gold_code_col])
    }


def load_val_npz(path: Path):
    """Return ``(note_ids, candidate_codes, probs)`` ragged arrays."""
    z = np.load(path, allow_pickle=True)
    return z["note_ids"], z["candidate_codes"], z["probs"]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _global_micro_f1(
    note_ids,
    cands,
    probs,
    gold: Dict[str, set],
    threshold: float,
) -> Tuple[float, float, float, int, int, int]:
    tp = fp = fn = 0
    for i, nid in enumerate(note_ids):
        nid_s = str(nid)
        keep = [c for c, p in zip(cands[i], probs[i]) if p >= threshold]
        preds = set(keep)
        g = gold.get(nid_s, set())
        tp += len(preds & g)
        fp += len(preds - g)
        fn += len(g - preds)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, p, r, tp, fp, fn


def sweep_default_threshold(
    note_ids,
    cands,
    probs,
    gold: Dict[str, set],
    grid: List[float],
) -> Tuple[float, Dict[str, float]]:
    best_f1 = -1.0
    best_thr = grid[len(grid) // 2]
    best_stats: Dict[str, float] = {}
    for thr in grid:
        f1, p, r, tp, fp, fn = _global_micro_f1(
            note_ids, cands, probs, gold, thr
        )
        LOGGER.debug("thr=%.3f F1=%.4f P=%.4f R=%.4f", thr, f1, p, r)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_stats = {
                "threshold": thr,
                "micro_f1": f1,
                "micro_p": p,
                "micro_r": r,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
    return best_thr, best_stats


def sweep_per_code_thresholds(
    note_ids,
    cands,
    probs,
    gold: Dict[str, set],
    grid: List[float],
    min_pos_per_code: int,
    default_thr: float,
) -> Dict[str, float]:
    """For each code with >= ``min_pos_per_code`` val positives, pick the
    threshold that maximizes that code's F1 (ties → lower threshold).

    Codes below the floor inherit the global default and are NOT recorded
    in the returned dict (the predict path falls back automatically)."""
    # Build per-code (probs, label) arrays.
    per_code_probs: Dict[str, List[float]] = {}
    per_code_labels: Dict[str, List[int]] = {}
    for i, nid in enumerate(note_ids):
        nid_s = str(nid)
        g = gold.get(nid_s, set())
        for c, p in zip(cands[i], probs[i]):
            cu = str(c).upper()
            per_code_probs.setdefault(cu, []).append(float(p))
            per_code_labels.setdefault(cu, []).append(1 if cu in g else 0)

    out: Dict[str, float] = {}
    n_tuned = 0
    n_skipped_lowpos = 0
    for code, probs_list in per_code_probs.items():
        labels = np.array(per_code_labels[code], dtype=np.int32)
        n_pos = int(labels.sum())
        if n_pos < min_pos_per_code:
            n_skipped_lowpos += 1
            continue
        probs_arr = np.array(probs_list, dtype=np.float32)

        best_f1 = -1.0
        best_thr = default_thr
        for thr in grid:
            preds = (probs_arr >= thr).astype(np.int32)
            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        # Only record a per-code override if it beats the default by a
        # margin (else the default carries it).  Tiny per-code wins are
        # noise and bloat the JSON.
        default_preds = (probs_arr >= default_thr).astype(np.int32)
        tp = int(((default_preds == 1) & (labels == 1)).sum())
        fp = int(((default_preds == 1) & (labels == 0)).sum())
        fn = int(((default_preds == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        default_f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if best_f1 > default_f1 + 1e-3:
            out[code] = float(best_thr)
            n_tuned += 1

    LOGGER.info(
        "Per-code: tuned=%d, kept-default=%d, skipped(low-pos<%d)=%d",
        n_tuned, len(per_code_probs) - n_tuned - n_skipped_lowpos,
        min_pos_per_code, n_skipped_lowpos,
    )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def calibrate(
    val_csv: Path,
    val_npz: Path,
    out_json: Path,
    gold_code_col: str = "proc_codes",
    note_id_col: Optional[str] = None,
    grid_min: float = 0.05,
    grid_max: float = 0.95,
    grid_step: float = 0.01,
    per_code: bool = True,
    min_pos_per_code: int = 5,
) -> Dict:
    note_ids, cands, probs = load_val_npz(val_npz)
    gold = load_val_gold(val_csv, gold_code_col, note_id_col)

    LOGGER.info(
        "Loaded NPZ: %d notes, gold map size=%d", len(note_ids), len(gold),
    )

    grid = list(np.round(np.arange(grid_min, grid_max + 1e-9, grid_step), 4))

    default_thr, default_metrics = sweep_default_threshold(
        note_ids, cands, probs, gold, grid
    )
    LOGGER.info(
        "Best default threshold=%.3f micro_f1=%.4f (P=%.4f R=%.4f)",
        default_thr, default_metrics["micro_f1"],
        default_metrics["micro_p"], default_metrics["micro_r"],
    )

    per_code_map: Dict[str, float] = {}
    if per_code:
        per_code_map = sweep_per_code_thresholds(
            note_ids, cands, probs, gold, grid,
            min_pos_per_code=min_pos_per_code,
            default_thr=default_thr,
        )

    payload = {
        "default_threshold": float(default_thr),
        "default_metrics": default_metrics,
        "per_code": per_code_map,
        "search_grid": [float(x) for x in grid],
        "objective": "micro_f1",
        "min_pos_per_code": int(min_pos_per_code),
        "n_val_notes": int(len(note_ids)),
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    LOGGER.info(
        "Wrote calibration -> %s (default=%.3f, per_code overrides=%d)",
        out_json, default_thr, len(per_code_map),
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep sigmoid thresholds on val to maximize micro-F1."
    )
    p.add_argument("--val-csv", required=True, type=Path)
    p.add_argument("--val-npz", required=True, type=Path)
    p.add_argument("--out", dest="out_json", required=True, type=Path)
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--grid-min", type=float, default=0.05)
    p.add_argument("--grid-max", type=float, default=0.95)
    p.add_argument("--grid-step", type=float, default=0.01)
    p.add_argument("--no-per-code", dest="per_code", action="store_false",
                   help="Disable per-code threshold tuning (only emit "
                        "default_threshold).")
    p.set_defaults(per_code=True)
    p.add_argument("--min-pos-per-code", type=int, default=5,
                   help="Minimum positive val occurrences before a code "
                        "is eligible for per-code threshold tuning.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    calibrate(
        val_csv=args.val_csv,
        val_npz=args.val_npz,
        out_json=args.out_json,
        gold_code_col=args.gold_code_col,
        note_id_col=args.note_id_col,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_step=args.grid_step,
        per_code=args.per_code,
        min_pos_per_code=args.min_pos_per_code,
    )


if __name__ == "__main__":
    main()
