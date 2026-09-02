# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Threshold / top-k sweep over a saved verifier score NPZ (CPU only).

The verifier ``predict`` step decodes a *fixed* decision (cfg ``best_threshold``
or ``--top-k``) when it writes ``predictions.csv``.  When a new candidate
source (e.g. the LLM-concept pool) enlarges the pool, that fixed threshold lets
more false positives through, so downstream micro-F1 can stay flat even though
pool *recall* rose.  This tool reads the raw-score NPZ
(``note_ids / candidate_codes / probs`` — see :mod:`cpt_rec.pipeline.crossencoder.predict_verifier`)
and sweeps the *decision* without re-running the model, so we can tell whether
the flat F1 is a **calibration** artifact (a better operating point recovers F1)
or a **scorer** limit (the new candidates are genuinely unseparable → re-train).

It reuses the production metric (:func:`evaluation.metrics.compute_set_metrics`)
and the same gold loader / code normalization as the harness, so the numbers are
directly comparable to ``metrics.json``.

Example::

    python3 -m cpt_rec.common.evaluation.threshold_sweep \\
        --npz outputs/verifier/sections192_baseline/predictions/val__default_llm_concept_v2/predictions.npz \\
        --gold outputs/datasets/vumc/val_eval_sectioned.csv \\
        --ref-threshold 0.5 \\
        --train-stats outputs/datasets/vumc/code_frequency_stats.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from cpt_rec.common.evaluation.harness import load_gold
from cpt_rec.common.evaluation.metrics import compute_set_metrics

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def load_score_npz(
    npz_path: Path | str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load ``{note_id: (codes[str], probs[float])}`` from a predict NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    note_ids = list(data["note_ids"])
    cand_codes = list(data["candidate_codes"])
    probs = list(data["probs"])
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for nid, codes, ps in zip(note_ids, cand_codes, probs):
        c = np.array([str(x).strip().upper() for x in codes], dtype=object)
        p = np.asarray(ps, dtype=float)
        out[str(nid).strip()] = (c, p)
    LOGGER.info("Loaded scores for %d notes from %s", len(out), npz_path)
    return out


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_threshold(
    codes: np.ndarray, probs: np.ndarray, threshold: float
) -> Set[str]:
    if len(probs) == 0:
        return set()
    return set(codes[probs >= threshold].tolist())


def decode_top_k(codes: np.ndarray, probs: np.ndarray, k: int) -> Set[str]:
    if len(probs) == 0:
        return set()
    keep = np.argsort(-probs)[:k]
    return set(codes[keep].tolist())


def _pred_pairs(
    gold: Dict[str, Set[str]],
    scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    decode,
) -> List[Tuple[Set[str], Set[str]]]:
    """Build (gold_set, pred_set) pairs aligned on gold note_ids."""
    pairs: List[Tuple[Set[str], Set[str]]] = []
    for nid, gset in gold.items():
        entry = scores.get(nid)
        pset = decode(*entry) if entry is not None else set()
        pairs.append((gset, pset))
    return pairs


# ---------------------------------------------------------------------------
# Per-bin micro-F1 at a fixed decision
# ---------------------------------------------------------------------------

def per_bin_micro_f1(
    pairs: Sequence[Tuple[Set[str], Set[str]]],
    code_bin: Dict[str, str],
    bins: Sequence[str],
    unseen_bin: str = "unseen",
) -> Dict[str, Dict[str, float]]:
    """Micro P/R/F1 per frequency bin, binning each code by its own bin
    (matches the harness per-bin semantics)."""
    tally = {b: [0, 0, 0] for b in bins}  # [tp, fp, fn]

    def _bin(code: str) -> str:
        return code_bin.get(code.strip().upper(), unseen_bin)

    for gset, pset in pairs:
        for c in gset & pset:
            tally[_bin(c)][0] += 1
        for c in pset - gset:
            tally[_bin(c)][1] += 1
        for c in gset - pset:
            tally[_bin(c)][2] += 1

    out: Dict[str, Dict[str, float]] = {}
    for b, (tp, fp, fn) in tally.items():
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[b] = {
            "micro_precision": round(prec, 6),
            "micro_recall": round(rec, 6),
            "micro_f1": round(f1, 6),
            "n_tp": tp, "n_fp": fp, "n_fn": fn,
        }
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    gold: Dict[str, Set[str]],
    scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    grid: Sequence[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for t in grid:
        pairs = _pred_pairs(gold, scores, lambda c, p: decode_threshold(c, p, t))
        m = compute_set_metrics(pairs, compute_per_category=False)
        rows.append({
            "threshold": round(float(t), 4),
            "micro_f1": round(m.micro_f1, 6),
            "micro_precision": round(m.micro_precision, 6),
            "micro_recall": round(m.micro_recall, 6),
            "n_pred_codes": m.n_pred_codes,
        })
    return rows


def sweep_top_k(
    gold: Dict[str, Set[str]],
    scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ks: Sequence[int],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for k in ks:
        pairs = _pred_pairs(gold, scores, lambda c, p: decode_top_k(c, p, k))
        m = compute_set_metrics(pairs, compute_per_category=False)
        rows.append({
            "top_k": int(k),
            "micro_f1": round(m.micro_f1, 6),
            "micro_precision": round(m.micro_precision, 6),
            "micro_recall": round(m.micro_recall, 6),
            "n_pred_codes": m.n_pred_codes,
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep threshold/top-k over a verifier score NPZ (CPU)."
    )
    p.add_argument("--npz", required=True, type=Path,
                   help="predict_verifier --save-npz output (note_ids/candidate_codes/probs).")
    p.add_argument("--gold", required=True, type=Path,
                   help="Gold CSV (note_id + proc_codes), e.g. val_eval_sectioned.csv.")
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--t-min", type=float, default=0.05)
    p.add_argument("--t-max", type=float, default=0.95)
    p.add_argument("--t-steps", type=int, default=19,
                   help="Number of threshold grid points in [t-min, t-max].")
    p.add_argument("--ref-threshold", type=float, default=None,
                   help="Also report F1 at this threshold (e.g. the cfg best_threshold).")
    p.add_argument("--top-k", nargs="*", type=int, default=None,
                   help="Optional per-note top-k budgets to sweep as well.")
    p.add_argument("--train-stats", type=Path, default=None,
                   help="code_frequency_stats.csv → per-bin micro-F1 at the best threshold.")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    gold = load_gold(args.gold, code_col=args.gold_code_col)
    scores = load_score_npz(args.npz)

    # Evenly spaced sweep grid; splice in the production reference threshold
    # (e.g. the cfg ``best_threshold``) so its row is computed on the exact same
    # gold/scores and is line-comparable to the swept operating points.
    grid = list(np.round(np.linspace(args.t_min, args.t_max, args.t_steps), 4))
    if args.ref_threshold is not None and args.ref_threshold not in grid:
        grid = sorted(set(grid) | {round(float(args.ref_threshold), 4)})

    rows = sweep_thresholds(gold, scores, grid)
    best = max(rows, key=lambda r: r["micro_f1"])

    print("\nThreshold sweep (micro):")
    print(f"  {'thr':>6}  {'micro-F1':>9}  {'P':>7}  {'R':>7}  {'n_pred':>7}")
    for r in rows:
        flag = "  <- best" if r is best else ""
        ref = "  (ref)" if (args.ref_threshold is not None
                            and r["threshold"] == round(float(args.ref_threshold), 4)) else ""
        print(f"  {r['threshold']:6.3f}  {r['micro_f1']:9.4f}  "
              f"{r['micro_precision']:7.4f}  {r['micro_recall']:7.4f}  "
              f"{r['n_pred_codes']:7d}{flag}{ref}")
    print(f"\nBEST global threshold = {best['threshold']:.3f}  "
          f"micro-F1 = {best['micro_f1']:.4f}  "
          f"(P {best['micro_precision']:.4f} / R {best['micro_recall']:.4f})")

    topk_rows: List[Dict[str, float]] = []
    if args.top_k:
        topk_rows = sweep_top_k(gold, scores, args.top_k)
        best_k = max(topk_rows, key=lambda r: r["micro_f1"])
        print("\nTop-k sweep (micro):")
        for r in topk_rows:
            flag = "  <- best" if r is best_k else ""
            print(f"  k={r['top_k']:<4d}  micro-F1={r['micro_f1']:.4f}  "
                  f"P={r['micro_precision']:.4f}  R={r['micro_recall']:.4f}  "
                  f"n_pred={r['n_pred_codes']}{flag}")

    per_bin: Dict[str, Dict[str, float]] = {}
    if args.train_stats is not None:
        from cpt_rec.common.evaluation.code_bins import BINS, load_code_bins
        code_bin = load_code_bins(str(args.train_stats))
        pairs = _pred_pairs(
            gold, scores, lambda c, p: decode_threshold(c, p, best["threshold"])
        )
        per_bin = per_bin_micro_f1(pairs, code_bin, BINS)
        print(f"\nPer-bin micro-F1 at best threshold {best['threshold']:.3f}:")
        for b in BINS:
            if b in per_bin:
                m = per_bin[b]
                print(f"  {b:7s}  F1={m['micro_f1']:.4f}  "
                      f"P={m['micro_precision']:.4f}  R={m['micro_recall']:.4f}  "
                      f"(tp {m['n_tp']} / fp {m['n_fp']} / fn {m['n_fn']})")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "inputs": {"npz": str(args.npz), "gold": str(args.gold)},
            "threshold_sweep": rows,
            "best_threshold": best,
            "top_k_sweep": topk_rows,
            "per_bin_at_best": per_bin,
        }
        args.out_json.write_text(json.dumps(payload, indent=2))
        LOGGER.info("Wrote sweep results -> %s", args.out_json)


if __name__ == "__main__":
    main()
