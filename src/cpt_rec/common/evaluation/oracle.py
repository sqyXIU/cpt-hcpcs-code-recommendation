#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Oracle decomposition for the M6 pipeline (no GPU, no model re-run).

Reads a ``predictions.npz`` (written by ``cptrec-verifier-predict --save-npz``,
columns ``note_ids`` / ``candidate_codes`` / ``probs``) and the matching
gold ``*_sectioned.csv`` and answers one question:

    *Is the bottleneck candidate retrieval, scoring, calibration, or
    decoding?*

It breaks the gap between current micro-F1 and a perfect system into four
ceilings, each computable offline from files already on disk:

1. **Candidate recall @K** — fraction of gold codes that even enter the
   pool the scorer sees.  This is the hard ceiling: no scorer can recover
   a code that was never retrieved.
2. **Scoring oracle (top-|gold|)** — if we kept exactly the right *number*
   of codes per note, ranked by the model's own probability, what micro-F1
   do we get?  Isolates *ranking* quality from *thresholding*.
3. **Calibration headroom** — best achievable micro-F1 under a single
   global threshold on this split (oracle threshold) vs. the F1 at the
   threshold actually applied.  With ``--val-npz``/``--val-gold`` it also
   reports the honest val->test calibration gap.
4. **Set-size error** — mean predicted vs. gold set size and the
   mean-absolute-error of cardinality, plus the pred/gold count ratio
   (>1 over-predicts, <1 under-predicts).

Decision gate
-------------

* CandRecall@100 low (<~0.85)            -> retrieval-bound  (upgrade the
                                            candidate pool: add neighbor /
                                            LLM-prior sources).
* CandRecall high but scoring-oracle F1
  far above realized F1                  -> scoring/calibration-bound
                                            (cross-encoder, ranking loss,
                                            set-size decoding).

Run
---

::

    python -m cpt_rec.common.evaluation.oracle \\
        --npz   outputs/verifier/wholenote320_control/predictions/test/predictions.npz \\
        --gold  outputs/datasets/vumc/test_eval_sectioned.csv \\
        --calibration outputs/verifier/wholenote320_control/calibration.json \\
        --val-npz  outputs/verifier/wholenote320_control/predictions/val/predictions.npz \\
        --val-gold outputs/datasets/vumc/val_eval_sectioned.csv \\
        --by-category \\
        --out-json outputs/evaluation/wholenote320_control/oracle.json

The candidate pool stored at predict time is capped by the assembler's
per-source top-k (``bm25_top_k`` / ``dense_top_k`` / ``neighbor_top_k``, 25
each by default), so CandRecall is only meaningful up to that depth.  To probe
deeper, raise those fields in :class:`~cpt_rec.pipeline.pool.candidate_pool.AssemblerConfig`,
re-run predict with ``--save-npz``, and point ``--npz`` at the new file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from cpt_rec.baselines.common import _ID_CANDIDATES, _pick_col
from cpt_rec.common.evaluation.code_bins import BINS, load_code_bins
from cpt_rec.common.evaluation.metrics import classify_code
from cpt_rec.common.preprocess.code_utils import pipe_split_codes

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _norm(code: str) -> str:
    return str(code).strip().upper()


def load_npz(path: Path):
    """Return ``(note_ids, candidate_codes, probs)`` as python lists.

    ``candidate_codes[i]`` is a list[str]; ``probs[i]`` a np.ndarray[float]
    aligned to it.  Codes are upper-cased so they line up with gold.
    """
    z = np.load(path, allow_pickle=True)
    note_ids = [str(n) for n in z["note_ids"]]
    cands = [[_norm(c) for c in row] for row in z["candidate_codes"]]
    probs = [np.asarray(p, dtype=np.float64) for p in z["probs"]]
    return note_ids, cands, probs


def load_gold(csv_path: Path, gold_col: str = "proc_codes") -> Dict[str, Set[str]]:
    df = pd.read_csv(csv_path, dtype=str)
    id_col = _pick_col(df, _ID_CANDIDATES, f"note-id (in {csv_path})")
    if gold_col not in df.columns:
        raise ValueError(f"Gold CSV {csv_path} missing '{gold_col}'.")
    return {
        str(nid).strip(): {_norm(c) for c in pipe_split_codes(codes)}
        for nid, codes in zip(df[id_col], df[gold_col])
    }


def load_calibration(path: Optional[Path]) -> Tuple[float, Dict[str, float]]:
    """Return ``(default_threshold, per_code_map)`` (per_code upper-cased)."""
    if path is None:
        return 0.5, {}
    with open(path) as f:
        cal = json.load(f)
    default_thr = float(cal.get("default_threshold", 0.5))
    per_code = {_norm(k): float(v) for k, v in cal.get("per_code", {}).items()}
    return default_thr, per_code


# ---------------------------------------------------------------------------
# Realized prediction set (mirrors the predictor's thresholding)
# ---------------------------------------------------------------------------

def realized_pred(
    cands: Sequence[str],
    probs: np.ndarray,
    default_thr: float,
    per_code: Dict[str, float],
) -> Set[str]:
    """The set predict.py would emit: keep code c iff p >= thr(c)."""
    out: Set[str] = set()
    for c, p in zip(cands, probs):
        if p >= per_code.get(c, default_thr):
            out.add(c)
    return out


# ---------------------------------------------------------------------------
# (1) Candidate recall @K
# ---------------------------------------------------------------------------

def candidate_recall(
    note_ids: Sequence[str],
    cands: Sequence[Sequence[str]],
    gold: Dict[str, Set[str]],
    ks: Sequence[int],
) -> Dict[int, float]:
    """micro CandRecall@K = sum |gold ∩ pool[:K]| / sum |gold|."""
    out: Dict[int, float] = {}
    for k in ks:
        hit = tot = 0
        for nid, pool in zip(note_ids, cands):
            g = gold.get(nid, set())
            if not g:
                continue
            topk = set(pool[:k])
            hit += len(g & topk)
            tot += len(g)
        out[k] = hit / tot if tot else 0.0
    return out


# ---------------------------------------------------------------------------
# (2) Scoring oracle — keep top-|gold| per note, ranked by prob
# ---------------------------------------------------------------------------

def scoring_oracle_f1(
    note_ids: Sequence[str],
    cands: Sequence[Sequence[str]],
    probs: Sequence[np.ndarray],
    gold: Dict[str, Set[str]],
) -> float:
    """micro-F1 if we kept exactly |gold_n| top-prob candidates per note.

    With #pred == #gold per note, micro P == R == F1, so this equals
    sum_n |topk_n ∩ gold_n| / sum_n |gold_n|.
    """
    hit = tot = 0
    for nid, pool, pr in zip(note_ids, cands, probs):
        g = gold.get(nid, set())
        k = len(g)
        tot += k
        if k == 0 or len(pool) == 0:
            continue
        order = np.argsort(-pr)[:k]
        topk = {pool[i] for i in order}
        hit += len(g & topk)
    return hit / tot if tot else 0.0


# ---------------------------------------------------------------------------
# (3) Threshold sweep / calibration headroom
# ---------------------------------------------------------------------------

def _micro_f1_at(
    note_ids, cands, probs, gold: Dict[str, Set[str]], thr: float
) -> Tuple[float, float, float, int, int, int]:
    tp = fp = fn = 0
    for nid, pool, pr in zip(note_ids, cands, probs):
        preds = {c for c, p in zip(pool, pr) if p >= thr}
        g = gold.get(nid, set())
        tp += len(preds & g)
        fp += len(preds - g)
        fn += len(g - preds)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, p, r, tp, fp, fn


def sweep_threshold(
    note_ids, cands, probs, gold, grid: Sequence[float]
) -> Tuple[float, float]:
    """Return ``(best_threshold, best_micro_f1)`` on this split."""
    best_thr, best_f1 = grid[len(grid) // 2], -1.0
    for thr in grid:
        f1 = _micro_f1_at(note_ids, cands, probs, gold, thr)[0]
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# ---------------------------------------------------------------------------
# (4) Set-size analysis
# ---------------------------------------------------------------------------

def set_size_stats(
    note_ids, cands, probs, gold, default_thr, per_code
) -> Dict[str, float]:
    """Cardinality diagnosis of the *realized* prediction sets.

    Compares how many codes the model emits per note (under its real
    calibration) against the gold count, to separate a *cardinality* bias from
    a *ranking* error.  Returns the per-note ``mean_pred_size`` /
    ``mean_gold_size``, the mean-absolute cardinality error ``set_size_mae``,
    and ``pred_gold_ratio`` = total predicted / total gold (``>1`` the decoder
    systematically over-predicts, ``<1`` it under-predicts).
    """
    n_pred = n_gold = 0
    abs_err = 0.0
    n = 0
    for nid, pool, pr in zip(note_ids, cands, probs):
        g = gold.get(nid, set())
        preds = realized_pred(pool, pr, default_thr, per_code)
        n_pred += len(preds)
        n_gold += len(g)
        abs_err += abs(len(preds) - len(g))
        n += 1
    return {
        "mean_pred_size": n_pred / n if n else 0.0,
        "mean_gold_size": n_gold / n if n else 0.0,
        "set_size_mae": abs_err / n if n else 0.0,
        "pred_gold_ratio": n_pred / n_gold if n_gold else 0.0,
    }


# ---------------------------------------------------------------------------
# (5) Per-strata decomposition  (head / torso / tail / unseen)
# ---------------------------------------------------------------------------

def per_bin_gold_recall(
    note_ids: Sequence[str],
    selected_by_note: Dict[str, Set[str]],
    gold: Dict[str, Set[str]],
    code_bin: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Per train-frequency bin micro-recall of gold given a per-note selection.

    For each bin ``b`` in ``head/torso/tail/unseen``: over every gold-code
    occurrence whose training bin is ``b``, the fraction that appear in that
    note's ``selected_by_note`` set.  Codes absent from ``code_bin`` count as
    ``unseen``.  Uses the *same* per-code binning as the eval harness
    (:func:`code_bins.compute_bin_metrics`), so the ``pool_recall_ceiling``
    here lines up bin-for-bin with ``per_bin_metrics.csv``.

    Returns ``{bin: {"n_gold": int, "recall": float}}`` for all four bins.
    """
    hit = {b: 0 for b in BINS}
    tot = {b: 0 for b in BINS}
    for nid in note_ids:
        g = gold.get(nid, set())
        sel = selected_by_note.get(nid, set())
        for c in g:
            b = code_bin.get(_norm(c), "unseen")
            if b not in hit:
                b = "unseen"
            tot[b] += 1
            if c in sel:
                hit[b] += 1
    return {
        b: {"n_gold": tot[b], "recall": (hit[b] / tot[b] if tot[b] else 0.0)}
        for b in BINS
    }


def strata_breakdown(
    note_ids: Sequence[str],
    cands: Sequence[Sequence[str]],
    probs: Sequence[np.ndarray],
    gold: Dict[str, Set[str]],
    code_bin: Dict[str, str],
    default_thr: float,
    per_code: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Split the retrieval-vs-scorer diagnosis by train-frequency bin.

    For each bin reports three recalls over that bin's gold codes:

    * ``pool_recall_ceiling`` — gold in the candidate pool at all (the hard
      retrieval ceiling for the bin; no scorer can beat it).
    * ``scoring_oracle_recall`` — gold kept when we select exactly ``|gold_n|``
      top-probability candidates per note (ranking quality, threshold removed).
    * ``realized_recall`` — gold kept by the set predict.py actually emits
      (``p >= thr(c)`` with the applied calibration).

    Read: ``pool_recall_ceiling`` low  -> that bin is retrieval-bound (fix the
    pool / add LLM-prior + neighbour sources); ``pool_recall_ceiling`` high but
    ``scoring_oracle_recall`` / ``realized_recall`` far below it -> that bin is
    scorer/threshold-bound.  This is the per-bin read that decides whether
    Stage-3 (recall) or Stage-4 (scorer) owns the long tail.
    """
    pool_sel: Dict[str, Set[str]] = {}
    oracle_sel: Dict[str, Set[str]] = {}
    realized_sel: Dict[str, Set[str]] = {}
    for nid, pool, pr in zip(note_ids, cands, probs):
        pool_sel[nid] = set(pool)
        realized_sel[nid] = realized_pred(pool, pr, default_thr, per_code)
        g = gold.get(nid, set())
        k = len(g)
        if k and len(pool):
            order = np.argsort(-pr)[:k]
            oracle_sel[nid] = {pool[i] for i in order}
        else:
            oracle_sel[nid] = set()

    pool_r = per_bin_gold_recall(note_ids, pool_sel, gold, code_bin)
    oracle_r = per_bin_gold_recall(note_ids, oracle_sel, gold, code_bin)
    realized_r = per_bin_gold_recall(note_ids, realized_sel, gold, code_bin)

    out: Dict[str, Dict[str, float]] = {}
    for b in BINS:
        out[b] = {
            "n_gold": pool_r[b]["n_gold"],
            "pool_recall_ceiling": round(pool_r[b]["recall"], 4),
            "scoring_oracle_recall": round(oracle_r[b]["recall"], 4),
            "realized_recall": round(realized_r[b]["recall"], 4),
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    npz: Path,
    gold_csv: Path,
    ks: Sequence[int],
    calibration: Optional[Path],
    val_npz: Optional[Path],
    val_gold: Optional[Path],
    by_category: bool,
    train_stats: Optional[Path] = None,
    by_strata: bool = False,
    grid_step: float = 0.01,
) -> Dict:
    note_ids, cands, probs = load_npz(npz)
    gold = load_gold(gold_csv)
    default_thr, per_code = load_calibration(calibration)

    pool_sizes = [len(c) for c in cands]
    max_depth = max(pool_sizes) if pool_sizes else 0
    usable_ks = [k for k in ks if k <= max_depth] or [max_depth]
    skipped = [k for k in ks if k > max_depth]

    LOGGER.info(
        "Loaded %d notes; pool size mean=%.1f max=%d; applied default_thr=%.3f "
        "(+%d per-code overrides)",
        len(note_ids), float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        max_depth, default_thr, len(per_code),
    )

    # (1) candidate recall
    cand_rec = candidate_recall(note_ids, cands, gold, usable_ks)
    full_recall = candidate_recall(note_ids, cands, gold, [max_depth])[max_depth]

    # (2) scoring oracle
    oracle_f1 = scoring_oracle_f1(note_ids, cands, probs, gold)

    # (3) calibration headroom
    grid = [round(x, 4) for x in np.arange(0.01, 0.99 + 1e-9, grid_step)]
    realized = _micro_f1_at(note_ids, cands, probs, gold, default_thr)
    # NB: realized at the *flat* default ignores per-code overrides; we also
    # report the true realized (with per-code) set below.
    test_best_thr, test_best_f1 = sweep_threshold(note_ids, cands, probs, gold, grid)

    val_to_test = None
    if val_npz is not None and val_gold is not None:
        v_ids, v_c, v_p = load_npz(val_npz)
        v_gold = load_gold(val_gold)
        v_best_thr, v_best_f1 = sweep_threshold(v_ids, v_c, v_p, v_gold, grid)
        f1_applied = _micro_f1_at(note_ids, cands, probs, gold, v_best_thr)[0]
        val_to_test = {
            "val_best_threshold": v_best_thr,
            "val_best_micro_f1": round(v_best_f1, 4),
            "test_micro_f1_at_val_threshold": round(f1_applied, 4),
            "calibration_gap_vs_test_oracle": round(test_best_f1 - f1_applied, 4),
        }

    # (4) set size (uses the real per-code calibration)
    ss = set_size_stats(note_ids, cands, probs, gold, default_thr, per_code)

    result = {
        "inputs": {
            "npz": str(npz),
            "gold": str(gold_csv),
            "calibration": str(calibration) if calibration else None,
            "n_notes": len(note_ids),
            "pool_size_mean": round(float(np.mean(pool_sizes)), 2) if pool_sizes else 0.0,
            "pool_size_max": max_depth,
        },
        "candidate_recall_at_k": {str(k): round(v, 4) for k, v in cand_rec.items()},
        "candidate_recall_full_pool": round(full_recall, 4),
        "ks_skipped_pool_too_shallow": skipped,
        "scoring_oracle_topgold_f1": round(oracle_f1, 4),
        "realized_micro_f1_at_default_thr": round(realized[0], 4),
        "test_oracle_threshold": test_best_thr,
        "test_oracle_micro_f1": round(test_best_f1, 4),
        "calibration": val_to_test,
        "set_size": {k: round(v, 4) for k, v in ss.items()},
    }

    if by_category:
        cats = ("CPT_I", "CPT_II", "CPT_III", "PLA", "MAAA", "HCPCS")
        per_cat: Dict[str, Dict] = {}
        for cat in cats:
            g_cat = {
                nid: {c for c in g if classify_code(c) == cat}
                for nid, g in gold.items()
            }
            n_gold_cat = sum(len(v) for v in g_cat.values())
            if n_gold_cat == 0:
                continue
            c_cat = [[c for c in pool if classify_code(c) == cat] for pool in cands]
            # align probs to the filtered pool
            p_cat = []
            for pool, pr in zip(cands, probs):
                p_cat.append(
                    np.asarray(
                        [p for c, p in zip(pool, pr) if classify_code(c) == cat],
                        dtype=np.float64,
                    )
                )
            rec100 = candidate_recall(note_ids, c_cat, g_cat, [min(100, max_depth)])
            per_cat[cat] = {
                "n_gold": n_gold_cat,
                "cand_recall_at_100": round(list(rec100.values())[0], 4),
                "scoring_oracle_topgold_f1": round(
                    scoring_oracle_f1(note_ids, c_cat, p_cat, g_cat), 4
                ),
            }
        result["by_category"] = per_cat

    if by_strata and train_stats is not None:
        code_bin = load_code_bins(str(train_stats))
        result["by_strata"] = strata_breakdown(
            note_ids, cands, probs, gold, code_bin, default_thr, per_code
        )

    return result


def _print_report(r: Dict) -> None:
    inp = r["inputs"]
    print("=" * 68)
    print("ORACLE DECOMPOSITION")
    print("=" * 68)
    print(f"NPZ        : {inp['npz']}")
    print(f"Gold       : {inp['gold']}")
    print(f"Notes      : {inp['n_notes']:,}   pool size mean={inp['pool_size_mean']} "
          f"max={inp['pool_size_max']}")
    print("-" * 68)
    print("(1) Candidate recall @K  (retrieval ceiling)")
    for k, v in r["candidate_recall_at_k"].items():
        print(f"      @{k:<4} : {v:.4f}")
    print(f"      full  : {r['candidate_recall_full_pool']:.4f}  "
          f"(<- max recall ANY threshold can reach)")
    if r["ks_skipped_pool_too_shallow"]:
        print(f"      (skipped {r['ks_skipped_pool_too_shallow']} — pool shallower; "
              f"re-predict with a larger assembler top-k)")
    print("-" * 68)
    print("(2) Scoring oracle  (ranking ceiling, #pred == #gold)")
    print(f"      top-|gold| micro-F1 : {r['scoring_oracle_topgold_f1']:.4f}")
    print("-" * 68)
    print("(3) Calibration")
    print(f"      realized micro-F1 @default thr : {r['realized_micro_f1_at_default_thr']:.4f}")
    print(f"      test-oracle threshold          : {r['test_oracle_threshold']:.2f}")
    print(f"      test-oracle micro-F1           : {r['test_oracle_micro_f1']:.4f}")
    if r["calibration"]:
        c = r["calibration"]
        print(f"      val-best thr {c['val_best_threshold']:.2f} -> test F1 "
              f"{c['test_micro_f1_at_val_threshold']:.4f}  "
              f"(gap vs test-oracle {c['calibration_gap_vs_test_oracle']:+.4f})")
    print("-" * 68)
    print("(4) Set size")
    s = r["set_size"]
    print(f"      mean pred / gold : {s['mean_pred_size']:.2f} / {s['mean_gold_size']:.2f}")
    print(f"      set-size MAE     : {s['set_size_mae']:.2f}")
    print(f"      pred/gold ratio  : {s['pred_gold_ratio']:.2f}  "
          f"({'over' if s['pred_gold_ratio'] > 1 else 'under'}-predicts)")
    if "by_category" in r:
        print("-" * 68)
        print("By category   (n_gold | CandRec@100 | oracle top-|gold| F1)")
        for cat, cm in r["by_category"].items():
            print(f"      {cat:7s} : {cm['n_gold']:>5} | "
                  f"{cm['cand_recall_at_100']:.4f} | "
                  f"{cm['scoring_oracle_topgold_f1']:.4f}")
    if "by_strata" in r:
        print("-" * 68)
        print("By train-frequency bin   (n_gold | pool-ceiling | rank-oracle | realized)")
        print("      (pool-ceiling low -> retrieval-bound; high but realized low -> scorer-bound)")
        for b, bm in r["by_strata"].items():
            print(f"      {b:7s} : {bm['n_gold']:>5} | "
                  f"{bm['pool_recall_ceiling']:.4f} | "
                  f"{bm['scoring_oracle_recall']:.4f} | "
                  f"{bm['realized_recall']:.4f}")
    print("=" * 68)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Oracle decomposition for the bi-encoder.")
    p.add_argument("--npz", required=True, type=Path, help="test predictions.npz")
    p.add_argument("--gold", required=True, type=Path, help="test *_sectioned.csv")
    p.add_argument("--calibration", type=Path, default=None,
                   help="calibration.json (default_threshold + per_code).")
    p.add_argument("--val-npz", type=Path, default=None)
    p.add_argument("--val-gold", type=Path, default=None)
    p.add_argument("--ks", default="20,50,100,200,500",
                   help="comma-separated candidate depths.")
    p.add_argument("--by-category", action="store_true")
    p.add_argument("--train-stats", type=Path, default=None,
                   help="code_frequency_stats.csv (head/torso/tail bins) for "
                        "the per-strata breakdown.")
    p.add_argument("--by-strata", action="store_true",
                   help="Per-bin (head/torso/tail/unseen) pool-ceiling vs "
                        "scorer decomposition; needs --train-stats.")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    if args.by_strata and args.train_stats is None:
        p.error("--by-strata requires --train-stats")

    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    r = run(
        npz=args.npz, gold_csv=args.gold, ks=ks,
        calibration=args.calibration,
        val_npz=args.val_npz, val_gold=args.val_gold,
        by_category=args.by_category,
        train_stats=args.train_stats, by_strata=args.by_strata,
    )
    _print_report(r)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(r, f, indent=2, sort_keys=True)
        LOGGER.info("Wrote %s", args.out_json)


if __name__ == "__main__":
    main()
