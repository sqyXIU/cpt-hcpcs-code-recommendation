#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Unified evaluation harness for modifier-agnostic CPT/HCPCS predictions.

Reads a *predictions* CSV and a *gold* (normalized) CSV, aligns them on
note id, and emits:

* ``metrics.json`` — all set-level, per-category, and per-bin metrics
* ``per_bin_metrics.csv`` — head/torso/tail/unseen slice table
* ``per_code_metrics.csv`` — per-code TP / FP / FN for error analysis
* ``constraint_metrics.json`` — NCCI violation rates (optional, if
  ``--ncci-dir`` is provided)
* ``summary.txt`` — human-readable one-page summary

With ``--scores-npz`` it additionally computes the **shortlist-review** suite
(``R@B``, ``Coverage@B``, ``B@R``, ``FamilyMRR``, ``MAP``/``nDCG``, ``AUC-PR``)
into a ``"ranking"`` block — see
:mod:`cpt_rec.common.evaluation.rank_metrics` for why micro-F1 is
no longer the decision metric.  Without the flag, every output byte is
unchanged.

Expected input schemas
----------------------

**Predictions CSV** (columns):

- ``note_id`` (string; matches the gold NOTE_ID)
- ``pred_codes`` (pipe-separated string of predicted codes)
- ``pred_scores`` (optional; pipe-separated floats aligned with codes)

**Gold CSV** (columns):

- ``NOTE_ID`` (or ``note_id``)
- ``proc_codes`` (pipe-separated gold codes, from ``label_normalizer``)

Example usage
-------------

::

    cptrec-evaluate \\
      --predictions outputs/baselines/m3_zeroshot_frontier/predictions/test.csv \\
      --gold outputs/datasets/vumc/test_eval_sectioned.csv \\
      --train-stats outputs/datasets/vumc/code_frequency_stats.csv \\
      --ncci-dir data/ncci \\
      --out-dir outputs/evaluation/m3_zeroshot_frontier/test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from cpt_rec.baselines.common import _ID_CANDIDATES, _pick_col
from cpt_rec.common.evaluation.constraint_metrics import (
    compute_constraint_metrics,
)
from cpt_rec.common.evaluation.code_bins import (
    bin_metrics_table,
    compute_bin_metrics,
    load_code_bins,
)
from cpt_rec.common.evaluation.metrics import (
    bootstrap_set_metrics,
    compute_set_metrics,
)
from cpt_rec.common.evaluation.rank_metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_RECALL_TARGETS,
    bootstrap_rank_metrics,
    compute_rank_metrics,
    load_scores_npz,
)
from cpt_rec.common.preprocess.code_utils import (
    pipe_split_codes as _pipe_split,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_gold(gold_csv: Path | str, code_col: str = "proc_codes") -> Dict[str, Set[str]]:
    """Load ``note_id -> set(codes)`` from a normalized gold CSV."""
    df = pd.read_csv(gold_csv, dtype=str, usecols=lambda c: True)
    id_col = _pick_col(df, _ID_CANDIDATES, f"note-id (in {gold_csv})")
    if code_col not in df.columns:
        raise ValueError(f"Gold CSV {gold_csv} missing code column '{code_col}'.")
    # Column-wise zip is ~10-20x faster than ``df.iterrows`` for 2-column reads.
    out: Dict[str, Set[str]] = {
        str(nid).strip(): set(_pipe_split(codes))
        for nid, codes in zip(df[id_col], df[code_col])
    }
    LOGGER.info("Loaded gold: %d notes from %s", len(out), gold_csv)
    return out


def load_predictions(
    pred_csv: Path | str,
    code_col: str = "pred_codes",
) -> Dict[str, Set[str]]:
    """Load ``note_id -> set(predicted codes)``."""
    df = pd.read_csv(pred_csv, dtype=str)
    id_col = _pick_col(df, _ID_CANDIDATES, f"note-id (in {pred_csv})")
    if code_col not in df.columns:
        raise ValueError(f"Predictions CSV {pred_csv} missing '{code_col}'.")
    out: Dict[str, Set[str]] = {
        str(nid).strip(): set(_pipe_split(codes))
        for nid, codes in zip(df[id_col], df[code_col])
    }
    LOGGER.info("Loaded predictions: %d notes from %s", len(out), pred_csv)
    return out


def align(
    gold: Dict[str, Set[str]],
    pred: Dict[str, Set[str]],
    strict: bool = False,
) -> List[Tuple[str, Set[str], Set[str]]]:
    """
    Align predictions to gold on note_id.

    Notes present in gold but missing from predictions are treated as an
    empty predicted set (false negatives on everything).  If ``strict``,
    raises when any gold note is missing.
    """
    aligned = []
    n_missing = 0
    extra_pred = 0

    for nid, gset in gold.items():
        pset = pred.get(nid)
        if pset is None:
            n_missing += 1
            if strict:
                raise ValueError(f"No prediction for note_id {nid}")
            pset = set()
        aligned.append((nid, gset, pset))

    # Count predictions with no matching gold note (won't be evaluated)
    for nid in pred:
        if nid not in gold:
            extra_pred += 1

    if n_missing:
        LOGGER.warning(
            "%d gold notes have no prediction; treated as empty predicted set.",
            n_missing,
        )
    if extra_pred:
        LOGGER.warning(
            "%d prediction notes have no matching gold note and were skipped.",
            extra_pred,
        )
    return aligned


def load_family_map(kb_csv: Path | str, codes: Set[str]) -> Dict[str, Optional[str]]:
    """Build ``code -> deepest-KB-range`` for the sibling companion metrics.

    "Same family" is a string compare on the deepest range, matching
    :mod:`sibling_analysis` so ``FamilyMRR`` and ``sibling_fp_rate`` partition
    the label space identically.  Restricted to ``codes`` (the codes actually
    seen in gold or in a candidate list) to avoid a full-KB scan.
    """
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
    out: Dict[str, Optional[str]] = {}
    for code in codes:
        h = kb.hierarchy(code)
        out[code] = h[-1][0] if h else None
    n_known = sum(1 for v in out.values() if v is not None)
    LOGGER.info("Family map: %d/%d codes resolved to a KB range.", n_known, len(out))
    return out


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def run_evaluation(
    predictions_csv: Path,
    gold_csv: Path,
    out_dir: Path,
    train_stats_csv: Optional[Path] = None,
    ncci_dir: Optional[Path] = None,
    gold_code_col: str = "proc_codes",
    pred_code_col: str = "pred_codes",
    bootstrap: int = 0,
    bootstrap_seed: int = 12345,
    scores_npz: Optional[Path] = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    recall_targets: Sequence[float] = DEFAULT_RECALL_TARGETS,
    kb_csv: Optional[Path] = None,
    rank_bootstrap: int = 0,
    rank_topn: Optional[int] = None,
) -> Dict:
    """
    End-to-end evaluation pipeline.

    Returns the aggregated metrics dict (also written to ``out_dir/metrics.json``).

    ``bootstrap > 0`` additionally computes note-level percentile-bootstrap 95%
    CIs for the headline set metrics (``bootstrap`` resamples) and adds a
    ``"bootstrap"`` block to ``metrics.json`` / ``summary.txt``. Default 0 =
    off, output unchanged.

    ``scores_npz`` points at the full-pool score companion written by
    ``cptrec-verifier-predict --save-npz``.  When given, the shortlist-review suite
    is computed at each budget in ``budgets`` and added as a ``"ranking"`` block
    (plus ``rank_metrics.json``); ``kb_csv`` additionally enables the
    ``FamilyMRR`` companions.  Default ``None`` = off, output unchanged.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold(gold_csv, code_col=gold_code_col)
    pred = load_predictions(predictions_csv, code_col=pred_code_col)
    aligned = align(gold, pred)
    pairs = [(g, p) for (_nid, g, p) in aligned]

    # ------------ set-level metrics ------------
    set_metrics = compute_set_metrics(pairs, compute_per_category=True)
    LOGGER.info(
        "Set metrics: micro-F1=%.4f  macro-F1=%.4f  exact-match=%.4f",
        set_metrics.micro_f1,
        set_metrics.macro_f1,
        set_metrics.exact_match,
    )

    # ------------ per-code table ------------
    per_code_rows = []
    for code, (tp, fp, fn) in sorted(set_metrics.per_code.items()):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_code_rows.append(
            {
                "code": code,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
    per_code_df = pd.DataFrame(per_code_rows).sort_values("f1", ascending=False)
    per_code_df.to_csv(out_dir / "per_code_metrics.csv", index=False)

    # ------------ drift / bin metrics ------------
    bin_results = None
    code_bin_map = None
    if train_stats_csv is not None:
        code_bin_map = load_code_bins(train_stats_csv)
        bin_results = compute_bin_metrics(pairs, code_bin_map)
        bin_df = bin_metrics_table(bin_results)
        bin_df.to_csv(out_dir / "per_bin_metrics.csv", index=False)

    # ------------ constraint metrics ------------
    constraint_results = None
    if ncci_dir is not None:
        from cpt_rec.common.ncci.rule_checker import NCCIRuleChecker
        checker = NCCIRuleChecker.from_data_dir(ncci_dir)
        constraint_results = compute_constraint_metrics(
            (p for _g, p in pairs), checker
        )
        with open(out_dir / "constraint_metrics.json", "w") as f:
            json.dump(constraint_results.to_dict(), f, indent=2, sort_keys=True)

    # ------------ bootstrap CIs (optional) ------------
    bootstrap_results = None
    if bootstrap > 0:
        LOGGER.info("Bootstrapping %d note-level resamples (seed=%d) ...",
                    bootstrap, bootstrap_seed)
        bootstrap_results = bootstrap_set_metrics(
            pairs, n_boot=bootstrap, seed=bootstrap_seed
        )

    # ------------ shortlist-review (ranking) metrics ------------
    rank_results = None
    if scores_npz is not None:
        rankings = load_scores_npz(scores_npz, topn=rank_topn)
        family_of = None
        if kb_csv is not None:
            seen = {c for gset in gold.values() for c in gset}
            for codes, _scores in rankings.values():
                seen.update(codes)
            family_of = load_family_map(kb_csv, seen)
        rank_metrics = compute_rank_metrics(
            rankings,
            gold,
            budgets=budgets,
            recall_targets=recall_targets,
            family_of=family_of,
            code_bin=code_bin_map,
        )
        rank_results = rank_metrics.to_dict()
        if rank_bootstrap > 0:
            LOGGER.info("Bootstrapping %d note-level resamples for R@B (seed=%d) ...",
                        rank_bootstrap, bootstrap_seed)
            rank_results["bootstrap"] = bootstrap_rank_metrics(
                rankings, gold, budgets=budgets,
                n_boot=rank_bootstrap, seed=bootstrap_seed,
            )
        primary = rank_metrics.recall_at.get(5)
        LOGGER.info(
            "Ranking: R@5=%s  R@10=%s  Coverage@5=%s  pool ceiling=%.4f",
            f"{primary:.4f}" if primary is not None else "n/a",
            f"{rank_metrics.recall_at[10]:.4f}" if 10 in rank_metrics.recall_at else "n/a",
            f"{rank_metrics.coverage_at[5]:.4f}" if 5 in rank_metrics.coverage_at else "n/a",
            rank_metrics.pool_ceiling,
        )
        with open(out_dir / "rank_metrics.json", "w") as f:
            json.dump(rank_results, f, indent=2, sort_keys=True)

    # ------------ aggregated JSON ------------
    aggregated = {"set": set_metrics.to_dict(include_per_code=False)}
    if bin_results is not None:
        aggregated["per_bin"] = {b: m.to_dict() for b, m in bin_results.items()}
    if constraint_results is not None:
        aggregated["constraints"] = constraint_results.to_dict(include_examples=True)
    if bootstrap_results:
        aggregated["bootstrap"] = bootstrap_results
    if rank_results is not None:
        aggregated["ranking"] = rank_results

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, sort_keys=True)

    # ------------ human summary ------------
    _write_summary_txt(out_dir / "summary.txt", aggregated)

    return aggregated


def _write_summary_txt(path: Path, agg: Dict) -> None:
    lines = []
    s = agg["set"]
    lines.append("=" * 60)
    lines.append("EVALUATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Notes evaluated      : {s['n_notes']:,}")
    lines.append(f"Gold codes (total)   : {s['n_gold_codes']:,}")
    lines.append(f"Predicted codes      : {s['n_pred_codes']:,}")
    lines.append(f"TP / FP / FN         : {s['n_tp']:,} / {s['n_fp']:,} / {s['n_fn']:,}")
    lines.append("")
    lines.append(f"Micro  P / R / F1    : {s['micro_precision']:.4f} / "
                 f"{s['micro_recall']:.4f} / {s['micro_f1']:.4f}")
    lines.append(f"Macro  P / R / F1    : {s['macro_precision']:.4f} / "
                 f"{s['macro_recall']:.4f} / {s['macro_f1']:.4f}")
    lines.append(f"Exact-set match      : {s['exact_match']:.4f}")
    lines.append(f"Mean Jaccard         : {s['jaccard_mean']:.4f}")
    if "example_f1" in s:
        lines.append(f"Example (per-note) F1: {s['example_f1']:.4f}")

    if "bootstrap" in agg:
        b = agg["bootstrap"]
        meta = b.get("_meta", {})
        lines.append("")
        lines.append(
            f"Bootstrap 95% CIs ({int(meta.get('n_boot', 0))} note-level "
            f"resamples, seed={int(meta.get('seed', 0))}):"
        )
        for name in ("micro_f1", "macro_f1", "example_f1",
                     "exact_match", "jaccard_mean"):
            if name in b:
                m = b[name]
                lines.append(
                    f"  {name:12s} : {m['mean']:.4f} "
                    f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]  (sd {m['std']:.4f})"
                )

    if "per_category" in s:
        lines.append("")
        lines.append("Per-category micro-F1 / macro-F1 :")
        for cat, cm in s["per_category"].items():
            lines.append(
                f"  {cat:8s} : micro-F1={cm['micro_f1']:.4f}  "
                f"macro-F1={cm['macro_f1']:.4f}  (n_gold={cm['n_gold_codes']:,})"
            )

    if "per_bin" in agg:
        lines.append("")
        lines.append("Per-bin metrics (head / torso / tail / unseen):")
        for b, bm in agg["per_bin"].items():
            lines.append(
                f"  {b:6s} : n_codes={bm['n_codes']:,}  "
                f"micro-F1={bm['micro_f1']:.4f}  macro-F1={bm['macro_f1']:.4f}"
            )

    if "ranking" in agg:
        r = agg["ranking"]
        lines.append("")
        lines.append("SHORTLIST REVIEW  (primary decision metric: R@5)")
        lines.append(f"  Candidates/note (mean): {r['n_candidates_mean']:.1f}"
                     f"   pool ceiling: {r['pool_ceiling']:.4f}")
        lines.append(f"  {'budget':>7} {'shown':>7} {'R@B':>8} {'P@B':>8} "
                     f"{'Cover@B':>9} {'MAP@B':>8} {'nDCG@B':>8}")
        for b in sorted(r["recall_at"], key=int):
            star = " *" if b == "5" else "  "
            lines.append(
                f"  {b:>7}{star}{r['shown_at'][b]:>6.1f} {r['recall_at'][b]:>8.4f} "
                f"{r['precision_at'][b]:>8.4f} {r['coverage_at'][b]:>9.4f} "
                f"{r['map_at'][b]:>8.4f} {r['ndcg_at'][b]:>8.4f}"
            )
        burdens = ", ".join(
            f"R>={t}: {v:.2f}/note" if v is not None else f"R>={t}: unreachable"
            for t, v in sorted(r["burden_at_recall"].items())
        )
        lines.append(f"  Burden to reach   : {burdens}")
        if "auc_pr" in r:
            lines.append(f"  AUC-PR            : {r['auc_pr']:.4f}")
        if "family" in r:
            fm = r["family"]
            lines.append(
                f"  FamilyMRR (pool)  : {fm['family_mrr']:.4f}   "
                f"top-1-in-family: {fm['top1_in_family']:.4f}   "
                f"(n={fm['n_gold_scored']:,})"
            )
            at = ", ".join(
                f"@{b}: {v:.4f}" for b, v in sorted(fm["family_mrr_at"].items(), key=lambda kv: int(kv[0]))
                if v is not None
            )
            if at:
                lines.append(f"  FamilyMRR in list : {at}")
        if "per_bin" in r:
            lines.append("  Per-bin recall:")
            for bn, bm in r["per_bin"].items():
                at = "  ".join(
                    f"R@{b}={v:.4f}" for b, v in sorted(bm["recall_at"].items(), key=lambda kv: int(kv[0]))
                )
                lines.append(f"    {bn:6s} (n_gold={bm['n_gold_codes']:,}, "
                             f"ceiling={bm['pool_ceiling']:.4f})  {at}")
        if "bootstrap" in r:
            lines.append("  Bootstrap 95% CIs:")
            for name, m in r["bootstrap"].items():
                if name == "_meta":
                    continue
                lines.append(f"    {name:16s} : {m['mean']:.4f} "
                             f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")

    if "constraints" in agg:
        c = agg["constraints"]
        lines.append("")
        lines.append("NCCI constraint violation rates :")
        lines.append(f"  Hard PTP               : {c['rate_hard_ptp_violations']:.4f}")
        lines.append(f"  Modifier-contingent PTP: {c['rate_modifier_ptp_violations']:.4f}")
        lines.append(f"  AOC (Type 1 hard)      : {c['rate_aoc_hard_violations']:.4f}")
        lines.append(f"  AOC (Type 3 contractor): {c['rate_aoc_contractor_violations']:.4f}")
        lines.append(f"  MUE                    : {c['rate_mue_violations']:.4f}")
        lines.append(f"  Hard-valid (pass all)  : {c['rate_hard_valid']:.4f}")
    lines.append("=" * 60)

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate modifier-agnostic CPT/HCPCS predictions."
    )
    p.add_argument("--predictions", required=True, type=Path,
                   help="Predictions CSV (columns: note_id, pred_codes, [pred_scores]).")
    p.add_argument("--gold", required=True, type=Path,
                   help="Normalized gold CSV (columns: NOTE_ID, proc_codes).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for metrics files. Defaults to "
            "<predictions parent>/<predictions stem>_metrics/ — e.g. predictions "
            "at outputs/baselines/kb_index/predictions/val.csv lands metrics at "
            "outputs/baselines/kb_index/predictions/val_metrics/."
        ),
    )
    p.add_argument("--train-stats", type=Path, default=None,
                   help="code_frequency_stats.csv for head/torso/tail binning.")
    p.add_argument("--ncci-dir", type=Path, default=None,
                   help="Path to data/ncci for constraint checks.")
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--pred-code-col", default="pred_codes")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="N note-level bootstrap resamples for 95%% CIs on the "
                        "headline metrics (0 = off; 1000 is typical).")
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument(
        "--scores-npz",
        type=Path,
        default=None,
        help=(
            "Full-pool candidate scores from `cptrec-verifier-predict --save-npz`, "
            "`cptrec-m1-bm25-knn|cptrec-build-kb-index predict --dump-scores-npz`, or "
            "`cptrec-m2-longformer dump-scores`. Enables the shortlist-review suite "
            "(R@B, Coverage@B, B@R, MAP, nDCG, AUC-PR) — the primary decision "
            "metric is R@5. Omit for byte-identical legacy output."
        ),
    )
    p.add_argument(
        "--rank-topn",
        type=int,
        default=None,
        help=(
            "Ranking depth kept per note. Default: the whole retrieved pool "
            "for a ragged NPZ, top-1000 labels for a dense (full-label-space) "
            "one. Only bounds `pool_ceiling`; R@B for B <= this is unaffected."
        ),
    )
    p.add_argument(
        "--budgets",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BUDGETS),
        help="Comma-separated review budgets (codes shown per note).",
    )
    p.add_argument(
        "--recall-targets",
        type=str,
        default=",".join(str(t) for t in DEFAULT_RECALL_TARGETS),
        help="Comma-separated recall targets for the B@R dual.",
    )
    p.add_argument(
        "--kb",
        type=Path,
        default=None,
        help=(
            "codes_with_ranges.csv — enables the FamilyMRR / top-1-in-family "
            "companions, which keep sibling confusion visible (R@B alone scores a "
            "shortlist containing both siblings as a success)."
        ),
    )
    p.add_argument("--rank-bootstrap", type=int, default=0,
                   help="N note-level bootstrap resamples for R@B / Coverage@B "
                        "CIs (0 = off).")
    return p.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in str(raw).split(",") if str(x).strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in str(raw).split(",") if str(x).strip()]


def _default_out_dir(predictions_csv: Path) -> Path:
    """Sibling folder named ``<stem>_metrics`` next to the predictions CSV.

    e.g. ``.../predictions/val.csv`` -> ``.../predictions/val_metrics/``, and
    ``.../val_default_v2/predictions.csv`` -> ``.../val_default_v2/predictions_metrics/``.

    The ``_metrics`` suffix (rather than the bare stem) avoids two problems:
    a CSV literally named ``predictions.csv`` would otherwise nest metrics in a
    confusing ``predictions/predictions/`` path, and several prediction CSVs
    sharing one ``predictions/`` dir (``val.csv``, ``test.csv``) would otherwise
    each need a distinct stem to avoid colliding.
    """
    p = Path(predictions_csv)
    return p.parent / f"{p.stem}_metrics"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir(args.predictions)
    run_evaluation(
        predictions_csv=args.predictions,
        gold_csv=args.gold,
        out_dir=out_dir,
        train_stats_csv=args.train_stats,
        ncci_dir=args.ncci_dir,
        gold_code_col=args.gold_code_col,
        pred_code_col=args.pred_code_col,
        bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        scores_npz=args.scores_npz,
        budgets=_parse_int_list(args.budgets),
        recall_targets=_parse_float_list(args.recall_targets),
        kb_csv=args.kb,
        rank_bootstrap=args.rank_bootstrap,
        rank_topn=args.rank_topn,
    )


if __name__ == "__main__":
    main()
