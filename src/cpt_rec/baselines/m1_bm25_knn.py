#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
M1 (``m1_bm25_knn``) — BM25 retrieval over training notes,
weighted code vote.

For every test note:

1. Tokenize the note and query a BM25 index built over the *training*
   operative-note corpus.
2. Retrieve the top-``k`` most similar training notes.
3. For every code that appears in any neighbor's gold set, accumulate a
   vote score ``= sum_{neighbor with code} bm25_score`` (optionally
   normalized by ``sum(neighbor_scores)``).
4. Predict every code whose vote score is at least ``--vote-threshold``.
5. If thresholding produces an empty prediction, fall back to the gold
   code set of the single nearest neighbor (capped at ``--min-codes``)
   so the harness never sees a zero-prediction row.

This is a strong non-LLM, no-GPU baseline.  It memorizes the training
distribution and benefits directly from how repetitive operative notes
are within a hospital.

Three CLI subcommands
---------------------

::

    cptrec-m1-bm25-knn build-index \\
        --train outputs/datasets/vumc/train_eval.csv \\
        --index-out outputs/baselines/m1_bm25/index.pkl

    cptrec-m1-bm25-knn tune-threshold \\
        --notes outputs/datasets/vumc/val_eval.csv \\
        --index outputs/baselines/m1_bm25/index.pkl \\
        --out-json outputs/baselines/m1_bm25_knn/threshold.json

    cptrec-m1-bm25-knn predict \\
        --notes outputs/datasets/vumc/test_eval.csv \\
        --index outputs/baselines/m1_bm25/index.pkl \\
        --out outputs/baselines/m1_bm25_knn/predictions/test.csv \\
        --threshold-json outputs/baselines/m1_bm25_knn/threshold.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from cpt_rec.baselines.bm25_index import TrainNoteBM25Index
from cpt_rec.baselines.common import (
    Prediction,
    apply_seed_and_limit,
    crosswalk_codes_for_date,
    load_notes_for_prediction,
    log_prediction_stats,
    stats_sidecar,
    maybe_load_code_history,
    parse_as_of,
    write_predictions,
    write_scores_npz,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def build_index(
    train_csv: Path,
    index_out: Path,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    gold_code_col: str = "proc_codes",
) -> int:
    idx = TrainNoteBM25Index.from_csv(
        train_csv,
        note_id_col=note_id_col,
        note_text_col=note_text_col,
        gold_code_col=gold_code_col,
    )
    idx.save(index_out)
    return len(idx.note_ids)


# ---------------------------------------------------------------------------
# Vote aggregation
# ---------------------------------------------------------------------------

def vote_codes(
    neighbors: Sequence[Tuple[str, float, Set[str]]],
    threshold: float,
    normalize: bool,
) -> List[Tuple[str, float]]:
    """
    Aggregate codes from a neighbor list ``[(note_id, score, code_set), ...]``
    into a sorted list of ``(code, vote_score)`` above the threshold.
    """
    votes: Dict[str, float] = defaultdict(float)
    total_score = sum(max(0.0, s) for _, s, _ in neighbors)
    for _nid, score, code_set in neighbors:
        if score <= 0:
            continue
        weight = score
        for c in code_set:
            votes[c] += weight

    if normalize and total_score > 0:
        for c in list(votes):
            votes[c] /= total_score

    items = [(c, v) for c, v in votes.items() if v >= threshold]
    items.sort(key=lambda x: -x[1])
    return items


def _empty_prediction_fallback(
    neighbors: Sequence[Tuple[str, float, Set[str]]],
    min_codes: int,
) -> List[Tuple[str, float]]:
    """
    When thresholded voting returns nothing, fall back to the gold codes of
    the single nearest neighbor (sorted by neighbor score), capped at
    ``min_codes``.  Real op-notes always have ≥ 1 code; the harness should
    never see a row with ``pred_codes=""``.
    """
    if min_codes <= 0:
        return []
    for _nid, score, code_set in neighbors:
        if score > 0 and code_set:
            picked = list(code_set)[:min_codes]
            return [(c, float(score)) for c in picked]
    return []


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict_b1(
    notes_csv: Path,
    index_path: Path,
    out_csv: Path,
    top_k: int = 25,
    vote_threshold: float = 0.10,
    normalize_votes: bool = True,
    min_codes: int = 1,
    seed: int = 42,
    limit: Optional[int] = None,
    history_changes: Optional[Path] = None,
    history_deleted: Optional[Path] = None,
    kb_csv: Optional[Path] = None,
    as_of_col: str = "PROCEDURE_DATE",
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    dump_scores_npz: Optional[Path] = None,
) -> int:
    LOGGER.info("Loading BM25 train-note index from %s", index_path)
    index = TrainNoteBM25Index.load(index_path)
    LOGGER.info("Loaded index: %d train notes", len(index.note_ids))

    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)
    LOGGER.info("Loaded %d notes from %s", len(df), notes_csv)

    history, _ = maybe_load_code_history(
        history_changes=history_changes,
        history_deleted=history_deleted,
        kb_csv=kb_csv,
    )
    if history is not None and as_of_col not in df.columns:
        LOGGER.warning(
            "M1 history requested but as_of column %r not in CSV; "
            "crosswalk will be a no-op", as_of_col,
        )

    predictions: List[Prediction] = []
    pool_rankings: List[Tuple[str, List[str], List[float]]] = []
    n_fallback = 0
    n_crosswalked = 0
    note_ids = df["note_id"].astype(str).tolist()
    texts = df["note_text"].astype(str).tolist()
    as_of_raw = (
        df[as_of_col].tolist() if (history is not None and as_of_col in df.columns)
        else [None] * len(df)
    )
    for note_id, text, raw_as_of in tqdm(
        zip(note_ids, texts, as_of_raw), total=len(df), desc="M1 BM25 KNN-vote"
    ):
        as_of = parse_as_of(raw_as_of) if history is not None else None
        nbrs = index.neighbors(text, top_k=top_k)
        if history is not None and as_of is not None:
            new_nbrs = []
            for nid, score, code_set in nbrs:
                aligned = crosswalk_codes_for_date(
                    code_set, history=history, as_of=as_of, keep_unresolved=False,
                )
                if aligned != code_set:
                    n_crosswalked += 1
                new_nbrs.append((nid, score, aligned))
            nbrs = new_nbrs
        if dump_scores_npz is not None:
            # The *unthresholded* vote pool: what a reviewer could be shown at
            # any budget B, as opposed to the vote-threshold operating point.
            full = vote_codes(nbrs, threshold=0.0, normalize=normalize_votes)
            pool_rankings.append(
                (note_id, [c for c, _ in full], [float(v) for _, v in full])
            )
        ranked = vote_codes(
            nbrs, threshold=vote_threshold, normalize=normalize_votes
        )
        if not ranked:
            ranked = _empty_prediction_fallback(nbrs, min_codes=min_codes)
            if ranked:
                n_fallback += 1
        codes = [c for c, _ in ranked]
        scores = [float(v) for _, v in ranked]
        predictions.append(
            Prediction(note_id=note_id, codes=codes, scores=scores)
        )

    write_predictions(predictions, out_csv, include_scores=True)
    if dump_scores_npz is not None:
        write_scores_npz(pool_rankings, dump_scores_npz)
    log_prediction_stats(predictions, label="M1",
                         out_path=stats_sidecar(out_csv))
    if n_fallback:
        LOGGER.info(
            "M1: %d/%d notes used the nearest-neighbor fallback "
            "(threshold produced 0 codes)",
            n_fallback, len(predictions),
        )
    if history is not None and n_crosswalked:
        LOGGER.info(
            "M1: CodeHistory crosswalk modified %d neighbor code sets "
            "(retired-code remapping)", n_crosswalked,
        )
    return len(predictions)


# ---------------------------------------------------------------------------
# Threshold tuning (sweep against val gold)
# ---------------------------------------------------------------------------

def _micro_f1(pred_sets: Sequence[Set[str]], gold_sets: Sequence[Set[str]]) -> float:
    tp = fp = fn = 0
    for p, g in zip(pred_sets, gold_sets):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom else 0.0


def tune_threshold(
    notes_csv: Path,
    index_path: Path,
    out_json: Path,
    top_k: int = 25,
    normalize_votes: bool = True,
    grid: Sequence[float] = (
        0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90,
    ),
    min_codes: int = 1,
    gold_code_col: str = "proc_codes",
    seed: int = 42,
    limit: Optional[int] = None,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Sweep ``--vote-threshold`` against val-set micro-F1 and write the chosen
    value to a JSON sidecar that ``predict`` can consume via
    ``--threshold-json``.  Iterating once and re-thresholding is O(threshold-
    grid) cheaper than re-running BM25 per setting.
    """
    LOGGER.info("Loading BM25 train-note index from %s", index_path)
    index = TrainNoteBM25Index.load(index_path)

    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    if gold_code_col not in df.columns:
        raise ValueError(
            f"Tune CSV missing gold column {gold_code_col!r}; "
            f"have {list(df.columns)}"
        )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)

    gold_sets: List[Set[str]] = []
    for s in df[gold_code_col].fillna(""):
        gold_sets.append({c.strip().upper() for c in str(s).split("|") if c.strip()})

    # Run BM25 once and collect the (code -> raw_vote) dicts so the grid
    # sweep is just re-thresholding.
    per_note_votes: List[Dict[str, float]] = []
    per_note_neighbors: List[List[Tuple[str, float, Set[str]]]] = []
    for text in tqdm(
        df["note_text"].astype(str), total=len(df), desc="M1 tune (run BM25)"
    ):
        nbrs = index.neighbors(text, top_k=top_k)
        per_note_neighbors.append(nbrs)
        votes: Dict[str, float] = defaultdict(float)
        total = sum(max(0.0, s) for _, s, _ in nbrs)
        for _nid, score, code_set in nbrs:
            if score <= 0:
                continue
            for c in code_set:
                votes[c] += score
        if normalize_votes and total > 0:
            for c in list(votes):
                votes[c] /= total
        per_note_votes.append(dict(votes))

    results: List[Dict[str, float]] = []
    for thr in grid:
        pred_sets: List[Set[str]] = []
        for votes, nbrs in zip(per_note_votes, per_note_neighbors):
            kept = {c for c, v in votes.items() if v >= thr}
            if not kept:
                fb = _empty_prediction_fallback(nbrs, min_codes=min_codes)
                kept = {c for c, _ in fb}
            pred_sets.append(kept)
        f1 = _micro_f1(pred_sets, gold_sets)
        results.append({"threshold": thr, "micro_f1": round(f1, 4)})
        LOGGER.info("  threshold=%.3f -> micro_f1=%.4f", thr, f1)

    best = max(results, key=lambda r: r["micro_f1"])
    sorted_grid = sorted(grid)
    at_lower_edge = best["threshold"] == sorted_grid[0]
    at_upper_edge = best["threshold"] == sorted_grid[-1]

    payload = {
        "best_threshold": best["threshold"],
        "best_micro_f1": best["micro_f1"],
        "top_k": top_k,
        "normalize_votes": normalize_votes,
        "min_codes": min_codes,
        "n_notes": len(df),
        "grid": results,
        "best_at_lower_edge": at_lower_edge,
        "best_at_upper_edge": at_upper_edge,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info(
        "M1 tune: best threshold=%.3f (micro_f1=%.4f) -> %s",
        best["threshold"], best["micro_f1"], out_json,
    )
    if at_upper_edge:
        LOGGER.warning(
            "M1 tune: best threshold=%.3f is the LARGEST value in the grid; "
            "F1 may still be climbing.  Re-run with --grid extended above "
            "%.2f (e.g. `--grid 0.50 0.60 0.70 0.80 0.90 0.95`) to find "
            "the true peak.",
            best["threshold"], best["threshold"],
        )
    if at_lower_edge:
        LOGGER.warning(
            "M1 tune: best threshold=%.3f is the SMALLEST value in the "
            "grid; consider re-running with smaller --grid values to find "
            "the true peak.",
            best["threshold"],
        )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_index_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("build-index",
                      help="Build a BM25 index over training notes.")
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--index-out", required=True, type=Path)
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)
    p.add_argument("--gold-code-col", default="proc_codes")


def _build_predict_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("predict",
                      help="Predict codes via BM25 KNN-vote.")
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--vote-threshold", type=float, default=0.10,
                   help="Min (normalized) vote score to predict a code; "
                        "ignored if --threshold-json is provided.")
    p.add_argument("--threshold-json", type=Path, default=None,
                   help="Optional JSON from `tune-threshold` whose "
                        "`best_threshold` overrides --vote-threshold.")
    p.add_argument("--no-normalize-votes", dest="normalize_votes",
                   action="store_false")
    p.set_defaults(normalize_votes=True)
    p.add_argument("--min-codes", type=int, default=1,
                   help="Floor for empty-prediction fallback (use the "
                        "nearest neighbor's gold codes capped at N).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Predict only the first N notes (smoke runs).")
    p.add_argument("--history-changes", type=Path, default=None,
                   help="Optional code_changes.csv for date-aware "
                        "neighbor code crosswalk; pair with --history-"
                        "deleted and --kb.")
    p.add_argument("--history-deleted", type=Path, default=None)
    p.add_argument("--kb", type=Path, default=None,
                   help="KB CSV (required when history flags are set).")
    p.add_argument("--as-of-col", default="PROCEDURE_DATE",
                   help="Date column on the test CSV used as the "
                        "crosswalk anchor.")
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)
    p.add_argument("--dump-scores-npz", type=Path, default=None,
                   help="Also write the UNTHRESHOLDED vote pool to this "
                        "NPZ (note_ids/candidate_codes/probs), for "
                        "`cptrec-evaluate --scores-npz`.  Off by default; the "
                        "predictions CSV is unchanged either way.")


def _build_tune_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("tune-threshold",
                      help="Sweep --vote-threshold on a labeled CSV.")
    p.add_argument("--notes", required=True, type=Path,
                   help="Labeled CSV (val split with proc_codes column).")
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--no-normalize-votes", dest="normalize_votes",
                   action="store_false")
    p.set_defaults(normalize_votes=True)
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--min-codes", type=int, default=1)
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25,
                            0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                            0.70, 0.80, 0.90],
                   help="Threshold values to sweep.  Default spans 0.05 "
                        "to 0.90; the tuner warns if the best value is "
                        "at either edge so you know to extend.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="M1: BM25 KNN-vote over training notes."
    )
    sp = parser.add_subparsers(dest="cmd", required=True)
    _build_index_parser(sp)
    _build_predict_parser(sp)
    _build_tune_parser(sp)
    args = parser.parse_args()

    if args.cmd == "build-index":
        n = build_index(
            train_csv=args.train,
            index_out=args.index_out,
            note_id_col=args.note_id_col,
            note_text_col=args.note_text_col,
            gold_code_col=args.gold_code_col,
        )
        LOGGER.info("M1 build-index done: %d train notes indexed -> %s",
                    n, args.index_out)
    elif args.cmd == "predict":
        # Resolve threshold from sidecar JSON if provided.
        thr = args.vote_threshold
        if args.threshold_json is not None:
            with open(args.threshold_json) as f:
                payload = json.load(f)
            thr = float(payload["best_threshold"])
            LOGGER.info(
                "M1 predict: using --threshold-json (%.4f, micro_f1=%s)",
                thr, payload.get("best_micro_f1"),
            )
        n = predict_b1(
            notes_csv=args.notes,
            index_path=args.index,
            out_csv=args.out,
            top_k=args.top_k,
            vote_threshold=thr,
            normalize_votes=args.normalize_votes,
            min_codes=args.min_codes,
            seed=args.seed,
            limit=args.limit,
            history_changes=args.history_changes,
            history_deleted=args.history_deleted,
            kb_csv=args.kb,
            as_of_col=args.as_of_col,
            dump_scores_npz=args.dump_scores_npz,
            note_id_col=args.note_id_col,
            note_text_col=args.note_text_col,
        )
        LOGGER.info("M1 predict done: %d notes scored -> %s", n, args.out)
    elif args.cmd == "tune-threshold":
        tune_threshold(
            notes_csv=args.notes,
            index_path=args.index,
            out_json=args.out_json,
            top_k=args.top_k,
            normalize_votes=args.normalize_votes,
            grid=tuple(args.grid),
            min_codes=args.min_codes,
            gold_code_col=args.gold_code_col,
            seed=args.seed,
            limit=args.limit,
            note_id_col=args.note_id_col,
            note_text_col=args.note_text_col,
        )


if __name__ == "__main__":
    main()
