#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Candidate-source union — retrieval-ceiling check (no GPU).

The oracle decomposition (:mod:`evaluation.oracle`) showed the model_core
pipeline is *retrieval-bound*: with the KB-description-only candidate pool
(BM25 ∪ dense over CPT/HCPCS descriptions), the full-pool CandRecall tops
out around 0.56 — i.e. ~44% of gold codes never enter the pool the scorer
sees, so no scorer / threshold / decoder can ever recover them.

This tool answers the prerequisite question *before* spending any GPU on a
wider-pool retrain:

    *If we union additional candidate sources into each note's pool, how
    much does the retrieval ceiling (full-pool CandRecall) actually rise,
    and at what pool-size cost?*

Two extra sources, mirroring the strong baselines:

* ``C^neighbor`` — gold code sets of the top-k BM25 training-note
  neighbours of the test note (the same ``bm25.pkl`` index M1/M4 use).
  Captures codes that co-occur in lexically similar operative notes but
  whose KB *description* doesn't lexically/semantically match the note.
* ``C^LLM-prior`` — the codes a GPT baseline (M3 zero-shot / M4 RAG)
  predicted for the note.  Captures codes the description retriever misses
  but a frontier LLM surfaces from the note text.

Crucially this uses **no test gold** to *build* the pools (only to
*measure* recall), and the neighbour source draws only on *training* gold —
so it is a legitimate candidate generator, not leakage.  The scorer would
still have to pick the right codes out of the (now larger) pool.

What it reports
---------------

* Full-pool recall for KB alone (the current ceiling) and for each
  marginal source added on top, plus the full union (the *new* ceiling).
* Of the gold codes currently unreachable (not in the KB pool), how many
  each source recovers — the direct ceiling-lift measure.
* Pool-size inflation (mean candidates/note) for each combination — the
  GPU cost the scorer pays at train/infer time.
* A neighbour-depth sweep (recall vs. top-k) so you can pick the knee.
* Optional per-category breakdown of the recovered gold.

Decision gate
-------------

* Union recall ≫ KB recall (e.g. 0.56 → 0.75+)  -> the extra sources pay:
  wire the winning source(s) into ``candidate_gen`` / ``predict`` and
  retrain the scorer on the wider pool.
* Union recall barely moves                      -> retrieval isn't fixable
  by these sources; revisit the KB / dense encoder or the gold itself.

Run
---

::

    python -m cpt_rec.common.evaluation.candidate_union \\
        --npz   outputs/verifier/wholenote320_control/predictions/test/predictions.npz \\
        --gold  outputs/datasets/vumc/test_eval_sectioned.csv \\
        --bm25-index outputs/indices/note_bm25/bm25.pkl \\
        --neighbor-top-k 25 \\
        --llm-pred outputs/baselines/m3_zeroshot_frontier/predictions/test_gptchatlatest.csv \\
                   outputs/baselines/m4_rag_frontier/predictions/test_gptchatlatest.csv \\
        --by-category \\
        --out-json outputs/evaluation/candidate_union/test.json

``--notes`` defaults to ``--gold`` (the sectioned CSV carries both the note
text and the gold codes).  The KB pool depth is whatever ``--npz`` was saved
with, so for an honest ceiling raise the assembler's per-source top-k and
re-predict that NPZ first.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from cpt_rec.baselines.common import _ID_CANDIDATES, _pick_col
from cpt_rec.common.evaluation.code_bins import BINS, load_code_bins
from cpt_rec.common.evaluation.metrics import classify_code
from cpt_rec.common.evaluation.oracle import (
    _norm,
    load_gold,
    load_npz,
    per_bin_gold_recall,
)
from cpt_rec.common.preprocess.code_utils import pipe_split_codes

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source pool builders  (each returns {note_id: set[str]})
# ---------------------------------------------------------------------------

def kb_pools(npz: Path) -> Tuple[List[str], Dict[str, Set[str]]]:
    """The KB-description-only pool the scorer currently sees, per note."""
    note_ids, cands, _ = load_npz(npz)
    pools = {nid: set(pool) for nid, pool in zip(note_ids, cands)}
    return note_ids, pools


def neighbor_pools(
    notes_csv: Path,
    bm25_index_path: Path,
    max_top_k: int,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    note_ids: Optional[Set[str]] = None,
    query_chunk: int = 512,
) -> Dict[str, List[Tuple[str, Set[str]]]]:
    """For each query note return the ranked neighbour list as
    ``[(neighbor_note_id, neighbor_gold_set), ...]`` (descending score),
    truncated to ``max_top_k``.  Slicing this list gives any top-k ≤
    ``max_top_k`` without re-querying the index.

    ``note_ids`` (optional) restricts the query set to those note ids — pass it
    when only a subset of ``notes_csv`` will be used (e.g. ``--max-train-notes``
    smoke runs) so we don't pay the BM25 query for every row.  Scoring is
    vectorized via :meth:`TrainNoteBM25Index.batch_neighbors`.
    """
    from cpt_rec.baselines.bm25_index import TrainNoteBM25Index
    from cpt_rec.baselines.common import load_notes_for_prediction

    index = TrainNoteBM25Index.load(bm25_index_path)
    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    nids = [str(n) for n in df["note_id"]]
    texts = [str(t) for t in df["note_text"]]
    if note_ids is not None:
        wanted = {str(n) for n in note_ids}
        keep = [i for i, n in enumerate(nids) if n in wanted]
        nids = [nids[i] for i in keep]
        texts = [texts[i] for i in keep]

    LOGGER.info(
        "neighbor_pools: querying %d notes (top_k=%d) via vectorized BM25",
        len(nids), max_top_k,
    )
    ranked = index.batch_neighbors(texts, top_k=max_top_k, query_chunk=query_chunk)
    out: Dict[str, List[Tuple[str, Set[str]]]] = {}
    for nid, triples in zip(nids, ranked):
        out[nid] = [
            (nbr_id, {_norm(c) for c in code_set})
            for nbr_id, _score, code_set in triples
        ]
    return out


def neighbor_union_at_k(
    neighbors: Dict[str, List[Tuple[str, Set[str]]]], top_k: int
) -> Dict[str, Set[str]]:
    """Union of the top-k neighbours' gold sets, per note."""
    out: Dict[str, Set[str]] = {}
    for nid, ranked in neighbors.items():
        acc: Set[str] = set()
        for _nbr_id, code_set in ranked[:top_k]:
            acc |= code_set
        out[nid] = acc
    return out


def llm_pools(pred_csvs: Sequence[Path]) -> Dict[str, Set[str]]:
    """Union of LLM-prior candidate codes across one or more CSVs.

    Accepts either schema an LLM candidate-prior generator emits:
      * wide ``<out>.csv``         — pipe-separated ``pred_codes`` column
      * long ``<out>.sources.csv`` — one source-tagged ``code`` per row
    """
    out: Dict[str, Set[str]] = {}
    for pred_csv in pred_csvs:
        df = pd.read_csv(pred_csv, dtype=str)
        id_col = _pick_col(df, _ID_CANDIDATES, f"note-id (in {pred_csv})")
        if "pred_codes" in df.columns:            # wide <out>.csv
            for nid, codes in zip(df[id_col], df["pred_codes"].fillna("")):
                s = out.setdefault(str(nid).strip(), set())
                s |= {_norm(c) for c in pipe_split_codes(codes)}
        elif "code" in df.columns:                # long <out>.sources.csv
            for nid, code in zip(df[id_col], df["code"].fillna("")):
                code = str(code).strip()
                if code:
                    out.setdefault(str(nid).strip(), set()).add(_norm(code))
        else:
            raise ValueError(
                f"{pred_csv} has neither 'pred_codes' (wide) nor 'code' "
                f"(long .sources.csv) column; got {list(df.columns)}."
            )
    return out


# ---------------------------------------------------------------------------
# Recall over arbitrary per-note pools
# ---------------------------------------------------------------------------

def micro_recall(
    note_ids: Sequence[str],
    pools: Dict[str, Set[str]],
    gold: Dict[str, Set[str]],
) -> float:
    """micro recall = sum|gold ∩ pool| / sum|gold| over the given notes."""
    hit = tot = 0
    for nid in note_ids:
        g = gold.get(nid, set())
        if not g:
            continue
        hit += len(g & pools.get(nid, set()))
        tot += len(g)
    return hit / tot if tot else 0.0


def _union(*pool_dicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Per-note union across several {note_id: set} dicts."""
    keys: Set[str] = set()
    for d in pool_dicts:
        keys |= set(d.keys())
    out: Dict[str, Set[str]] = {}
    for k in keys:
        acc: Set[str] = set()
        for d in pool_dicts:
            acc |= d.get(k, set())
        out[k] = acc
    return out


def _mean_pool_size(note_ids: Sequence[str], pools: Dict[str, Set[str]]) -> float:
    if not note_ids:
        return 0.0
    return float(np.mean([len(pools.get(nid, set())) for nid in note_ids]))


# ---------------------------------------------------------------------------
# Per-strata retrieval ceiling  (head / torso / tail / unseen)
# ---------------------------------------------------------------------------

def recall_by_strata(
    note_ids: Sequence[str],
    combos: Sequence[Tuple[str, Dict[str, Set[str]]]],
    gold: Dict[str, Set[str]],
    code_bin: Dict[str, str],
) -> Tuple[Dict[str, int], List[Dict]]:
    """Full-pool recall per train-frequency bin for each source combination.

    Shows *which bin* each candidate source lifts: e.g. KB recall on ``tail``
    is near-floor while ``KB + llm-prior`` lifts it.  Per-code binning matches
    the eval harness, so these line up with ``oracle --by-strata`` and
    ``per_bin_metrics.csv``.

    Returns ``(n_gold_by_bin, rows)`` where each row is
    ``{"source": name, "head": r, "torso": r, "tail": r, "unseen": r}``.
    """
    n_gold_by_bin: Dict[str, int] = {}
    rows: List[Dict] = []
    for name, pool in combos:
        pb = per_bin_gold_recall(note_ids, pool, gold, code_bin)
        if not n_gold_by_bin:
            n_gold_by_bin = {b: pb[b]["n_gold"] for b in BINS}
        row: Dict = {"source": name}
        row.update({b: round(pb[b]["recall"], 4) for b in BINS})
        rows.append(row)
    return n_gold_by_bin, rows


def recovery_by_strata(
    note_ids: Sequence[str],
    kb: Dict[str, Set[str]],
    nbr: Dict[str, Set[str]],
    llm: Dict[str, Set[str]],
    union_extra: Dict[str, Set[str]],
    gold: Dict[str, Set[str]],
    code_bin: Dict[str, str],
) -> Dict[str, Dict[str, object]]:
    """Of gold unreachable from the KB pool, how many each source recovers, per bin.

    Mirrors :func:`recovery_by_category` but bins by training frequency so the
    long-tail story is explicit (tail/unseen is exactly where the KB pool fails
    and a neighbour / LLM-prior source must rescue the code).
    """
    miss = {b: 0 for b in BINS}
    rn = {b: 0 for b in BINS}
    rl = {b: 0 for b in BINS}
    ru = {b: 0 for b in BINS}
    for nid in note_ids:
        g = gold.get(nid, set())
        m = g - kb.get(nid, set())
        for c in m:
            b = code_bin.get(_norm(c), "unseen")
            if b not in miss:
                b = "unseen"
            miss[b] += 1
            if c in nbr.get(nid, set()):
                rn[b] += 1
            if c in llm.get(nid, set()):
                rl[b] += 1
            if c in union_extra.get(nid, set()):
                ru[b] += 1
    out: Dict[str, Dict[str, object]] = {}
    for b in BINS:
        if miss[b] == 0:
            continue
        out[b] = {
            "gold_missing_from_kb": miss[b],
            "recovered_by_neighbor": rn[b],
            "recovered_by_llm_prior": rl[b],
            "recovered_by_union": ru[b],
            "recovered_by_union_frac": round(ru[b] / miss[b], 4),
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    npz: Path,
    gold_csv: Path,
    notes_csv: Optional[Path],
    bm25_index: Optional[Path],
    neighbor_top_k: int,
    neighbor_sweep: Sequence[int],
    llm_pred: Sequence[Path],
    by_category: bool,
    train_stats: Optional[Path] = None,
    by_strata: bool = False,
) -> Dict:
    note_ids, kb = kb_pools(npz)
    gold = load_gold(gold_csv)
    notes_csv = notes_csv or gold_csv

    n_gold = sum(len(gold.get(nid, set())) for nid in note_ids)

    # ---- per-note source pools -------------------------------------------
    nbr_ranked: Optional[Dict[str, List[Tuple[str, Set[str]]]]] = None
    nbr: Dict[str, Set[str]] = {}
    if bm25_index is not None:
        max_k = max([neighbor_top_k, *neighbor_sweep]) if neighbor_sweep else neighbor_top_k
        nbr_ranked = neighbor_pools(notes_csv, bm25_index, max_top_k=max_k)
        nbr = neighbor_union_at_k(nbr_ranked, neighbor_top_k)

    llm: Dict[str, Set[str]] = {}
    if llm_pred:
        llm = llm_pools(llm_pred)

    # ---- recall for each source combination ------------------------------
    combos: List[Tuple[str, Dict[str, Set[str]]]] = [("KB", kb)]
    if nbr:
        combos.append((f"KB + neighbor@{neighbor_top_k}", _union(kb, nbr)))
    if llm:
        combos.append(("KB + llm-prior", _union(kb, llm)))
    if nbr and llm:
        combos.append((f"KB + neighbor@{neighbor_top_k} + llm-prior", _union(kb, nbr, llm)))

    kb_recall = micro_recall(note_ids, kb, gold)
    rows = []
    for name, pool in combos:
        rec = micro_recall(note_ids, pool, gold)
        rows.append({
            "source": name,
            "recall": round(rec, 4),
            "delta_vs_kb": round(rec - kb_recall, 4),
            "mean_pool_size": round(_mean_pool_size(note_ids, pool), 1),
        })

    # ---- recovery of currently-unreachable gold --------------------------
    # gold codes NOT in the KB pool, and which extra source recovers them.
    missing_total = 0
    rec_nbr = rec_llm = rec_union = 0
    union_extra = _union(nbr, llm) if (nbr or llm) else {}
    for nid in note_ids:
        g = gold.get(nid, set())
        miss = g - kb.get(nid, set())
        if not miss:
            continue
        missing_total += len(miss)
        rec_nbr += len(miss & nbr.get(nid, set()))
        rec_llm += len(miss & llm.get(nid, set()))
        rec_union += len(miss & union_extra.get(nid, set()))
    recovery = {
        "gold_missing_from_kb": missing_total,
        "recovered_by_neighbor": rec_nbr,
        "recovered_by_llm_prior": rec_llm,
        "recovered_by_union": rec_union,
        "still_unreachable": missing_total - rec_union,
        "recovered_by_neighbor_frac": round(rec_nbr / missing_total, 4) if missing_total else 0.0,
        "recovered_by_llm_frac": round(rec_llm / missing_total, 4) if missing_total else 0.0,
        "recovered_by_union_frac": round(rec_union / missing_total, 4) if missing_total else 0.0,
    }

    # ---- neighbour-depth sweep ------------------------------------------
    sweep_rows = []
    if nbr_ranked is not None and neighbor_sweep:
        for k in neighbor_sweep:
            nbr_k = neighbor_union_at_k(nbr_ranked, k)
            pool_k = _union(kb, nbr_k, llm) if llm else _union(kb, nbr_k)
            sweep_rows.append({
                "neighbor_top_k": k,
                "neighbor_only_recall": round(micro_recall(note_ids, nbr_k, gold), 4),
                "union_recall": round(micro_recall(note_ids, pool_k, gold), 4),
                "mean_pool_size": round(_mean_pool_size(note_ids, pool_k), 1),
            })

    result = {
        "inputs": {
            "npz": str(npz),
            "gold": str(gold_csv),
            "notes": str(notes_csv),
            "bm25_index": str(bm25_index) if bm25_index else None,
            "neighbor_top_k": neighbor_top_k,
            "llm_pred": [str(p) for p in llm_pred],
            "n_notes": len(note_ids),
            "n_gold": n_gold,
        },
        "kb_full_recall": round(kb_recall, 4),
        "by_source": rows,
        "recovery_of_unreachable_gold": recovery,
        "neighbor_depth_sweep": sweep_rows,
    }

    if by_category and (nbr or llm):
        cats = ("CPT_I", "CPT_II", "CPT_III", "PLA", "MAAA", "HCPCS")
        per_cat: Dict[str, Dict] = {}
        for cat in cats:
            miss_c = rec_nbr_c = rec_llm_c = rec_union_c = 0
            for nid in note_ids:
                g = {c for c in gold.get(nid, set()) if classify_code(c) == cat}
                miss = g - kb.get(nid, set())
                if not miss:
                    continue
                miss_c += len(miss)
                rec_nbr_c += len(miss & nbr.get(nid, set()))
                rec_llm_c += len(miss & llm.get(nid, set()))
                rec_union_c += len(miss & union_extra.get(nid, set()))
            if miss_c == 0:
                continue
            per_cat[cat] = {
                "gold_missing_from_kb": miss_c,
                "recovered_by_neighbor": rec_nbr_c,
                "recovered_by_llm_prior": rec_llm_c,
                "recovered_by_union": rec_union_c,
                "recovered_by_union_frac": round(rec_union_c / miss_c, 4),
            }
        result["recovery_by_category"] = per_cat

    if by_strata and train_stats is not None:
        code_bin = load_code_bins(str(train_stats))
        n_gold_by_bin, recall_rows = recall_by_strata(note_ids, combos, gold, code_bin)
        result["recall_by_strata"] = {
            "n_gold_by_bin": n_gold_by_bin,
            "by_source": recall_rows,
        }
        if nbr or llm:
            result["recovery_by_strata"] = recovery_by_strata(
                note_ids, kb, nbr, llm, union_extra, gold, code_bin
            )

    return result


def _print_report(r: Dict) -> None:
    inp = r["inputs"]
    print("=" * 72)
    print("CANDIDATE-SOURCE UNION  (retrieval-ceiling check)")
    print("=" * 72)
    print(f"Notes      : {inp['n_notes']:,}   gold codes: {inp['n_gold']:,}")
    print(f"KB NPZ     : {inp['npz']}")
    print(f"BM25 index : {inp['bm25_index']}")
    print(f"LLM preds  : {inp['llm_pred'] or '(none)'}")
    print("-" * 72)
    print("Full-pool recall by source   (recall | Δ vs KB | mean pool size)")
    for row in r["by_source"]:
        tag = "  <- current ceiling" if row["source"] == "KB" else (
            "  <- NEW ceiling" if row["source"].count("+") >= 2 else "")
        print(f"   {row['source']:<34s} {row['recall']:.4f} | "
              f"{row['delta_vs_kb']:+.4f} | {row['mean_pool_size']:>6.1f}{tag}")
    print("-" * 72)
    rc = r["recovery_of_unreachable_gold"]
    print(f"Gold codes unreachable from KB pool : {rc['gold_missing_from_kb']:,}")
    print(f"   recovered by neighbor  : {rc['recovered_by_neighbor']:>5}  "
          f"({rc['recovered_by_neighbor_frac']:.1%})")
    print(f"   recovered by llm-prior : {rc['recovered_by_llm_prior']:>5}  "
          f"({rc['recovered_by_llm_frac']:.1%})")
    print(f"   recovered by union     : {rc['recovered_by_union']:>5}  "
          f"({rc['recovered_by_union_frac']:.1%})")
    print(f"   still unreachable      : {rc['still_unreachable']:>5}")
    if r["neighbor_depth_sweep"]:
        print("-" * 72)
        print("Neighbour-depth sweep   (top_k | nbr-only recall | union recall | pool)")
        for s in r["neighbor_depth_sweep"]:
            print(f"   k={s['neighbor_top_k']:<3} : {s['neighbor_only_recall']:.4f} | "
                  f"{s['union_recall']:.4f} | {s['mean_pool_size']:>6.1f}")
    if "recovery_by_category" in r:
        print("-" * 72)
        print("Recovery by category  (missing | by-nbr | by-llm | by-union | union%)")
        for cat, cm in r["recovery_by_category"].items():
            print(f"   {cat:7s} : {cm['gold_missing_from_kb']:>5} | "
                  f"{cm['recovered_by_neighbor']:>5} | {cm['recovered_by_llm_prior']:>5} | "
                  f"{cm['recovered_by_union']:>5} | {cm['recovered_by_union_frac']:.1%}")
    if "recall_by_strata" in r:
        rbs = r["recall_by_strata"]
        ng = rbs["n_gold_by_bin"]
        print("-" * 72)
        print("Full-pool recall by train-frequency bin   (head / torso / tail / unseen)")
        print(f"   gold codes/bin : " + " | ".join(f"{b}={ng.get(b, 0)}" for b in BINS))
        for row in rbs["by_source"]:
            print(f"   {row['source']:<34s} " +
                  " | ".join(f"{row.get(b, 0.0):.4f}" for b in BINS))
    if "recovery_by_strata" in r:
        print("-" * 72)
        print("Recovery by train-freq bin  (missing | by-nbr | by-llm | by-union | union%)")
        for b, cm in r["recovery_by_strata"].items():
            print(f"   {b:7s} : {cm['gold_missing_from_kb']:>5} | "
                  f"{cm['recovered_by_neighbor']:>5} | {cm['recovered_by_llm_prior']:>5} | "
                  f"{cm['recovered_by_union']:>5} | {cm['recovered_by_union_frac']:.1%}")
    print("=" * 72)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Candidate-source union retrieval-ceiling check."
    )
    p.add_argument("--npz", required=True, type=Path, help="KB-pool test predictions.npz")
    p.add_argument("--gold", required=True, type=Path, help="test *_sectioned.csv")
    p.add_argument("--notes", type=Path, default=None,
                   help="Notes CSV for neighbour querying (default: --gold).")
    p.add_argument("--bm25-index", type=Path, default=None,
                   help="M1 train-note BM25 index (bm25.pkl) for the neighbour source.")
    p.add_argument("--neighbor-top-k", type=int, default=25,
                   help="Neighbour depth for the headline union row.")
    p.add_argument("--neighbor-sweep", default="5,10,25,50",
                   help="Comma-separated neighbour depths for the sweep table.")
    p.add_argument("--llm-pred", nargs="*", type=Path, default=[],
                   help="One or more GPT baseline predictions CSVs (M3/M4).")
    p.add_argument("--by-category", action="store_true")
    p.add_argument("--train-stats", type=Path, default=None,
                   help="code_frequency_stats.csv (head/torso/tail bins) for "
                        "the per-strata recall ceiling.")
    p.add_argument("--by-strata", action="store_true",
                   help="Per-bin (head/torso/tail/unseen) recall ceiling and "
                        "unreachable-gold recovery; needs --train-stats.")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    if args.by_strata and args.train_stats is None:
        p.error("--by-strata requires --train-stats")

    sweep = [int(x) for x in str(args.neighbor_sweep).split(",") if x.strip()]
    r = run(
        npz=args.npz,
        gold_csv=args.gold,
        notes_csv=args.notes,
        bm25_index=args.bm25_index,
        neighbor_top_k=args.neighbor_top_k,
        neighbor_sweep=sweep,
        llm_pred=args.llm_pred,
        by_category=args.by_category,
        train_stats=args.train_stats,
        by_strata=args.by_strata,
    )
    _print_report(r)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(r, f, indent=2, sort_keys=True)
        LOGGER.info("Wrote %s", args.out_json)


if __name__ == "__main__":
    main()
