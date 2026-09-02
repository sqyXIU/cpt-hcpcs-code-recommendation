#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Stratified VALIDATION cohort builder (``cptrec-val-cohort``).

Model selection for the LLM-prior candidate generator is done on the
**validation** set (the test split stays frozen).  Running every candidate
model × mode over *all* of val would be slow and would also drown the
interesting, code-rare notes under a sea of easy single-CPT cases.  This
tool carves a small (default ~400-note) cohort out of
``val_eval_sectioned.csv`` that is *stratified* so the headline ΔR@K is
read on a balanced, decision-relevant slice rather than the head-heavy
natural distribution.

Stratification axes
-------------------

* **Frequency tier** — each gold code is binned ``head / torso / tail``
  by the training ``code_frequency_stats.csv``; a code unseen in training
  is ``unseen``.  A *note's* tier is the **rarest** tier among its gold
  codes (so any note touching a tail/unseen code is treated as rare).
* **Set size** — ``single`` (1 code), ``small`` (2–4), ``large`` (5+).
  Multi-code notes stress recall under a candidate budget the most.
* **Category presence** — CPT-I / CPT-II / CPT-III / PLA / MAAA / HCPCS
  (via :func:`evaluation.metrics.classify_code`).  Reported for balance
  auditing; the rare categories (CPT-III, HCPCS) are exactly where an
  LLM prior is expected to help over KB-description retrieval.

The cross of ``tier × set_size`` defines the sampling cells.  Allocation
is a floored, largest-remainder proportional split: every non-empty cell
gets at least ``--floor`` notes (capped at the cell size) so rare cells
are never starved, then the remainder is filled proportionally to reach
``--target``.

Optional **hard slice** — "proposed-fails-but-BM25-RAG-succeeds"
----------------------------------------------------------------

With ``--npz`` (the bi-encoder val ``predictions.npz``) and
``--bm25-index`` (M1 ``bm25.pkl``), the tool also flags notes whose gold
contains a code **missing from the KB-description pool but recovered by
the top-k BM25 train-note neighbours**.  These are precisely the notes
where the current retrieval ceiling fails but a non-KB source rescues the
code, so they are force-included (up to ``--hard-slice-cap``) — the
LLM-prior must at minimum match BM25 here.

Outputs (under ``--out-dir``)
-----------------------------
``cohort_sectioned.csv``  filtered rows of the input notes CSV (all
                          original columns preserved) — a drop-in
                          ``--gold`` / ``--notes`` for ``candidate_union``
                          and any downstream candidate generator.
``cohort_strata.csv``     per-note labels: ``note_id,n_gold,tier,
                          size_bucket,categories,in_hard_slice,reason``.
``cohort_report.json``    composition: per-tier / per-size / per-category
                          available-vs-selected counts, the allocation
                          table, hard-slice size, and the run config.

Run
---
::

    python3 -m cpt_rec.common.evaluation.strata \\
        --val-gold   outputs/datasets/vumc/val_eval_sectioned.csv \\
        --train-stats outputs/datasets/vumc/code_frequency_stats.csv \\
        --npz        outputs/verifier/wholenote320_control/predictions/val/predictions.npz \\
        --bm25-index outputs/indices/note_bm25/bm25.pkl \\
        --neighbor-top-k 25 \\
        --target 400 --floor 8 --seed 42 \\
        --out-dir outputs/evaluation/val_cohort
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from cpt_rec.baselines.common import _ID_CANDIDATES, _pick_col
from cpt_rec.common.evaluation.metrics import classify_code
from cpt_rec.common.evaluation.oracle import load_gold

LOGGER = logging.getLogger(__name__)

TIERS = ("head", "torso", "tail", "unseen")
RARITY = {"head": 0, "torso": 1, "tail": 2, "unseen": 3}
SIZE_BUCKETS = ("single", "small", "large")
CATEGORIES = ("CPT_I", "CPT_II", "CPT_III", "PLA", "MAAA", "HCPCS")


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def load_code_bins(stats_csv: Path) -> Dict[str, str]:
    """code -> train frequency bin (head/torso/tail) from the stats CSV."""
    df = pd.read_csv(stats_csv, dtype={"code": str})
    if "bin" not in df.columns or "code" not in df.columns:
        raise ValueError(
            f"{stats_csv} must have 'code' and 'bin' columns; "
            f"have {list(df.columns)}"
        )
    return {str(c).strip().upper(): str(b).strip() for c, b in zip(df["code"], df["bin"])}


def note_tier(gold: Set[str], code_bin: Dict[str, str]) -> str:
    """Rarest tier among the note's gold codes (unseen > tail > torso > head)."""
    if not gold:
        return "head"
    worst, worst_r = "head", -1
    for c in gold:
        b = code_bin.get(c, "unseen")
        r = RARITY.get(b, 3)
        if r > worst_r:
            worst, worst_r = b, r
    return worst


def size_bucket(n: int) -> str:
    if n <= 1:
        return "single"
    if n <= 4:
        return "small"
    return "large"


def compute_labels(
    gold_map: Dict[str, Set[str]], code_bin: Dict[str, str]
) -> Dict[str, Dict[str, object]]:
    labels: Dict[str, Dict[str, object]] = {}
    for nid, gold in gold_map.items():
        cats = sorted({classify_code(c) for c in gold} - {"UNKNOWN"})
        labels[nid] = {
            "note_id": nid,
            "n_gold": len(gold),
            "tier": note_tier(gold, code_bin),
            "size_bucket": size_bucket(len(gold)),
            "categories": cats,
        }
    return labels


# ---------------------------------------------------------------------------
# Hard slice — KB pool misses gold but BM25 neighbours recover it
# ---------------------------------------------------------------------------

def hard_slice_ids(
    npz: Path,
    notes_csv: Path,
    bm25_index: Path,
    gold_map: Dict[str, Set[str]],
    neighbor_top_k: int,
) -> Set[str]:
    """Note ids where (gold \\ KB-pool) ∩ neighbour-union@k is non-empty."""
    from cpt_rec.common.evaluation.candidate_union import (
        kb_pools,
        neighbor_pools,
        neighbor_union_at_k,
    )

    _, kb = kb_pools(npz)
    nbr_ranked = neighbor_pools(notes_csv, bm25_index, max_top_k=neighbor_top_k)
    nbr = neighbor_union_at_k(nbr_ranked, neighbor_top_k)

    hard: Set[str] = set()
    for nid, gold in gold_map.items():
        miss = gold - kb.get(nid, set())
        if miss and (miss & nbr.get(nid, set())):
            hard.add(nid)
    LOGGER.info(
        "Hard slice: %d/%d notes have a gold code missed by KB but "
        "recovered by BM25 neighbour@%d", len(hard), len(gold_map), neighbor_top_k,
    )
    return hard


# ---------------------------------------------------------------------------
# Allocation + selection
# ---------------------------------------------------------------------------

def allocate(cell_counts: Dict, target: int, floor: int) -> Dict:
    """Floored largest-remainder proportional allocation across cells.

    Every non-empty cell gets ``min(floor, size)`` first (coverage
    guarantee), then the remaining budget toward ``target`` is filled
    proportionally to each cell's leftover capacity.  If the floors alone
    already meet/exceed ``target`` the floors win (coverage > exact count).
    """
    cells = list(cell_counts)
    alloc = {c: min(cell_counts[c], floor) for c in cells}

    def total() -> int:
        return sum(alloc.values())

    remaining = target - total()
    if remaining <= 0:
        return alloc

    # ``cap`` = how many more notes each cell *could* still take after its floor.
    cap = {c: cell_counts[c] - alloc[c] for c in cells}
    total_cap = sum(cap.values())
    if total_cap <= 0:
        return alloc

    # Largest-remainder method. ``ideal`` is each cell's exact (fractional)
    # proportional share of the leftover budget; first hand out its FLOOR
    # (capped at the cell's remaining capacity)...
    ideal = {c: remaining * cap[c] / total_cap for c in cells}
    for c in cells:
        add = min(cap[c], int(ideal[c]))
        alloc[c] += add

    # ...then distribute the few still-unallocated notes one at a time to the
    # cells with the largest fractional remainder (the "largest remainder"),
    # skipping any cell already at capacity. ``guard`` bounds the loop in case
    # rounding leaves nothing further allocatable.
    left = target - total()
    order = sorted(cells, key=lambda c: ideal[c] - int(ideal[c]), reverse=True)
    guard = 0
    idx = 0
    while left > 0 and any(alloc[c] < cell_counts[c] for c in cells):
        c = order[idx % len(order)]
        if alloc[c] < cell_counts[c]:
            alloc[c] += 1
            left -= 1
        idx += 1
        guard += 1
        if guard > 10 * (len(cells) + 1) + target:
            break
    return alloc


def select_cohort(
    labels: Dict[str, Dict[str, object]],
    target: int,
    floor: int,
    seed: int,
    hard: Set[str],
) -> Tuple[Set[str], Dict[str, str], Dict]:
    """Return (selected_ids, reason_by_id, allocation_table)."""
    cells: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for nid, lab in labels.items():
        cells[(str(lab["tier"]), str(lab["size_bucket"]))].append(nid)
    cell_counts = {c: len(v) for c, v in cells.items()}
    alloc = allocate(cell_counts, target, floor)

    rng = random.Random(seed)
    selected: Set[str] = set()
    reason: Dict[str, str] = {}

    for cell, ids in cells.items():
        ids_sorted = sorted(ids)          # deterministic base order
        rng.shuffle(ids_sorted)           # seeded shuffle
        # Prefer hard-slice members within the cell (stable sort keeps the
        # shuffled order inside each group).
        ids_sorted.sort(key=lambda x: 0 if x in hard else 1)
        k = int(alloc.get(cell, 0))
        for nid in ids_sorted[:k]:
            selected.add(nid)
            reason[nid] = "stratified"

    # Force-include every hard-slice note (subject to the caller's cap,
    # which is applied before this function).
    for nid in hard:
        if nid in selected:
            reason[nid] = "stratified+hard"
        else:
            selected.add(nid)
            reason[nid] = "hard_bm25_recovers"

    alloc_table = {
        f"{tier}|{size}": {
            "available": cell_counts.get((tier, size), 0),
            "allocated": int(alloc.get((tier, size), 0)),
        }
        for tier in TIERS
        for size in SIZE_BUCKETS
        if (tier, size) in cell_counts
    }
    return selected, reason, alloc_table


# ---------------------------------------------------------------------------
# Reporting + I/O
# ---------------------------------------------------------------------------

def build_report(
    labels: Dict[str, Dict[str, object]],
    selected: Set[str],
    reason: Dict[str, str],
    hard: Set[str],
    alloc_table: Dict,
    config: Dict,
) -> Dict:
    def _tier_counts(ids) -> Dict[str, int]:
        c = {t: 0 for t in TIERS}
        for nid in ids:
            c[str(labels[nid]["tier"])] += 1
        return c

    def _size_counts(ids) -> Dict[str, int]:
        c = {s: 0 for s in SIZE_BUCKETS}
        for nid in ids:
            c[str(labels[nid]["size_bucket"])] += 1
        return c

    def _cat_counts(ids) -> Dict[str, int]:
        c = {cat: 0 for cat in CATEGORIES}
        for nid in ids:
            for cat in labels[nid]["categories"]:  # type: ignore[index]
                if cat in c:
                    c[cat] += 1
        return c

    all_ids = list(labels)
    sel_ids = sorted(selected)
    return {
        "config": config,
        "n_available": len(all_ids),
        "n_selected": len(sel_ids),
        "hard_slice": {
            "available": len(hard),
            "selected": len(hard & selected),
        },
        "by_tier": {
            t: {"available": _tier_counts(all_ids)[t],
                "selected": _tier_counts(sel_ids)[t]}
            for t in TIERS
        },
        "by_size_bucket": {
            s: {"available": _size_counts(all_ids)[s],
                "selected": _size_counts(sel_ids)[s]}
            for s in SIZE_BUCKETS
        },
        "by_category_presence": {
            cat: {"available": _cat_counts(all_ids)[cat],
                  "selected": _cat_counts(sel_ids)[cat]}
            for cat in CATEGORIES
        },
        "allocation": alloc_table,
        "reason_counts": dict(
            _count_reasons(reason, selected)
        ),
    }


def _count_reasons(reason: Dict[str, str], selected: Set[str]) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for nid in selected:
        c[reason.get(nid, "stratified")] += 1
    return c


def write_cohort_csv(notes_csv: Path, selected: Set[str], out_csv: Path) -> int:
    """Filtered rows of *notes_csv* (all columns preserved), input order."""
    df = pd.read_csv(notes_csv, dtype=str)
    id_col = _pick_col(df, _ID_CANDIDATES, "note-id")
    mask = df[id_col].astype(str).str.strip().isin(selected)
    sub = df[mask].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    LOGGER.info("Wrote %d cohort rows -> %s", len(sub), out_csv)
    return len(sub)


def write_strata_csv(
    labels: Dict[str, Dict[str, object]],
    selected: Set[str],
    reason: Dict[str, str],
    hard: Set[str],
    out_csv: Path,
) -> None:
    rows = []
    for nid in sorted(selected):
        lab = labels[nid]
        rows.append({
            "note_id": nid,
            "n_gold": lab["n_gold"],
            "tier": lab["tier"],
            "size_bucket": lab["size_bucket"],
            "categories": "|".join(lab["categories"]),  # type: ignore[arg-type]
            "in_hard_slice": int(nid in hard),
            "reason": reason.get(nid, "stratified"),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["note_id", "n_gold", "tier", "size_bucket",
                 "categories", "in_hard_slice", "reason"],
    ).to_csv(out_csv, index=False)
    LOGGER.info("Wrote %d strata rows -> %s", len(rows), out_csv)


def _print_report(r: Dict) -> None:
    print("=" * 70)
    print("STRATIFIED VALIDATION COHORT")
    print("=" * 70)
    print(f"Available notes : {r['n_available']:,}")
    print(f"Selected notes  : {r['n_selected']:,}  "
          f"(target {r['config']['target']}, floor {r['config']['floor']})")
    print("-" * 70)
    print("By tier      (available -> selected)")
    for t in TIERS:
        b = r["by_tier"][t]
        print(f"   {t:7s} : {b['available']:>5} -> {b['selected']:>4}")
    print("By set size  (available -> selected)")
    for s in SIZE_BUCKETS:
        b = r["by_size_bucket"][s]
        print(f"   {s:7s} : {b['available']:>5} -> {b['selected']:>4}")
    print("By category presence  (available -> selected)")
    for cat in CATEGORIES:
        b = r["by_category_presence"][cat]
        print(f"   {cat:7s} : {b['available']:>5} -> {b['selected']:>4}")
    hs = r["hard_slice"]
    print("-" * 70)
    print(f"Hard slice (KB-miss / BM25-recovers): {hs['available']} available, "
          f"{hs['selected']} in cohort")
    print(f"Selection reasons: {r['reason_counts']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    val_gold: Path,
    train_stats: Path,
    out_dir: Path,
    notes_csv: Optional[Path],
    target: int,
    floor: int,
    seed: int,
    npz: Optional[Path],
    bm25_index: Optional[Path],
    neighbor_top_k: int,
    hard_slice_cap: Optional[int],
) -> Dict:
    notes_csv = notes_csv or val_gold
    gold_map = load_gold(val_gold)
    code_bin = load_code_bins(train_stats)
    labels = compute_labels(gold_map, code_bin)
    LOGGER.info("Loaded %d val notes; %d unique training-binned codes.",
                len(labels), len(code_bin))

    hard: Set[str] = set()
    if npz is not None and bm25_index is not None:
        hard = hard_slice_ids(npz, notes_csv, bm25_index, gold_map, neighbor_top_k)
        if hard_slice_cap is not None and len(hard) > hard_slice_cap:
            rng = random.Random(seed)
            hard = set(rng.sample(sorted(hard), hard_slice_cap))
            LOGGER.info("Capped hard slice to %d notes.", hard_slice_cap)
    elif npz is not None or bm25_index is not None:
        LOGGER.warning(
            "Hard slice needs BOTH --npz and --bm25-index; skipping it.")

    selected, reason, alloc_table = select_cohort(
        labels, target=target, floor=floor, seed=seed, hard=hard
    )

    config = {
        "val_gold": str(val_gold),
        "notes_csv": str(notes_csv),
        "train_stats": str(train_stats),
        "npz": str(npz) if npz else None,
        "bm25_index": str(bm25_index) if bm25_index else None,
        "neighbor_top_k": neighbor_top_k,
        "target": target,
        "floor": floor,
        "seed": seed,
        "hard_slice_cap": hard_slice_cap,
    }
    report = build_report(labels, selected, reason, hard, alloc_table, config)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_cohort_csv(notes_csv, selected, out_dir / "cohort_sectioned.csv")
    write_strata_csv(labels, selected, reason, hard, out_dir / "cohort_strata.csv")
    with open(out_dir / "cohort_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    LOGGER.info("Wrote cohort report -> %s", out_dir / "cohort_report.json")

    _print_report(report)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Stratified validation cohort builder.")
    p.add_argument("--val-gold", required=True, type=Path,
                   help="val_eval_sectioned.csv (note text + proc_codes gold).")
    p.add_argument("--train-stats", required=True, type=Path,
                   help="code_frequency_stats.csv (head/torso/tail bins).")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--notes", type=Path, default=None,
                   help="Notes CSV for neighbour querying (default: --val-gold).")
    p.add_argument("--target", type=int, default=400,
                   help="Approximate cohort size to fill toward.")
    p.add_argument("--floor", type=int, default=8,
                   help="Minimum notes per non-empty (tier x size) cell.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--npz", type=Path, default=None,
                   help="bi-encoder val predictions.npz for the hard slice.")
    p.add_argument("--bm25-index", type=Path, default=None,
                   help="M1 bm25.pkl for the hard slice.")
    p.add_argument("--neighbor-top-k", type=int, default=25)
    p.add_argument("--hard-slice-cap", type=int, default=None,
                   help="Cap the number of force-included hard-slice notes.")
    args = p.parse_args()

    run(
        val_gold=args.val_gold,
        train_stats=args.train_stats,
        out_dir=args.out_dir,
        notes_csv=args.notes,
        target=args.target,
        floor=args.floor,
        seed=args.seed,
        npz=args.npz,
        bm25_index=args.bm25_index,
        neighbor_top_k=args.neighbor_top_k,
        hard_slice_cap=args.hard_slice_cap,
    )


if __name__ == "__main__":
    main()
