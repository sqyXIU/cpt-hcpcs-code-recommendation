#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
NCCI-aware constrained decoding for scored predictions.

Input: per-note ``(candidate_codes, probs)`` from
``cptrec-verifier-predict`` (saved to ``--out-npz``), or any NPZ with the
same layout.  Output: a new predictions CSV where each code set satisfies
the machine-checkable NCCI constraints:

* No **hard-PTP** incompatible pairs (CCMI ∈ {0, 9}).
* Every Type-1 **add-on code** has at least one acceptable primary in
  the set.
* **CCMI=1** (modifier-contingent) pairs are flagged but not dropped —
  the primary task is modifier-agnostic (see README §2.1).
* **MUE** checks are skipped (we predict code presence, not units).

Decoding strategy
-----------------

Greedy with repair:

1. Sort codes by descending probability.
2. For each code in order:

   * If adding it would create a hard-PTP violation with any code
     already selected, skip it.
   * If the code is an add-on whose primary set is disjoint from the
     current selection, **hold** the code and revisit after the full
     pass (a later-arriving primary may make it valid).

3. Re-visit held add-on codes once: add the ones whose primaries are now
   present.

4. Emit the final set plus a ``decode_trace`` per note describing hard
   violations averted, modifier-contingent flags, and repaired add-ons.

CLI
---

::

    cptrec-constrained-decode \\
        --scores outputs/verifier/wholenote320_control/predictions/test/predictions.npz \\
        --ncci-dir data/ncci \\
        --threshold 0.5

When ``--out`` is omitted the decoded CSV lands as
``<scores parent>/predictions_decoded.csv``.

Any predictions CSV with ``pred_codes`` + ``pred_scores`` columns can
also be decoded directly (see ``--from-csv``).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from cpt_rec.baselines.common import Prediction, write_predictions
from cpt_rec.common.ncci.rule_checker import NCCIRuleChecker

LOGGER = logging.getLogger(__name__)


@dataclass
class DecodeTrace:
    """Per-note record of what constrained decoding did."""

    note_id: str
    kept_codes: List[str] = field(default_factory=list)
    kept_scores: List[float] = field(default_factory=list)
    dropped_hard_ptp: List[Tuple[str, str, int]] = field(default_factory=list)
    modifier_contingent: List[Tuple[str, str, int]] = field(default_factory=list)
    dropped_orphan_aoc: List[str] = field(default_factory=list)
    repaired_aoc: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core decoder
# ---------------------------------------------------------------------------

def decode_one(
    note_id: str,
    candidate_codes: Sequence[str],
    probs: Sequence[float],
    checker: NCCIRuleChecker,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> DecodeTrace:
    """
    Run greedy constrained decoding for one note.  Returns a
    :class:`DecodeTrace` with the final code set and bookkeeping.
    """
    # Only consider codes above threshold (or top-k by prob).
    pairs: List[Tuple[str, float]] = [
        (c, float(p)) for c, p in zip(candidate_codes, probs)
    ]
    pairs.sort(key=lambda cp: -cp[1])
    if top_k is not None and top_k > 0:
        pairs = pairs[:top_k]
    else:
        pairs = [(c, p) for c, p in pairs if p >= threshold]

    trace = DecodeTrace(note_id=str(note_id))
    current: List[str] = []
    current_scores: List[float] = []
    held_aoc: List[Tuple[str, float]] = []

    def _hard_conflict(new_code: str, current_set: Set[str]) -> Optional[Tuple[str, str, int]]:
        for existing in current_set:
            for c1, c2 in [(new_code, existing), (existing, new_code)]:
                ccmi = checker.get_ptp_ccmi(c1, c2)
                if ccmi is None:
                    continue
                if ccmi in (0, 9):
                    return (c1, c2, ccmi)
        return None

    def _contingent_flag(new_code: str, current_set: Set[str]):
        for existing in current_set:
            for c1, c2 in [(new_code, existing), (existing, new_code)]:
                ccmi = checker.get_ptp_ccmi(c1, c2)
                if ccmi == 1:
                    trace.modifier_contingent.append((c1, c2, ccmi))

    def _is_typ1_aoc(code: str) -> bool:
        return checker.get_acceptable_primaries(code) is not None

    def _primary_ok(addon: str, current_set: Set[str]) -> bool:
        req = checker.get_acceptable_primaries(addon)
        return bool(req and (req & current_set))

    # Pass 1: greedy add in probability order; hold orphaned Type-1 AOCs.
    for code, prob in pairs:
        current_set = set(current)
        conflict = _hard_conflict(code, current_set)
        if conflict is not None:
            trace.dropped_hard_ptp.append(conflict)
            continue

        if _is_typ1_aoc(code) and not _primary_ok(code, current_set):
            held_aoc.append((code, prob))
            continue

        _contingent_flag(code, current_set)
        current.append(code)
        current_scores.append(prob)

    # Pass 2: revisit held AOCs in case their primaries showed up later.
    for code, prob in held_aoc:
        current_set = set(current)
        if not _primary_ok(code, current_set):
            trace.dropped_orphan_aoc.append(code)
            continue
        conflict = _hard_conflict(code, current_set)
        if conflict is not None:
            trace.dropped_hard_ptp.append(conflict)
            continue
        _contingent_flag(code, current_set)
        current.append(code)
        current_scores.append(prob)
        trace.repaired_aoc.append(code)

    trace.kept_codes = current
    trace.kept_scores = current_scores
    return trace


# ---------------------------------------------------------------------------
# NPZ / CSV -> predictions
# ---------------------------------------------------------------------------

def decode_from_npz(
    scores_npz: Path,
    ncci_dir: Path,
    out_csv: Path,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    trace_json: Optional[Path] = None,
) -> int:
    dat = np.load(scores_npz, allow_pickle=True)
    note_ids = list(dat["note_ids"])
    all_codes = list(dat["candidate_codes"])
    all_probs = list(dat["probs"])

    checker = NCCIRuleChecker.from_data_dir(ncci_dir)
    LOGGER.info(
        "Loaded NCCI: %d PTP pairs, %d add-on codes, %d MUE entries",
        checker.n_ptp_pairs, checker.n_addon_codes, checker.n_mue_codes,
    )

    predictions: List[Prediction] = []
    traces: List[DecodeTrace] = []
    for nid, cands, probs in zip(note_ids, all_codes, all_probs):
        t = decode_one(
            note_id=str(nid),
            candidate_codes=list(cands),
            probs=list(probs),
            checker=checker,
            threshold=threshold,
            top_k=top_k,
        )
        traces.append(t)
        predictions.append(
            Prediction(note_id=t.note_id, codes=t.kept_codes, scores=t.kept_scores)
        )

    write_predictions(predictions, out_csv, include_scores=True)

    if trace_json is not None:
        trace_json = Path(trace_json)
        trace_json.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_json, "w") as f:
            json.dump([asdict(t) for t in traces], f, indent=2, sort_keys=True)
        LOGGER.info("Wrote %d decode traces -> %s", len(traces), trace_json)
    return len(predictions)


def decode_from_csv(
    pred_csv: Path,
    ncci_dir: Path,
    out_csv: Path,
    threshold: float = 0.0,
    top_k: Optional[int] = None,
    trace_json: Optional[Path] = None,
) -> int:
    """
    Decode from an existing predictions CSV with ``pred_codes`` +
    ``pred_scores`` columns.  Uses the *existing* thresholded set as the
    candidate pool (per-note), re-ordered by score.
    """
    df = pd.read_csv(pred_csv, dtype=str)
    if "pred_codes" not in df.columns or "pred_scores" not in df.columns:
        raise ValueError(
            "decode_from_csv requires pred_codes AND pred_scores columns."
        )
    id_col = "note_id" if "note_id" in df.columns else df.columns[0]

    checker = NCCIRuleChecker.from_data_dir(ncci_dir)
    predictions: List[Prediction] = []
    traces: List[DecodeTrace] = []
    for _, row in df.iterrows():
        nid = str(row[id_col])
        cands = [c.strip().upper() for c in str(row["pred_codes"] or "").split("|") if c.strip()]
        try:
            probs = [float(x) for x in str(row["pred_scores"] or "").split("|") if x.strip()]
        except ValueError:
            probs = [1.0] * len(cands)
        if len(probs) != len(cands):
            probs = [1.0] * len(cands)

        t = decode_one(nid, cands, probs, checker, threshold=threshold, top_k=top_k)
        traces.append(t)
        predictions.append(
            Prediction(note_id=t.note_id, codes=t.kept_codes, scores=t.kept_scores)
        )

    write_predictions(predictions, out_csv, include_scores=True)

    if trace_json is not None:
        trace_json = Path(trace_json)
        trace_json.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_json, "w") as f:
            json.dump([asdict(t) for t in traces], f, indent=2, sort_keys=True)
    return len(predictions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NCCI-aware constrained decoding.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scores", type=Path, help="NPZ from cptrec-verifier-predict --out-npz.")
    src.add_argument(
        "--from-csv", dest="from_csv", type=Path,
        help="Existing predictions CSV to re-decode.",
    )
    p.add_argument("--ncci-dir", required=True, type=Path)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Decoded predictions CSV. Defaults to a sibling "
            "predictions_decoded.csv next to --scores (or --from-csv)."
        ),
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--trace-json", type=Path, default=None)
    return p.parse_args()


def _default_decoded_path(source: Path) -> Path:
    """Sibling 'predictions_decoded.csv' in the source's directory."""
    return Path(source).parent / "predictions_decoded.csv"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    source = args.scores if args.scores is not None else args.from_csv
    out_csv = args.out if args.out is not None else _default_decoded_path(source)
    if args.scores is not None:
        decode_from_npz(
            scores_npz=args.scores,
            ncci_dir=args.ncci_dir,
            out_csv=out_csv,
            threshold=args.threshold,
            top_k=args.top_k,
            trace_json=args.trace_json,
        )
    else:
        decode_from_csv(
            pred_csv=args.from_csv,
            ncci_dir=args.ncci_dir,
            out_csv=out_csv,
            threshold=args.threshold,
            top_k=args.top_k,
            trace_json=args.trace_json,
        )


if __name__ == "__main__":
    main()
