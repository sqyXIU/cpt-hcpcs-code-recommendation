#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Create a small, frozen evaluation sample from the normalized splits.

Rationale
---------
The full year-windowed splits (Train 2017-2023, Val 2024, Test 2025) are
unwieldy for a conference-paper baseline sweep:

* Clinical-Longformer on 330k train docs @ L=4096 is ~6 days on a single H100.
* GPT-4.1 on 50k test notes is thousands of dollars.

For comparable, reproducible numbers across **all** baselines we freeze a
much smaller evaluation sample and run every baseline on the same set.

Sampling strategy
-----------------
Within each split we stratify by procedure-date year and draw a random
subsample proportional to each year's share.  This preserves temporal
coverage (so e.g. 2023 isn't overrepresented in the train subsample just
because it has more notes) while remaining a simple random sample within
each stratum.

Outputs
-------
``<out_dir>/train_eval.csv``, ``val_eval.csv``, ``test_eval.csv`` — same
columns as the inputs, just fewer rows.

``<out_dir>/sample_manifest.json`` — metadata block with the seed, sizes,
per-year counts, and the list of sampled ``NOTE_ID``s per split.  Commit
this file so the sample is regeneratable.

Usage
-----
::

    cptrec-subsample-splits \\
        --splits-dir ./output/splits \\
        --out-dir    ./output/splits_eval \\
        --train-size 20000 \\
        --val-size   2000  \\
        --test-size  2000  \\
        --seed 42

Pass ``--include-drift`` (with optional ``--drift-size`` /
``--drift-name``) to also subsample the drift split — the default
3-way CLI leaves drift untouched so existing workflows keep working.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

_DATE_COL = "PROCEDURE_DATE"
_ID_COL = "NOTE_ID"


def _extract_year(series: pd.Series) -> pd.Series:
    """Pull the calendar year out of a mixed-format date column.

    Falls back to ``-1`` (string "UNK") when parsing fails so those rows
    still form a valid stratum rather than being dropped.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    years = parsed.dt.year.astype("Int64")
    return years.fillna(-1).astype(int).astype(str).replace("-1", "UNK")


def stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    stratum_col: str,
    seed: int,
) -> pd.DataFrame:
    """Proportionally allocate ``target_size`` rows across strata."""
    if len(df) <= target_size:
        LOGGER.info(
            "Input rows (%d) <= target (%d); returning full df.",
            len(df),
            target_size,
        )
        return df.reset_index(drop=True)

    counts = df[stratum_col].value_counts()
    total = counts.sum()

    # Proportional allocation, rounded.
    alloc = {s: int(round(target_size * c / total)) for s, c in counts.items()}

    # Cap per-stratum allocation at available rows.
    for s in alloc:
        alloc[s] = min(alloc[s], int(counts[s]))

    # Fix rounding so we land exactly on target_size (within +- a few).
    diff = target_size - sum(alloc.values())
    if diff > 0:
        # Distribute leftover to strata with spare capacity, largest first.
        spare = {s: int(counts[s]) - alloc[s] for s in alloc}
        for s, _ in sorted(spare.items(), key=lambda kv: -kv[1]):
            take = min(diff, spare[s])
            alloc[s] += take
            diff -= take
            if diff <= 0:
                break
    elif diff < 0:
        # Trim from largest allocations first.
        for s, _ in sorted(alloc.items(), key=lambda kv: -kv[1]):
            take = min(-diff, alloc[s])
            alloc[s] -= take
            diff += take
            if diff >= 0:
                break

    parts: List[pd.DataFrame] = []
    for s, n in alloc.items():
        if n <= 0:
            continue
        part = df.loc[df[stratum_col] == s].sample(
            n=n, random_state=seed, replace=False
        )
        parts.append(part)

    out = pd.concat(parts, axis=0).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    return out


def subsample_one_split(
    src_csv: Path,
    out_csv: Path,
    target_size: int,
    seed: int,
) -> Dict:
    LOGGER.info("Loading %s ...", src_csv)
    df = pd.read_csv(src_csv, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    if _ID_COL not in df.columns:
        raise ValueError(f"{src_csv} missing {_ID_COL} column")
    if _DATE_COL not in df.columns:
        raise ValueError(f"{src_csv} missing {_DATE_COL} column")

    df["_year"] = _extract_year(df[_DATE_COL])
    n_in = len(df)

    sampled = stratified_sample(
        df, target_size=target_size, stratum_col="_year", seed=seed
    )
    per_year_in = df["_year"].value_counts().to_dict()
    per_year_out = sampled["_year"].value_counts().to_dict()

    sampled = sampled.drop(columns="_year")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_csv, index=False)
    LOGGER.info(
        "Wrote %s (%d rows, from %d rows).",
        out_csv,
        len(sampled),
        n_in,
    )

    return {
        "source": str(src_csv),
        "output": str(out_csv),
        "rows_in": n_in,
        "rows_out": len(sampled),
        "target_size": target_size,
        "per_year_in": {str(k): int(v) for k, v in per_year_in.items()},
        "per_year_out": {str(k): int(v) for k, v in per_year_out.items()},
        "note_ids": sampled[_ID_COL].astype(str).tolist(),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Stratified subsample of normalized {train,val,test} CSVs "
                    "for cost-bounded baseline comparison."
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Folder containing train_normalized.csv / val_normalized.csv / "
             "test_normalized.csv.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Folder to write train_eval.csv / val_eval.csv / test_eval.csv "
             "and sample_manifest.json.",
    )
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--test-size", type=int, default=2000)
    p.add_argument(
        "--drift-size",
        type=int,
        default=2000,
        help="Rows to keep from the drift split (only used with "
             "--include-drift).  Default: 2000.",
    )
    p.add_argument(
        "--train-name",
        default="train_normalized.csv",
        help="Filename of the normalized train split under --splits-dir.",
    )
    p.add_argument("--val-name", default="val_normalized.csv")
    p.add_argument("--test-name", default="test_normalized.csv")
    p.add_argument(
        "--drift-name",
        default="drift_normalized.csv",
        help="Filename of the drift split under --splits-dir "
             "(only used with --include-drift).",
    )
    p.add_argument(
        "--include-drift",
        action="store_true",
        help="If set, also subsample the drift split into drift_eval.csv.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    splits_dir: Path = args.splits_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = [
        ("train", splits_dir / args.train_name,
         out_dir / "train_eval.csv", args.train_size),
        ("val", splits_dir / args.val_name,
         out_dir / "val_eval.csv", args.val_size),
        ("test", splits_dir / args.test_name,
         out_dir / "test_eval.csv", args.test_size),
    ]
    if args.include_drift:
        plan.append((
            "drift",
            splits_dir / args.drift_name,
            out_dir / "drift_eval.csv",
            args.drift_size,
        ))

    manifest: Dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "splits_dir": str(splits_dir),
        "out_dir": str(out_dir),
        "splits": {},
    }

    for name, src, dst, size in plan:
        if not src.exists():
            raise FileNotFoundError(
                f"Missing expected normalized split: {src}"
            )
        info = subsample_one_split(src, dst, target_size=size, seed=args.seed)
        manifest["splits"][name] = info

    manifest_path = out_dir / "sample_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("Wrote manifest -> %s", manifest_path)

    # Human-friendly summary table.
    print("\nSubsample summary")
    print("-----------------")
    print(f"{'split':<6} {'in':>10} {'out':>8} {'target':>8}")
    for name in ("train", "val", "test", "drift"):
        if name not in manifest["splits"]:
            continue
        s = manifest["splits"][name]
        print(
            f"{name:<6} {s['rows_in']:>10,} "
            f"{s['rows_out']:>8,} {s['target_size']:>8,}"
        )


if __name__ == "__main__":
    main()
