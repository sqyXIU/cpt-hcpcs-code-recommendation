#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Split OP report CSVs into train/val/test by PROCEDURE_DATE, optionally plus a
drift holdout window.

Primary windows
---------------
    train : 2017-11-01 — 2023-12-31
    val   : 2024-01-01 — 2024-12-31
    test  : 2025-01-01 — 2025-12-31

Optional drift window (``--include_drift``)
-------------------------------------------
    drift : 2026-01-01 — 2026-02-05

Input CSV schema (all files share columns)
------------------------------------------
    PAT_MRN_ID, ENCOUNTER_CSN_ID, NOTE_ID, PROCEDURE_DATE, NOTE_TIME,
    NOTE_TEXT, CPT_CODES
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from cpt_rec.common.constants import REQUIRED_COLS


@dataclass(frozen=True)
class DateWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).normalize()


PRIMARY_WINDOWS = {
    "train": DateWindow("train", _ts("2017-11-01"), _ts("2023-12-31")),
    "val": DateWindow("val", _ts("2024-01-01"), _ts("2024-12-31")),
    "test": DateWindow("test", _ts("2025-01-01"), _ts("2025-12-31")),
}

DRIFT_WINDOW = DateWindow("drift", _ts("2026-01-01"), _ts("2026-02-05"))


def iter_csv_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.rglob("*.csv")):
        if p.is_file():
            yield p


def ensure_required_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def parse_procedure_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.normalize()


def in_window(dt: pd.Series, w: DateWindow) -> pd.Series:
    return (dt >= w.start) & (dt <= w.end)


def append_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=write_header, index=False)


def read_in_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    # Keep IDs as strings to preserve leading zeros
    dtype = {
        "PAT_MRN_ID": "string",
        "ENCOUNTER_CSN_ID": "string",
        "NOTE_ID": "string",
        "PROCEDURE_DATE": "string",
        "NOTE_TIME": "string",
        "NOTE_TEXT": "string",
        "CPT_CODES": "string",
    }
    for chunk in pd.read_csv(path, dtype=dtype, chunksize=chunksize):
        # Defensive: strip whitespace from headers
        chunk.columns = [str(c).strip() for c in chunk.columns]
        yield chunk


def split_and_write(
    files: Iterable[Path],
    out_dir: Path,
    chunksize: int,
    include_drift: bool,
) -> Dict[str, int]:
    """
    Writes:
      - train.csv, val.csv, test.csv
      - optionally drift.csv

    Returns row counts per output.
    """
    counts: Dict[str, int] = {}

    def bump(name: str, n: int) -> None:
        counts[name] = counts.get(name, 0) + int(n)

    out_paths = {
        "train": out_dir / "train.csv",
        "val": out_dir / "val.csv",
        "test": out_dir / "test.csv",
    }
    if include_drift:
        out_paths["drift"] = out_dir / "drift.csv"

    for f in files:
        for chunk in read_in_chunks(f, chunksize):
            ensure_required_columns(chunk, f)

            proc_dt = parse_procedure_date(chunk["PROCEDURE_DATE"])
            chunk = chunk.copy()
            chunk["_PROC_DT"] = proc_dt

            # Primary splits
            for split_name in ("train", "val", "test"):
                w = PRIMARY_WINDOWS[split_name]
                m = in_window(chunk["_PROC_DT"], w)
                if m.any():
                    df_out = chunk.loc[m.to_numpy(), REQUIRED_COLS]
                    append_csv(df_out, out_paths[split_name])
                    bump(split_name, len(df_out))

            # Optional drift holdout
            if include_drift:
                m_drift = in_window(chunk["_PROC_DT"], DRIFT_WINDOW)
                if m_drift.any():
                    df_drift_out = chunk.loc[m_drift.to_numpy(), REQUIRED_COLS]
                    append_csv(df_drift_out, out_paths["drift"])
                    bump("drift", len(df_drift_out))

    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=str,
        default="./op_reports_all_years",
        help="Folder containing CSVs (searched recursively).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./op_reports_splits",
        help="Output folder for split CSVs.",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk when streaming CSVs.",
    )
    ap.add_argument(
        "--include_drift",
        action="store_true",
        help="If set, also write 2026-01-01—2026-02-05 drift holdout.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    files = list(iter_csv_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {input_dir.resolve()}")

    print(f"Found {len(files)} CSVs under {input_dir.resolve()}")

    counts = split_and_write(
        files=files,
        out_dir=out_dir,
        chunksize=args.chunksize,
        include_drift=bool(args.include_drift),
    )

    print("\nWrote outputs to:", out_dir.resolve())
    for k in sorted(counts.keys()):
        print(f"  {k:16s} : {counts[k]:,} rows")


if __name__ == "__main__":
    main()
