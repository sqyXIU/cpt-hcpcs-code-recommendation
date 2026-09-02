# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Parse CMS NCCI Procedure-to-Procedure (PTP) edit files.

CMS publishes PTP edits as tab-delimited TXT files split across four parts
(f1–f4), each sharing the same format:

    6 header lines (copyright, column legend), then data rows:
    Column 1 <tab> Column 2 <tab> pre-1996 flag <tab> Effective Date <tab>
    Deletion Date <tab> Modifier Indicator (CCMI) <tab> PTP Edit Rationale

Modifier Indicator values:
    0 = modifier not allowed  (hard incompatibility in our framework)
    1 = modifier allowed       (modifier-contingent, flagged but not hard-invalid)
    9 = not applicable         (hard incompatibility in our framework)

Dates are stored as YYYYMMDD strings; deletion date ``*`` means still active.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

PTP_HEADER_LINES = 6

CCMI_HARD: Set[int] = {0, 9}
CCMI_MODIFIER_CONTINGENT: Set[int] = {1}


def _parse_date(s: str) -> Optional[date]:
    s = s.strip()
    if not s or s == "*":
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None


def _iter_ptp_txt(path: Path) -> Iterable[dict]:
    """Yield one dict per data row from a single PTP TXT file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i < PTP_HEADER_LINES:
                continue
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue

            col1 = parts[0].strip()
            col2 = parts[1].strip()
            if not col1 or not col2:
                continue

            effective = _parse_date(parts[3])
            deletion = _parse_date(parts[4])
            try:
                ccmi = int(parts[5].strip())
            except ValueError:
                continue
            rationale = parts[6].strip() if len(parts) > 6 else ""

            yield {
                "col1": col1,
                "col2": col2,
                "effective": effective,
                "deletion": deletion,
                "ccmi": ccmi,
                "rationale": rationale,
            }


def load_ptp_file(path: Path | str) -> pd.DataFrame:
    """Load a single PTP TXT file into a DataFrame."""
    path = Path(path)
    records = list(_iter_ptp_txt(path))
    LOGGER.info("Loaded %d PTP rows from %s", len(records), path.name)
    return pd.DataFrame.from_records(records)


def load_ptp_dir(ptp_dir: Path | str) -> pd.DataFrame:
    """
    Load all PTP TXT files under *ptp_dir* (recursively) and concatenate.

    Expects the CMS layout with subfolders ``ccipra-v*-f1/`` through ``f4/``,
    each containing a ``.TXT`` or ``.txt`` file.
    """
    ptp_dir = Path(ptp_dir)
    frames = []
    for p in sorted(ptp_dir.rglob("*.TXT")) + sorted(ptp_dir.rglob("*.txt")):
        if p.suffix.lower() == ".txt":
            frames.append(load_ptp_file(p))

    if not frames:
        raise FileNotFoundError(f"No PTP .TXT files found under {ptp_dir}")

    # Deduplicate in case .TXT and .txt point to the same file
    seen_names: set[str] = set()
    unique: list[pd.DataFrame] = []
    for p, df in zip(
        sorted(ptp_dir.rglob("*.TXT")) + sorted(ptp_dir.rglob("*.txt")), frames
    ):
        if p.name.lower() not in seen_names:
            seen_names.add(p.name.lower())
            unique.append(df)

    df = pd.concat(unique, ignore_index=True)
    LOGGER.info("Total PTP rows after merging all files: %d", len(df))
    return df


def filter_active_ptp(
    df: pd.DataFrame,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """
    Keep only PTP edits that are active on *reference_date* (default: today).

    Active means effective <= reference_date AND (deletion is null OR deletion >= reference_date).
    """
    if reference_date is None:
        reference_date = date.today()

    mask_eff = df["effective"].isna() | (df["effective"] <= reference_date)
    mask_del = df["deletion"].isna() | (df["deletion"] >= reference_date)
    return df.loc[mask_eff & mask_del].copy()


def build_ptp_lookup(
    df: pd.DataFrame,
) -> Tuple[dict[Tuple[str, str], int], dict[Tuple[str, str], str]]:
    """
    Build fast lookup dicts from a PTP DataFrame.

    Returns:
        pair_to_ccmi: {(col1, col2): ccmi}  — if a pair appears multiple times,
                      keep the row with the smallest (most restrictive) CCMI.
        pair_to_rationale: {(col1, col2): rationale}
    """
    pair_to_ccmi: dict[Tuple[str, str], int] = {}
    pair_to_rationale: dict[Tuple[str, str], str] = {}

    for row in df.itertuples(index=False):
        key = (row.col1, row.col2)
        existing = pair_to_ccmi.get(key)
        if existing is None or row.ccmi < existing:
            pair_to_ccmi[key] = row.ccmi
            pair_to_rationale[key] = row.rationale

    return pair_to_ccmi, pair_to_rationale
