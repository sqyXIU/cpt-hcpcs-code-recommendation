# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Parse CMS NCCI Add-On Code (AOC) edit files.

The AOC flat file (``AOC_*-F-MCR.txt``) uses a fixed-width format with
97 usable characters per line (plus ``\\r\\n``):

    [0:6]   add-on code          (e.g. ``10071U``)
    [6:13]  add-on deletion date (YYYYDDD Julian, or spaces if active)
    [13:18] primary code         (e.g. ``0070U``, or ``CCCCC`` for contractor-defined)
    [18:25] primary deletion date(YYYYDDD Julian, or spaces if active)
    [25:32] effective date       (YYYYDDD Julian)
    [32:39] deletion date        (YYYYDDD Julian, or spaces if active)
    [39:97] description / note

AOC types (per CMS definitions):
    Type 1 — add-on has a specific list of acceptable primary codes
    Type 3 — primary = ``CCCCC`` (contractor-defined primaries; weaker constraint)

Type 2 (``ZZZZZ`` = all CPT primaries) does not appear in this dataset.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Set

import pandas as pd

LOGGER = logging.getLogger(__name__)

CONTRACTOR_DEFINED_PRIMARY = "CCCCC"


def _julian_to_date(s: str) -> Optional[date]:
    """Convert a YYYYDDD Julian date string to a Python date, or None."""
    s = s.strip()
    if not s:
        return None
    try:
        year = int(s[:4])
        day_of_year = int(s[4:])
        return date(year, 1, 1) + timedelta(days=day_of_year - 1)
    except (ValueError, IndexError):
        return None


def _iter_aoc_lines(path: Path) -> Iterable[dict]:
    """Yield one dict per line from the AOC flat file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.rstrip("\r\n")
            if len(line) < 32:
                LOGGER.debug("Skipping short line %d (len=%d)", line_no, len(line))
                continue

            addon_code = line[0:6].strip()
            addon_deletion = _julian_to_date(line[6:13])
            primary_code = line[13:18].strip()
            primary_deletion = _julian_to_date(line[18:25])
            effective = _julian_to_date(line[25:32])
            deletion = _julian_to_date(line[32:39]) if len(line) >= 39 else None
            description = line[39:].strip() if len(line) > 39 else ""

            if not addon_code or not primary_code:
                continue

            aoc_type: int
            if primary_code == CONTRACTOR_DEFINED_PRIMARY:
                aoc_type = 3
            else:
                aoc_type = 1

            yield {
                "addon_code": addon_code,
                "addon_deletion": addon_deletion,
                "primary_code": primary_code,
                "primary_deletion": primary_deletion,
                "effective": effective,
                "deletion": deletion,
                "aoc_type": aoc_type,
                "description": description,
            }


def load_aoc_file(path: Path | str) -> pd.DataFrame:
    """Load a single AOC flat file into a DataFrame."""
    path = Path(path)
    records = list(_iter_aoc_lines(path))
    LOGGER.info("Loaded %d AOC rows from %s", len(records), path.name)
    return pd.DataFrame.from_records(records)


def filter_active_aoc(
    df: pd.DataFrame,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """
    Keep only AOC rows that are active on *reference_date*.

    A row is active when:
      - effective <= reference_date
      - deletion is null OR deletion >= reference_date
      - addon_deletion is null OR addon_deletion >= reference_date
      - primary_deletion is null OR primary_deletion >= reference_date
    """
    if reference_date is None:
        reference_date = date.today()

    m = (
        (df["effective"].isna() | (df["effective"] <= reference_date))
        & (df["deletion"].isna() | (df["deletion"] >= reference_date))
        & (df["addon_deletion"].isna() | (df["addon_deletion"] >= reference_date))
        & (df["primary_deletion"].isna() | (df["primary_deletion"] >= reference_date))
    )
    return df.loc[m].copy()


def build_aoc_lookup(
    df: pd.DataFrame,
) -> tuple[dict[str, Set[str]], Set[str]]:
    """
    Build lookup structures from an AOC DataFrame.

    Returns:
        addon_to_primaries: {addon_code: {primary_code, ...}}
            For Type 1 entries only (specific primary codes).
        contractor_defined_addons: set of addon codes whose only constraint
            is contractor-defined (Type 3 / ``CCCCC``).
    """
    addon_to_primaries: dict[str, Set[str]] = defaultdict(set)
    contractor_defined: Set[str] = set()

    for row in df.itertuples(index=False):
        if row.aoc_type == 3:
            contractor_defined.add(row.addon_code)
        else:
            addon_to_primaries[row.addon_code].add(row.primary_code)

    # Convert defaultdict to regular dict
    addon_to_primaries = dict(addon_to_primaries)
    return addon_to_primaries, contractor_defined
