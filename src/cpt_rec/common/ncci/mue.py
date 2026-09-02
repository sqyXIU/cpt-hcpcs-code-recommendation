# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Parse CMS NCCI Medically Unlikely Edits (MUE) files.

The practitioner-services MUE CSV has a multi-line copyright header
followed by a data header and rows::

    "HCPCS/\\nCPT Code", Practitioner Services MUE Values,
    MUE Adjudication Indicator, MUE Rationale

MUE Adjudication Indicator values:
    1 = Claim Line Edit
    2 = Date of Service Edit: Policy
    3 = Date of Service Edit: Clinical
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _find_header_row(path: Path) -> int:
    """
    Return the 0-based line index of the row containing the MUE column header.

    The header row is identified by the presence of ``CPT Code`` (case-insensitive)
    as the first field.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if re.search(r"CPT\s+Code", line, re.IGNORECASE):
                return i
    raise ValueError(f"Cannot find MUE header row in {path}")


def _iter_mue_rows(path: Path, header_row: int) -> Iterable[dict]:
    """Yield cleaned dicts from MUE CSV starting after *header_row*."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for _ in range(header_row + 1):
            next(fh)

        reader = csv.reader(fh)
        for parts in reader:
            if len(parts) < 3:
                continue
            code = parts[0].strip()
            if not code or not code[0].isalnum():
                continue

            try:
                max_units = int(parts[1].strip())
            except ValueError:
                continue

            adjudication_raw = parts[2].strip()
            # Extract leading digit from strings like "2 Date of Service Edit: Policy"
            mai_match = re.match(r"(\d)", adjudication_raw)
            mai = int(mai_match.group(1)) if mai_match else None

            rationale = parts[3].strip() if len(parts) > 3 else ""

            yield {
                "code": code,
                "max_units": max_units,
                "mue_adjudication_indicator": mai,
                "mue_rationale": rationale,
            }


def load_mue_file(path: Path | str) -> pd.DataFrame:
    """Load a single MUE CSV file into a DataFrame."""
    path = Path(path)
    header_row = _find_header_row(path)
    records = list(_iter_mue_rows(path, header_row))
    LOGGER.info("Loaded %d MUE rows from %s", len(records), path.name)
    return pd.DataFrame.from_records(records)


def build_mue_lookup(df: pd.DataFrame) -> dict[str, int]:
    """
    Build a {code: max_units} lookup dict from an MUE DataFrame.

    If a code appears multiple times (shouldn't happen), keep the minimum.
    """
    lookup: dict[str, int] = {}
    for row in df.itertuples(index=False):
        existing = lookup.get(row.code)
        if existing is None or row.max_units < existing:
            lookup[row.code] = row.max_units
    return lookup
