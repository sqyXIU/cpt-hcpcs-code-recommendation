#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Unit tests for :mod:`cpt_rec.common.knowledge.code_history`.

Each test builds a tiny ``code_changes.csv`` / ``deleted_codes.csv``
pair in ``tmp_path`` so the tests don't depend on the real 19k-row
artifact checked into ``data/``.  Run with:

    pytest tests/test_code_history.py -q
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cpt_rec.common.knowledge.code_history import (
    CodeEvent,
    CodeHistory,
    _extract_codes,
)


# ---------------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------------

CHANGES_COLS = [
    "code_system", "year", "category", "change_type", "code",
    "description", "advice", "crosswalk_code", "new_descriptor",
]
DELETED_COLS = [
    "code", "description", "deleted_date", "substitute_code",
    "archived_info", "code_system",
]


def _write_csvs(tmp: Path, changes_rows, deleted_rows) -> tuple[Path, Path]:
    """Materialize two DataFrames as CSVs in *tmp* and return their paths."""
    cdf = pd.DataFrame(changes_rows, columns=CHANGES_COLS)
    ddf = pd.DataFrame(deleted_rows, columns=DELETED_COLS)
    c_path = tmp / "code_changes.csv"
    d_path = tmp / "deleted_codes.csv"
    cdf.to_csv(c_path, index=False)
    ddf.to_csv(d_path, index=False)
    return c_path, d_path


def _row_changes(
    code, year, change_type, *,
    system="CPT", desc=None, advice=None, xwalk=None, new_desc=None,
):
    return {
        "code_system": system,
        "year": year,
        "category": "",
        "change_type": change_type,
        "code": code,
        "description": desc,
        "advice": advice,
        "crosswalk_code": xwalk,
        "new_descriptor": new_desc,
    }


def _row_deleted(code, deleted_date, *, system="CPT", desc=None, sub=None):
    return {
        "code": code,
        "description": desc,
        "deleted_date": deleted_date,
        "substitute_code": sub,
        "archived_info": "",
        "code_system": system,
    }


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def test_parse_deleted_date_cutoff_40():
    """YY ∈ [00..40] → 20YY, else 19YY."""
    assert CodeHistory.parse_deleted_date("1-Jan-00") == date(2000, 1, 1)
    assert CodeHistory.parse_deleted_date("1-Jan-40") == date(2040, 1, 1)
    assert CodeHistory.parse_deleted_date("1-Jan-41") == date(1941, 1, 1)
    assert CodeHistory.parse_deleted_date("1-Jan-99") == date(1999, 1, 1)
    assert CodeHistory.parse_deleted_date("31-Dec-24") == date(2024, 12, 31)
    assert CodeHistory.parse_deleted_date("6-Oct-20") == date(2020, 10, 6)


def test_parse_deleted_date_malformed():
    """Garbage, empty, NaN, wrong month abbreviation → None."""
    assert CodeHistory.parse_deleted_date("") is None
    assert CodeHistory.parse_deleted_date(None) is None
    assert CodeHistory.parse_deleted_date(float("nan")) is None
    assert CodeHistory.parse_deleted_date("2020-01-01") is None
    assert CodeHistory.parse_deleted_date("1-Xxx-20") is None
    assert CodeHistory.parse_deleted_date("32-Jan-20") is None   # invalid day
    assert CodeHistory.parse_deleted_date("1-Feb-20-extra") is None


# ---------------------------------------------------------------------------
# Code extraction from free-form text
# ---------------------------------------------------------------------------

def test_extract_codes_clean_and_prose():
    assert _extract_codes("97127") == ["97127"]
    assert _extract_codes("To report, use 97127") == ["97127"]
    assert _extract_codes("Use 11719 or J1885") == ["11719", "J1885"]
    assert _extract_codes("See codes 0479T and 2029F") == ["0479T", "2029F"]
    assert _extract_codes(None) == []
    assert _extract_codes("") == []
    assert _extract_codes("no codes here, just text") == []
    # De-duplication, first-occurrence order preserved
    assert _extract_codes("97127 then 97127 and 11719") == ["97127", "11719"]


# ---------------------------------------------------------------------------
# is_active state machine
# ---------------------------------------------------------------------------

def test_is_active_no_events_defers_to_kb(tmp_path):
    c, d = _write_csvs(tmp_path, [], [])
    h = CodeHistory.from_csvs(c, d, active_kb_codes={"43235"})
    assert h.is_active("43235", date(2020, 6, 1)) is True
    assert h.is_active("99999", date(2020, 6, 1)) is False  # unknown


def test_is_active_deleted_only(tmp_path):
    """Code with a single Deleted event, not in active KB.

    Before deletion → active (pre-records baseline).
    On/after deletion → inactive.
    """
    c, d = _write_csvs(
        tmp_path,
        [_row_changes("11041", 2018, "Deleted", xwalk="11042")],
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes=set())
    assert h.is_active("11041", date(2017, 12, 31)) is True
    assert h.is_active("11041", date(2018, 1, 1)) is False
    assert h.is_active("11041", date(2025, 1, 1)) is False


def test_is_active_new_then_deleted(tmp_path):
    """Born 2018, Deleted 2022.  Inactive before birth, active between,
    inactive after."""
    c, d = _write_csvs(
        tmp_path,
        [
            _row_changes("00001", 2018, "New"),
            _row_changes("00001", 2022, "Deleted", xwalk="00002"),
        ],
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes=set())
    assert h.is_active("00001", date(2017, 6, 1)) is False
    assert h.is_active("00001", date(2018, 1, 1)) is True
    assert h.is_active("00001", date(2021, 12, 31)) is True
    assert h.is_active("00001", date(2022, 1, 1)) is False
    assert h.is_active("00001", date(2025, 1, 1)) is False


def test_is_active_reactivated(tmp_path):
    """Code deleted in 2018 then re-introduced in 2022.  Matches the real
    J1370 / J3290 pattern in HCPCS."""
    c, d = _write_csvs(
        tmp_path,
        [
            _row_changes("J1370", 2018, "Deleted", system="HCPCS", xwalk="J9999"),
            _row_changes("J1370", 2022, "New", system="HCPCS"),
        ],
        [],
    )
    # In the real world this code is now in the active 2026 KB.
    h = CodeHistory.from_csvs(c, d, active_kb_codes={"J1370"})
    # Pre-records: first state event is 'deleted' → existed before → True
    assert h.is_active("J1370", date(2017, 1, 1)) is True
    # Deleted window
    assert h.is_active("J1370", date(2020, 6, 1)) is False
    # Re-introduced
    assert h.is_active("J1370", date(2022, 1, 1)) is True
    assert h.is_active("J1370", date(2025, 1, 1)) is True


def test_is_active_revised_only_defers_to_kb(tmp_path):
    """A code with only Revised events is active iff it's in the KB."""
    c, d = _write_csvs(
        tmp_path,
        [_row_changes("43235", 2020, "Revised", advice="Description updated.")],
        [],
    )
    h_in = CodeHistory.from_csvs(c, d, active_kb_codes={"43235"})
    h_out = CodeHistory.from_csvs(c, d, active_kb_codes=set())
    assert h_in.is_active("43235", date(2019, 1, 1)) is True
    assert h_in.is_active("43235", date(2025, 1, 1)) is True
    assert h_out.is_active("43235", date(2025, 1, 1)) is False


# ---------------------------------------------------------------------------
# Reconciliation: code_changes Deleted + deleted_codes precise date
# ---------------------------------------------------------------------------

def test_reconciliation_upgrades_date_and_merges_xwalk(tmp_path):
    """A code marked Deleted in both files: date is upgraded to the
    precise day, and crosswalk falls back to substitute_code when
    code_changes.crosswalk_code is missing."""
    c, d = _write_csvs(
        tmp_path,
        [_row_changes("97532", 2018, "Deleted",
                      advice="The 2018 code set deletes 97532.")],
        [_row_deleted("97532", "1-Jan-18", sub="97127")],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes=set())
    evs = h.events_for("97532")
    # Still a single Deleted event (upgraded, not duplicated)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.change_type == "deleted"
    assert ev.effective_date == date(2018, 1, 1)
    assert ev.crosswalk_code == "97127"               # merged from substitute_code
    assert ev.advice == "The 2018 code set deletes 97532."
    assert "deleted_codes" in ev.source


def test_reconciliation_preserves_explicit_xwalk(tmp_path):
    """If code_changes has its own crosswalk_code, it wins over substitute_code."""
    c, d = _write_csvs(
        tmp_path,
        [_row_changes("97532", 2018, "Deleted", xwalk="97127")],
        [_row_deleted("97532", "15-Mar-18", sub="OTHER")],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes=set())
    ev = h.events_for("97532")[0]
    assert ev.crosswalk_code == "97127"
    assert ev.effective_date == date(2018, 3, 15)  # precise date still upgraded


# ---------------------------------------------------------------------------
# crosswalk chasing
# ---------------------------------------------------------------------------

def test_crosswalk_basic(tmp_path):
    c, d = _write_csvs(
        tmp_path,
        [_row_changes("97532", 2018, "Deleted", xwalk="97127")],
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes={"97127"})
    # Before deletion: active, no substitution
    assert h.crosswalk("97532", date(2017, 1, 1)) is None
    # After deletion: substitute is 97127
    assert h.crosswalk("97532", date(2020, 1, 1)) == "97127"


def test_crosswalk_prose_xwalk(tmp_path):
    """``crosswalk_code`` may be prose like 'To report, use 97127'."""
    c, d = _write_csvs(
        tmp_path,
        [_row_changes("97532", 2018, "Deleted",
                      xwalk="To report, use 97127")],
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes={"97127"})
    assert h.crosswalk("97532", date(2020, 1, 1)) == "97127"


def test_crosswalk_chain(tmp_path):
    """A → B (deleted later) → C (active).

    Uses realistic 5-digit CPT-shaped tokens because the extraction regex
    only accepts real CPT/HCPCS code shapes (\\d{4}[A-Z] | \\d{5} |
    [A-Z]\\d{4}) — all-letter placeholders like ``AAAAA`` would be
    correctly rejected, which is exactly the behaviour we want in
    production to avoid false extractions from AMA advice prose.
    """
    c, d = _write_csvs(
        tmp_path,
        [
            _row_changes("99991", 2018, "Deleted", xwalk="99992"),
            _row_changes("99992", 2020, "Deleted", xwalk="99993"),
        ],
        [],
    )
    # Only 99993 is in the active KB.
    h = CodeHistory.from_csvs(c, d, active_kb_codes={"99993"})
    # In 2019: 99991 deleted → 99992, 99992 still active (pre-2020) → return 99992
    assert h.crosswalk("99991", date(2019, 1, 1)) == "99992"
    # In 2021: 99991 deleted → 99992, 99992 also deleted → chase → 99993
    assert h.crosswalk("99991", date(2021, 1, 1)) == "99993"


def test_crosswalk_cycle_guard(tmp_path):
    """A → B → A is not an infinite loop; returns None."""
    c, d = _write_csvs(
        tmp_path,
        [
            _row_changes("99991", 2018, "Deleted", xwalk="99992"),
            _row_changes("99992", 2018, "Deleted", xwalk="99991"),
        ],
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes=set())
    assert h.crosswalk("99991", date(2020, 1, 1)) is None


def test_crosswalk_none_when_no_substitute(tmp_path):
    c, d = _write_csvs(
        tmp_path,
        [_row_changes("00001", 2018, "Deleted")],  # no xwalk, no new_desc
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes=set())
    assert h.crosswalk("00001", date(2020, 1, 1)) is None


# ---------------------------------------------------------------------------
# advice_for
# ---------------------------------------------------------------------------

def test_advice_for_year_window(tmp_path):
    c, d = _write_csvs(
        tmp_path,
        [
            _row_changes("43235", 2018, "Revised", advice="2018 advice"),
            _row_changes("43235", 2022, "Revised", advice="2022 advice"),
            _row_changes("43235", 2025, "Revised", advice="2025 advice"),
        ],
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes={"43235"})
    # Default window is ±1
    assert h.advice_for("43235", date(2022, 6, 1)) == ["2022 advice"]
    assert h.advice_for("43235", date(2021, 6, 1)) == ["2022 advice"]   # 2022 within ±1
    assert h.advice_for("43235", date(2019, 1, 1)) == ["2018 advice"]   # 2018 within ±1
    # Wider window
    assert h.advice_for("43235", date(2020, 6, 1), year_window=(-5, 5)) == [
        "2018 advice", "2022 advice", "2025 advice",
    ]


# ---------------------------------------------------------------------------
# active_codes set math
# ---------------------------------------------------------------------------

def test_active_codes_set_math(tmp_path):
    c, d = _write_csvs(
        tmp_path,
        [
            _row_changes("11041", 2018, "Deleted", xwalk="11042"),
            _row_changes("NEWER", 2022, "New"),
        ],
        [],
    )
    h = CodeHistory.from_csvs(
        c, d, active_kb_codes={"43235", "11042", "NEWER"},
    )
    # 2017: 11041 still active, NEWER not born yet
    s2017 = h.active_codes(date(2017, 1, 1))
    assert "11041" in s2017
    assert "NEWER" not in s2017
    assert "43235" in s2017
    # 2025: 11041 deleted, NEWER active
    s2025 = h.active_codes(date(2025, 1, 1))
    assert "11041" not in s2025
    assert "NEWER" in s2025
    assert "43235" in s2025
    assert "11042" in s2025  # always-active KB code


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def test_to_events_df_schema(tmp_path):
    c, d = _write_csvs(
        tmp_path,
        [_row_changes("97532", 2018, "Deleted", xwalk="97127")],
        [_row_deleted("97532", "1-Jan-18", sub="97127")],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes=set())
    df = h.to_events_df()
    assert not df.empty
    for col in ["code", "code_system", "change_type", "effective_date",
                "year", "description", "advice", "crosswalk_code",
                "new_descriptor", "source"]:
        assert col in df.columns


def test_summary_stats(tmp_path):
    c, d = _write_csvs(
        tmp_path,
        [
            _row_changes("A", 2018, "New"),
            _row_changes("B", 2019, "Revised"),
            _row_changes("C", 2020, "Deleted", xwalk="A"),
        ],
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes={"A", "B"})
    stats = h.summary_stats()
    assert stats["n_codes_with_events"] == 3
    assert stats["n_events_total"] == 3
    assert stats["n_events_by_type"] == {"new": 1, "revised": 1, "deleted": 1}
    assert stats["n_active_kb_codes"] == 2


def test_crosswalk_map(tmp_path):
    c, d = _write_csvs(
        tmp_path,
        [
            _row_changes("97532", 2018, "Deleted", xwalk="97127"),
            _row_changes("00001", 2019, "Deleted"),  # no resolvable xwalk
        ],
        [],
    )
    h = CodeHistory.from_csvs(c, d, active_kb_codes={"97127"})
    xmap = h.crosswalk_map(date(2025, 1, 1))
    assert xmap == {"97532": "97127"}   # 00001 has no substitute
