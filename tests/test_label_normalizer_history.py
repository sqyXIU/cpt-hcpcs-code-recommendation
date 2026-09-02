#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Tests for the temporal-enrichment path in
:mod:`cpt_rec.common.preprocess.label_normalizer`.

Covers:
* ``_parse_procedure_date`` across native and string inputs
* ``enrich_rows_with_history``: active passthrough, retired-with-crosswalk,
  retired-without-crosswalk, unparseable-date passthrough
* ``enrich_csv_with_history``: counter dict + chunked roundtrip
* ``normalize_split`` honours ``history=`` + adds the four columns

Run with:
    pytest tests/test_label_normalizer_history.py -q
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from cpt_rec.common.knowledge.code_history import CodeHistory
from cpt_rec.common.preprocess.label_normalizer import (
    _parse_procedure_date,
    enrich_csv_with_history,
    enrich_rows_with_history,
    normalize_split,
)


# ---------------------------------------------------------------------------
# Fixture writers (shape matches the production CSVs)
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
    cdf = pd.DataFrame(changes_rows, columns=CHANGES_COLS)
    ddf = pd.DataFrame(deleted_rows, columns=DELETED_COLS)
    c_path = tmp / "code_changes.csv"
    d_path = tmp / "deleted_codes.csv"
    cdf.to_csv(c_path, index=False)
    ddf.to_csv(d_path, index=False)
    return c_path, d_path


def _rc(code, year, change_type, *, sys="CPT", advice=None, xwalk=None):
    return {
        "code_system": sys, "year": year, "category": "",
        "change_type": change_type, "code": code,
        "description": None, "advice": advice,
        "crosswalk_code": xwalk, "new_descriptor": None,
    }


def _rd(code, deleted_date, *, sys="CPT", sub=None):
    return {
        "code": code, "description": None, "deleted_date": deleted_date,
        "substitute_code": sub, "archived_info": "", "code_system": sys,
    }


@pytest.fixture
def tiny_history(tmp_path: Path) -> CodeHistory:
    """A hand-crafted CodeHistory with three useful shapes:

    * ``11111`` — stable code, active throughout (no events, in active KB)
    * ``22222`` — deleted 2022-07-01 with crosswalk to ``11111``
    * ``33333`` — deleted 2022-07-01 with NO substitute
    """
    changes_rows = [
        _rc("22222", 2022, "Deleted", xwalk="11111"),
        _rc("33333", 2022, "Deleted"),
    ]
    deleted_rows = [
        _rd("22222", "1-Jul-22", sub="11111"),
        _rd("33333", "1-Jul-22"),
    ]
    c_path, d_path = _write_csvs(tmp_path, changes_rows, deleted_rows)
    return CodeHistory.from_csvs(
        changes_csv=c_path,
        deleted_csv=d_path,
        active_kb_codes={"11111"},
    )


# ---------------------------------------------------------------------------
# _parse_procedure_date
# ---------------------------------------------------------------------------

def test_parse_procedure_date_native_types():
    assert _parse_procedure_date(date(2023, 5, 4)) == date(2023, 5, 4)
    assert _parse_procedure_date(datetime(2023, 5, 4, 13, 30)) == date(2023, 5, 4)
    assert _parse_procedure_date(pd.Timestamp("2023-05-04")) == date(2023, 5, 4)


def test_parse_procedure_date_strings():
    assert _parse_procedure_date("2023-05-04") == date(2023, 5, 4)
    assert _parse_procedure_date("5/4/2023") == date(2023, 5, 4)


def test_parse_procedure_date_unparseable_returns_none():
    assert _parse_procedure_date(None) is None
    assert _parse_procedure_date("") is None
    assert _parse_procedure_date(float("nan")) is None
    assert _parse_procedure_date("totally not a date") is None
    assert _parse_procedure_date(pd.NaT) is None


# ---------------------------------------------------------------------------
# enrich_rows_with_history
# ---------------------------------------------------------------------------

def test_enrich_active_code_passthrough(tiny_history: CodeHistory):
    df = pd.DataFrame([
        {"proc_codes": "11111", "PROCEDURE_DATE": "2023-01-15"},
    ])
    out = enrich_rows_with_history(df, tiny_history)
    row = out.iloc[0]
    assert row["proc_codes_valid_on_date"] == "11111"
    assert row["proc_codes_crosswalked"] == "11111"
    assert row["n_codes_retired_on_date"] == 0
    assert row["n_codes_crosswalked"] == 0


def test_enrich_retired_with_crosswalk(tiny_history: CodeHistory):
    """Code deleted before PROCEDURE_DATE should be gated out of
    ``valid`` and replaced by the crosswalk substitute in
    ``crosswalked``."""
    df = pd.DataFrame([
        {"proc_codes": "22222|11111", "PROCEDURE_DATE": "2023-01-15"},
    ])
    out = enrich_rows_with_history(df, tiny_history)
    row = out.iloc[0]
    # 22222 was deleted 2022-07-01 → not valid on 2023-01-15
    assert row["proc_codes_valid_on_date"] == "11111"
    # but it has a crosswalk → replaced by 11111 in xwalked column
    assert row["proc_codes_crosswalked"] == "11111|11111"
    assert row["n_codes_retired_on_date"] == 1
    assert row["n_codes_crosswalked"] == 1


def test_enrich_retired_without_crosswalk_drops(tiny_history: CodeHistory):
    """Code deleted with no substitute → dropped from xwalked column."""
    df = pd.DataFrame([
        {"proc_codes": "33333|11111", "PROCEDURE_DATE": "2023-01-15"},
    ])
    out = enrich_rows_with_history(df, tiny_history)
    row = out.iloc[0]
    assert row["proc_codes_valid_on_date"] == "11111"
    assert row["proc_codes_crosswalked"] == "11111"  # 33333 dropped
    assert row["n_codes_retired_on_date"] == 1
    assert row["n_codes_crosswalked"] == 0


def test_enrich_before_deletion_keeps_code(tiny_history: CodeHistory):
    """Code that was still active on the encounter date should stay
    active in ``valid_on_date`` even though it's deleted later."""
    df = pd.DataFrame([
        {"proc_codes": "22222", "PROCEDURE_DATE": "2021-01-15"},
    ])
    out = enrich_rows_with_history(df, tiny_history)
    row = out.iloc[0]
    assert row["proc_codes_valid_on_date"] == "22222"
    assert row["proc_codes_crosswalked"] == "22222"
    assert row["n_codes_retired_on_date"] == 0


def test_enrich_unparseable_date_passthrough(tiny_history: CodeHistory):
    """If PROCEDURE_DATE is unparseable we can't gate — pass through."""
    df = pd.DataFrame([
        {"proc_codes": "22222|33333", "PROCEDURE_DATE": "not a date"},
    ])
    out = enrich_rows_with_history(df, tiny_history)
    row = out.iloc[0]
    assert row["proc_codes_valid_on_date"] == "22222|33333"
    assert row["proc_codes_crosswalked"] == "22222|33333"
    assert row["n_codes_retired_on_date"] == 0
    assert row["n_codes_crosswalked"] == 0


def test_enrich_preserves_original_proc_codes(tiny_history: CodeHistory):
    df = pd.DataFrame([
        {"proc_codes": "22222|11111", "PROCEDURE_DATE": "2023-01-15"},
    ])
    out = enrich_rows_with_history(df, tiny_history)
    assert out.iloc[0]["proc_codes"] == "22222|11111"


def test_enrich_handles_missing_date_column(tiny_history: CodeHistory):
    """If the date column is absent entirely, every row is passthrough."""
    df = pd.DataFrame([{"proc_codes": "22222"}])
    out = enrich_rows_with_history(df, tiny_history)
    assert out.iloc[0]["proc_codes_valid_on_date"] == "22222"
    assert out.iloc[0]["n_codes_retired_on_date"] == 0


# ---------------------------------------------------------------------------
# enrich_csv_with_history (chunked CSV roundtrip)
# ---------------------------------------------------------------------------

def test_enrich_csv_chunked_roundtrip(tmp_path: Path, tiny_history: CodeHistory):
    df = pd.DataFrame([
        {"NOTE_ID": "n1", "proc_codes": "11111", "PROCEDURE_DATE": "2023-01-15"},
        {"NOTE_ID": "n2", "proc_codes": "22222", "PROCEDURE_DATE": "2023-01-15"},
        {"NOTE_ID": "n3", "proc_codes": "33333|11111", "PROCEDURE_DATE": "2023-01-15"},
        {"NOTE_ID": "n4", "proc_codes": "22222", "PROCEDURE_DATE": "invalid"},
    ])
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    df.to_csv(in_csv, index=False)

    counters = enrich_csv_with_history(
        input_csv=in_csv,
        output_csv=out_csv,
        history=tiny_history,
        # deliberately small chunksize to exercise multi-chunk append path
        chunksize=2,
    )
    assert counters["rows_in"] == 4
    assert counters["rows_with_unparseable_date"] == 1
    # Parseable-date rows: 22222 retired in n2 + 33333 retired in n3 = 2
    assert counters["total_codes_retired_on_date"] == 2
    assert counters["total_codes_crosswalked"] == 1  # only 22222→11111
    assert counters["total_codes_dropped_no_crosswalk"] == 1  # 33333 dropped

    out = pd.read_csv(out_csv)
    assert list(out["NOTE_ID"]) == ["n1", "n2", "n3", "n4"]
    assert "proc_codes_valid_on_date" in out.columns
    assert "proc_codes_crosswalked" in out.columns
    assert "n_codes_retired_on_date" in out.columns
    assert "n_codes_crosswalked" in out.columns


# ---------------------------------------------------------------------------
# normalize_split with history=
# ---------------------------------------------------------------------------

def test_normalize_split_with_history(tmp_path: Path, tiny_history: CodeHistory):
    """End-to-end: raw CSV → normalize_split(history=...) → enriched CSV."""
    df = pd.DataFrame([
        {
            "NOTE_ID": "n1",
            "NOTE_TEXT": "word " * 200,  # long enough to clear min_tokens
            "CPT_CODES": "22222, 11111",
            "PROCEDURE_DATE": "2023-01-15",
        },
        {
            "NOTE_ID": "n2",
            "NOTE_TEXT": "word " * 200,
            "CPT_CODES": "11111",
            "PROCEDURE_DATE": "2021-01-15",  # before 22222 deletion
        },
    ])
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    df.to_csv(in_csv, index=False)

    counters = normalize_split(
        input_csv=in_csv,
        output_csv=out_csv,
        kb_codes={"11111", "22222", "33333"},
        min_tokens=50,
        token_col=None,  # force recomputation from NOTE_TEXT
        history=tiny_history,
    )
    # History counters must be present
    assert "total_codes_retired_on_date" in counters
    assert "total_codes_crosswalked" in counters
    assert counters["rows_out"] == 2
    # Only n1 has a retired code on its encounter date
    assert counters["total_codes_retired_on_date"] == 1
    assert counters["total_codes_crosswalked"] == 1

    out = pd.read_csv(out_csv)
    assert set(out.columns) >= {
        "proc_codes",
        "proc_codes_valid_on_date",
        "proc_codes_crosswalked",
        "n_codes_retired_on_date",
        "n_codes_crosswalked",
    }


def test_normalize_split_without_history_unchanged(tmp_path: Path):
    """history=None → no new columns, no new counter keys."""
    df = pd.DataFrame([
        {
            "NOTE_ID": "n1",
            "NOTE_TEXT": "word " * 200,
            "CPT_CODES": "11111",
            "PROCEDURE_DATE": "2023-01-15",
        },
    ])
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    df.to_csv(in_csv, index=False)

    counters = normalize_split(
        input_csv=in_csv,
        output_csv=out_csv,
        kb_codes={"11111"},
        min_tokens=50,
        token_col=None,
    )
    assert "total_codes_retired_on_date" not in counters
    out = pd.read_csv(out_csv)
    assert "proc_codes_valid_on_date" not in out.columns
    assert "proc_codes_crosswalked" not in out.columns


# ---------------------------------------------------------------------------
# Script entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
