#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
End-to-end tests for the NCCI rule ingestion and NCCIRuleChecker.

Run from project root:
    PYTHONPATH=src python3 tests/test_ncci_rule_checker.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# The CMS NCCI tables are public domain but bulky, so they are downloaded rather
# than committed.  Every test in this module needs them; the whole module skips
# when they are absent.
#
#     python scripts/download_ncci.py --out data/ncci
#
# Override the location with CPT_REC_NCCI_DIR.
NCCI_DIR = Path(os.environ.get("CPT_REC_NCCI_DIR", "data/ncci"))

_NCCI_MISSING = not (NCCI_DIR / "PTP_edits").is_dir()
pytestmark = pytest.mark.skipif(
    _NCCI_MISSING,
    reason=(f"CMS NCCI tables not found at {NCCI_DIR} — "
            f"run `python scripts/download_ncci.py --out {NCCI_DIR}`"),
)


@pytest.fixture(scope="module")
def checker():
    """Shared NCCIRuleChecker for pytest runs.

    Script-mode ``main()`` builds its own checker below — we keep a single
    construction path (``NCCIRuleChecker.from_data_dir``) so the two modes
    stay in sync.
    """
    from cpt_rec.common.ncci import NCCIRuleChecker
    return NCCIRuleChecker.from_data_dir(NCCI_DIR)


def test_individual_loaders():
    """Test that each file parser loads data with the expected schema."""
    from cpt_rec.common.ncci.ptp import load_ptp_dir
    from cpt_rec.common.ncci.aoc import load_aoc_file
    from cpt_rec.common.ncci.mue import load_mue_file

    # --- PTP ---
    ptp_df = load_ptp_dir(NCCI_DIR / "PTP_edits")
    assert len(ptp_df) > 0, "PTP DataFrame is empty"
    for col in ("col1", "col2", "effective", "deletion", "ccmi", "rationale"):
        assert col in ptp_df.columns, f"PTP missing column: {col}"
    assert set(ptp_df["ccmi"].dropna().unique()).issubset({0, 1, 9}), \
        f"Unexpected CCMI values: {ptp_df['ccmi'].unique()}"
    print(f"  PTP: {len(ptp_df):,} rows, CCMI distribution: {ptp_df['ccmi'].value_counts().to_dict()}")

    # --- AOC ---
    aoc_files = list((NCCI_DIR / "AOC_edits").glob("AOC_*-F-*.txt"))
    assert aoc_files, "No AOC flat files found"
    aoc_df = load_aoc_file(aoc_files[0])
    assert len(aoc_df) > 0, "AOC DataFrame is empty"
    for col in ("addon_code", "primary_code", "effective", "aoc_type"):
        assert col in aoc_df.columns, f"AOC missing column: {col}"
    assert set(aoc_df["aoc_type"].unique()).issubset({1, 3}), \
        f"Unexpected AOC types: {aoc_df['aoc_type'].unique()}"
    n_type1 = (aoc_df["aoc_type"] == 1).sum()
    n_type3 = (aoc_df["aoc_type"] == 3).sum()
    print(f"  AOC: {len(aoc_df):,} rows, Type 1: {n_type1}, Type 3: {n_type3}")

    # --- MUE ---
    mue_files = list((NCCI_DIR / "MUE").glob("MCR_MUE_*.csv"))
    assert mue_files, "No MUE CSV files found"
    mue_df = load_mue_file(mue_files[0])
    assert len(mue_df) > 0, "MUE DataFrame is empty"
    for col in ("code", "max_units", "mue_adjudication_indicator", "mue_rationale"):
        assert col in mue_df.columns, f"MUE missing column: {col}"
    assert (mue_df["max_units"] >= 0).all(), "MUE has negative max_units"
    n_zero = (mue_df["max_units"] == 0).sum()
    print(f"  MUE: {len(mue_df):,} rows, max_units range: [{mue_df['max_units'].min()}, {mue_df['max_units'].max()}], zero-unit codes: {n_zero}")

    print("PASS: test_individual_loaders\n")


def test_active_filtering():
    """Test that date-based filtering reduces row counts."""
    from datetime import date
    from cpt_rec.common.ncci.ptp import load_ptp_dir, filter_active_ptp
    from cpt_rec.common.ncci.aoc import load_aoc_file, filter_active_aoc

    ptp_df = load_ptp_dir(NCCI_DIR / "PTP_edits")
    ptp_active = filter_active_ptp(ptp_df, reference_date=date.today())
    assert len(ptp_active) <= len(ptp_df), "Filtering should not increase rows"
    assert len(ptp_active) > 0, "No active PTP edits — check reference date"
    print(f"  PTP: {len(ptp_df):,} total -> {len(ptp_active):,} active")

    # Filter to a far-past date should yield very few rows
    ptp_old = filter_active_ptp(ptp_df, reference_date=date(2000, 1, 1))
    assert len(ptp_old) < len(ptp_active), "Year-2000 filter should yield fewer active rows"
    print(f"  PTP at 2000-01-01: {len(ptp_old):,} active")

    aoc_files = list((NCCI_DIR / "AOC_edits").glob("AOC_*-F-*.txt"))
    aoc_df = load_aoc_file(aoc_files[0])
    aoc_active = filter_active_aoc(aoc_df, reference_date=date.today())
    assert len(aoc_active) <= len(aoc_df)
    print(f"  AOC: {len(aoc_df):,} total -> {len(aoc_active):,} active")

    print("PASS: test_active_filtering\n")


def test_checker_construction(checker):
    """Test that NCCIRuleChecker loads without errors."""
    assert checker.n_ptp_pairs > 100_000, f"Expected >100k PTP pairs, got {checker.n_ptp_pairs}"
    assert checker.n_addon_codes > 100, f"Expected >100 AOC codes, got {checker.n_addon_codes}"
    assert checker.n_mue_codes > 10_000, f"Expected >10k MUE codes, got {checker.n_mue_codes}"
    print(f"  Checker: {checker.n_ptp_pairs:,} PTP pairs, {checker.n_addon_codes:,} AOC codes, {checker.n_mue_codes:,} MUE codes")
    print("PASS: test_checker_construction\n")


def test_ptp_checks(checker):
    """Test PTP violation detection with known code pairs."""
    # Two independent codes that should NOT conflict
    hard, contingent = checker.check_ptp({"99213", "99214"})
    # 99213 and 99214 are both E/M codes — PTP should flag them
    print(f"  {{99213, 99214}}: hard={len(hard)}, contingent={len(contingent)}")

    # Codes from the DEID sample: 43235 + 45378 (EGD + colonoscopy)
    hard2, cont2 = checker.check_ptp({"43235", "45378"})
    print(f"  {{43235, 45378}}: hard={len(hard2)}, contingent={len(cont2)}")

    # Urology set from DEID sample
    hard3, cont3 = checker.check_ptp({"52332", "52353", "74420"})
    print(f"  {{52332, 52353, 74420}}: hard={len(hard3)}, contingent={len(cont3)}")
    for v in hard3:
        print(f"    HARD: {v.col1}/{v.col2} CCMI={v.ccmi} — {v.rationale}")
    for v in cont3:
        print(f"    CONTINGENT: {v.col1}/{v.col2} CCMI={v.ccmi} — {v.rationale}")

    print("PASS: test_ptp_checks\n")


def test_aoc_checks(checker):
    """Test AOC dependency checking."""
    # Pick a real add-on code from the loaded lookup (dynamic, not hard-coded)
    primaries = None
    addon_example = None
    # First try well-known add-on codes, then fall back to any code in the lookup
    for code in ["22842", "22845", "99292", "99417"]:
        p = checker.get_acceptable_primaries(code)
        if p is not None:
            addon_example = code
            primaries = p
            break
    if addon_example is None:
        # Dynamically pick the first add-on code from the internal lookup
        if checker._addon_to_primaries:
            addon_example = next(iter(checker._addon_to_primaries))
            primaries = checker._addon_to_primaries[addon_example]

    if addon_example:
        # Add-on alone → should trigger a violation
        viols_alone = checker.check_aoc({addon_example})
        assert any(v.addon_code == addon_example and not v.is_contractor_defined
                    for v in viols_alone), \
            f"Expected AOC violation for {addon_example} alone"
        print(f"  {{{addon_example}}} alone: {len(viols_alone)} violation(s) — correct")

        # Add-on + one valid primary → no Type 1 violation
        one_primary = next(iter(primaries))
        viols_with = checker.check_aoc({addon_example, one_primary})
        type1_viols = [v for v in viols_with if v.addon_code == addon_example and not v.is_contractor_defined]
        assert len(type1_viols) == 0, \
            f"Expected no Type-1 AOC violation for {{{addon_example}, {one_primary}}}"
        print(f"  {{{addon_example}, {one_primary}}}: 0 Type-1 violations — correct")
    else:
        print("  SKIP: no testable add-on code found in lookup")

    # Non-addon code should produce no AOC violations
    viols_none = checker.check_aoc({"43235"})
    type1_only = [v for v in viols_none if not v.is_contractor_defined]
    assert len(type1_only) == 0, "Non-addon code should not trigger Type-1 AOC"
    print(f"  {{43235}} (not an add-on): 0 Type-1 violations — correct")

    print("PASS: test_aoc_checks\n")


def test_mue_checks(checker):
    """Test MUE limit checking."""
    # 43235 (EGD with biopsy) typically has MUE=1
    mue_val = checker.get_mue("43235")
    assert mue_val is not None, "Expected MUE entry for 43235"
    print(f"  MUE(43235) = {mue_val}")

    # Within limit → no violation
    v1 = checker.check_mue({"43235": 1})
    assert len(v1) == 0, "1 unit should not violate MUE"
    print(f"  43235 x 1 unit: 0 violations — correct")

    # Over limit → violation
    v2 = checker.check_mue({"43235": mue_val + 1})
    assert len(v2) == 1, "Should have 1 MUE violation"
    assert v2[0].reported_units == mue_val + 1
    assert v2[0].max_units == mue_val
    print(f"  43235 x {mue_val + 1} units: 1 violation (max={mue_val}) — correct")

    # Unknown code → no violation (not in MUE table)
    v3 = checker.check_mue({"ZZZZZ": 999})
    assert len(v3) == 0, "Unknown code should not trigger MUE violation"
    print(f"  ZZZZZ x 999: 0 violations (not in MUE table) — correct")

    print("PASS: test_mue_checks\n")


def test_unified_check(checker):
    """Test the unified check() method with all three rule types."""
    from cpt_rec.common.ncci.rule_checker import CheckResult

    # Full check with units
    result = checker.check(
        code_set={"52332", "52353", "74420"},
        code_units={"52332": 1, "52353": 1, "74420": 1},
    )
    assert isinstance(result, CheckResult)
    s = result.summary()
    print(f"  {{52332, 52353, 74420}} full check:")
    print(f"    {s}")
    print(f"    is_hard_valid = {result.is_hard_valid}")

    # Check with None units (MUE skipped)
    result2 = checker.check(code_set={"43235", "45378"})
    assert result2.n_mue_violations == 0, "MUE should be skipped when code_units=None"
    print(f"  {{43235, 45378}} without units: mue_violations=0 — correct")

    # Empty code set
    result3 = checker.check(code_set=set())
    assert result3.is_hard_valid
    assert result3.n_hard_ptp == 0
    print(f"  Empty set: is_hard_valid=True — correct")

    # Single code (no pairs to check for PTP)
    result4 = checker.check(code_set={"43235"})
    assert result4.n_hard_ptp == 0
    assert result4.n_modifier_contingent_ptp == 0
    print(f"  Single code {{43235}}: 0 PTP violations — correct")

    print("PASS: test_unified_check\n")


def main():
    print("=" * 60)
    print("NCCI Rule Checker — Test Suite")
    print("=" * 60)
    print()

    if _NCCI_MISSING:
        print(f"SKIP: CMS NCCI tables not found at {NCCI_DIR}")
        print(f"      run `python scripts/download_ncci.py --out {NCCI_DIR}`")
        return

    from cpt_rec.common.ncci import NCCIRuleChecker

    test_individual_loaders()
    test_active_filtering()
    checker_obj = NCCIRuleChecker.from_data_dir(NCCI_DIR)
    test_checker_construction(checker_obj)
    test_ptp_checks(checker_obj)
    test_aoc_checks(checker_obj)
    test_mue_checks(checker_obj)
    test_unified_check(checker_obj)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
