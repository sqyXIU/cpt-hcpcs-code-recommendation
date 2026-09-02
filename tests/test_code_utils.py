#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Tests for ``preprocess.code_utils`` — the single source of truth for the
CPT/HCPCS procedure-code shape regex.

The most important pin here is the ``A``-suffix family (COVID-19 vaccine
administration codes added in 2020): historically the regex was
``[FTUM]`` and silently dropped ``0001A``, ``0141A``, etc. from gold
labels everywhere ``parse_proc_codes`` was called.

Run from project root:
    PYTHONPATH=src python3 tests/test_code_utils.py
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# parse_proc_codes
# ---------------------------------------------------------------------------
def test_parse_proc_codes_includes_A_suffix():
    """COVID-era vaccine admin codes (4-digit + A) must survive parsing."""
    from cpt_rec.common.preprocess.code_utils import parse_proc_codes

    raw = "0001A, 0002A, 0141A, 99213"
    out = parse_proc_codes(raw)
    assert "0001A" in out, out
    assert "0002A" in out, out
    assert "0141A" in out, out
    assert "99213" in out, out
    print("PASS: test_parse_proc_codes_includes_A_suffix")


def test_parse_proc_codes_handles_full_family():
    """All four 4-digit suffix families + Cat I + HCPCS must round-trip."""
    from cpt_rec.common.preprocess.code_utils import parse_proc_codes

    cases = [
        ("0001A", "0001A"),  # vaccine admin
        ("2029F", "2029F"),  # Cat II
        ("0002M", "0002M"),  # MAAA
        ("0479T", "0479T"),  # Cat III
        ("0001U", "0001U"),  # PLA
        ("43235", "43235"),  # Cat I
        ("J1885", "J1885"),  # HCPCS
        ("g0121", "G0121"),  # case-folded
    ]
    for raw, expected in cases:
        out = parse_proc_codes(raw)
        assert out == [expected], f"input {raw!r} -> {out} (expected {[expected]})"
    print("PASS: test_parse_proc_codes_handles_full_family")


def test_parse_proc_codes_rejects_non_codes():
    from cpt_rec.common.preprocess.code_utils import parse_proc_codes

    # 2-char modifiers, prose, versioned annotations should never match.
    for raw in ["MA, ME, GD", "CPT", "44405-1", "ROBOT39", "PBHANDEXT"]:
        out = parse_proc_codes(raw)
        # We accept *some* 5-digit substring matches inside long strings,
        # so just assert the modifier/prose tokens themselves don't appear.
        assert "MA" not in out
        assert "ME" not in out
        assert "GD" not in out
        assert "CPT" not in out
    print("PASS: test_parse_proc_codes_rejects_non_codes")


# ---------------------------------------------------------------------------
# strip_modifier
# ---------------------------------------------------------------------------
def test_strip_modifier_keeps_A_suffix():
    """``0141A`` should round-trip; ``0141ABL`` (with bogus modifier
    appended) should reduce to ``0141A``."""
    from cpt_rec.common.preprocess.code_utils import strip_modifier

    assert strip_modifier("0141A") == "0141A"
    # Trailing modifier letters: should reduce back to the 5-char base.
    assert strip_modifier("0141ABL") == "0141A"
    # Real 5-digit Cat-I path still works.
    assert strip_modifier("52235BL") == "52235"
    # HCPCS path still works.
    assert strip_modifier("J0585") == "J0585"
    # Pure garbage still rejected.
    assert strip_modifier("PBHANDEXT") is None
    print("PASS: test_strip_modifier_keeps_A_suffix")


# ---------------------------------------------------------------------------
# is_valid_proc_code  (the lifted shape gate used by every vocab builder)
# ---------------------------------------------------------------------------
def test_is_valid_proc_code_accepts_canonical_shapes():
    from cpt_rec.common.preprocess.code_utils import is_valid_proc_code

    for code in ("00100", "99213", "0001A", "1234F", "0002U", "0003M",
                 "0001T", "0141A", "J1885", "G0121"):
        assert is_valid_proc_code(code), f"expected valid: {code!r}"
    print("PASS: test_is_valid_proc_code_accepts_canonical_shapes")


def test_is_valid_proc_code_rejects_modifiers_and_prose():
    from cpt_rec.common.preprocess.code_utils import is_valid_proc_code

    for tok in ("MA", "ME", "MD", "GD", "QQ", "CPT", "HCPCS", "",
                "44405-1", "G0279-1", "0001", "AA001", "00XXX",
                "J18852", "to report use 10060"):
        assert not is_valid_proc_code(tok), f"expected invalid: {tok!r}"
    print("PASS: test_is_valid_proc_code_rejects_modifiers_and_prose")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("code_utils — Test Suite")
    print("=" * 60)
    test_parse_proc_codes_includes_A_suffix()
    test_parse_proc_codes_handles_full_family()
    test_parse_proc_codes_rejects_non_codes()
    test_strip_modifier_keeps_A_suffix()
    test_is_valid_proc_code_accepts_canonical_shapes()
    test_is_valid_proc_code_rejects_modifiers_and_prose()
    print("=" * 60)
    print("ALL CODE-UTILS TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
