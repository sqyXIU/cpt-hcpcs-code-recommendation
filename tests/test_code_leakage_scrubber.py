#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Tests for the CPT/HCPCS code-leakage scrubber.

Run from project root:
    PYTHONPATH=src python3 tests/test_code_leakage_scrubber.py

Covers:
  * KB-aware masking of 5-digit CPT + HCPCS tokens
  * Preservation of non-KB 5-digit tokens (phone, MRN, zip)
  * Preservation of ICD-10 bracketed codes
  * EHR "Procedure Performed" template tail stripping
  * Idempotence
  * CLI roundtrip through a temporary CSV + KB
  * Post-scrub audit on a real note corpus (CPT_REC_SAMPLE_CSV):
    zero ground-truth codes appear verbatim in cleaned NOTE_TEXT.
"""

from __future__ import annotations

from _skip import skip

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# NOTE: removed from the repo 2026-08-30 — despite the "_DEID" name this file was
# never de-identified (the scrubber masks CPT codes, not PHI). Tests below SKIP
# when it is absent; regenerate it locally from the server if you need it, and
# do not commit it. See README.md "Data access, PHI, and licensing".
# ---------------------------------------------------------------------------
# Corpus and data paths.  This repository distributes no clinical notes and no
# licensed code descriptors, so every path below is overridable and the tests
# that need real data skip when it is absent.  See docs/DATA.md.
#
#   CPT_REC_KB_CSV          knowledge base   (default: the shipped public KB)
#   CPT_REC_SAMPLE_CSV      raw note corpus
#   CPT_REC_SAMPLE_NORM_CSV normalized note corpus
#   CPT_REC_NCCI_DIR        CMS NCCI tables  (scripts/download_ncci.py)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
PUBLIC_KB = _REPO / "data" / "kb" / "hcpcs_level2_public.csv"
SAMPLE_CSV = Path(os.environ.get("CPT_REC_SAMPLE_CSV", "data/notes/sample_notes.csv"))
KB_CSV = Path(os.environ.get("CPT_REC_KB_CSV", str(PUBLIC_KB)))

MASK = "<CPT_MASK>"


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
# A tiny KB covering the codes used throughout these tests.  Kept small so
# we can unambiguously verify that ONLY KB codes are redacted — a
# non-KB 5-digit token like "99999" or a phone-number fragment is
# preserved exactly.
_TINY_KB_CODES = {
    "43235",  # EGD diagnostic
    "45378",  # colonoscopy diagnostic
    "52332",  # cystoscopy w/ stent
    "52353",  # ureteroscopy w/ lithotripsy
    "74420",  # retrograde pyelogram
    "29873",  # knee scope, lat release
    "29875",  # knee scope, synovectomy
    "42820",  # adenotonsillectomy
    "G0121",  # HCPCS — screening colonoscopy
    "J0585",  # HCPCS — Botox injection
}


def _make_scrubber():
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    return CodeLeakageScrubber(_TINY_KB_CODES)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_mask_basic_cpt_codes():
    s = _make_scrubber()
    text = "Procedure Performed: 43235 followed by 45378."
    r = s.scrub(text)
    assert MASK in r.text, r.text
    assert "43235" not in r.text
    assert "45378" not in r.text
    assert r.n_codes_redacted == 2
    assert r.n_pr_tails_stripped == 0
    print("PASS: test_mask_basic_cpt_codes")


def test_mask_hcpcs_codes():
    s = _make_scrubber()
    text = "Administered J0585 per protocol; screening code G0121 billed."
    r = s.scrub(text)
    assert "J0585" not in r.text, r.text
    assert "G0121" not in r.text, r.text
    assert r.text.count(MASK) == 2
    # Lowercase HCPCS variants should also match (case-insensitive).
    r_lc = s.scrub("patient received j0585 today")
    assert r_lc.n_codes_redacted == 1, r_lc
    assert MASK in r_lc.text
    print("PASS: test_mask_hcpcs_codes")


def test_preserve_non_kb_5digit_tokens():
    """Phone, MRN, zip, account numbers must stay — they're not CPT codes."""
    s = _make_scrubber()
    text = (
        "Phone: 613-798-6730  Fax: 591-3612  MRN: 448-316-627  "
        "Franklin, TN 37067  Account #: 833-679-7411  "
        "Random number 99999 is NOT in KB."
    )
    r = s.scrub(text)
    # Nothing masked — none of these are KB codes.
    assert r.n_codes_redacted == 0, r
    assert "37067" in r.text
    assert "99999" in r.text
    assert "316-627" in r.text
    assert MASK not in r.text
    print("PASS: test_preserve_non_kb_5digit_tokens")


def test_preserve_icd_bracketed_codes():
    s = _make_scrubber()
    text = "Pre-op Diagnosis: Renal stone [N20.0]; also [C67.9] bladder CA."
    r = s.scrub(text)
    assert r.text == text, f"ICD brackets unexpectedly modified: {r.text!r}"
    assert r.n_codes_redacted == 0
    print("PASS: test_preserve_icd_bracketed_codes")


def test_preserve_embedded_digit_runs():
    """'152332' contains '52332' but must not match (digit-boundary)."""
    s = _make_scrubber()
    text = "Reference 152332019 appeared on the ledger, and 4523789 too."
    r = s.scrub(text)
    assert r.n_codes_redacted == 0, r
    assert MASK not in r.text
    print("PASS: test_preserve_embedded_digit_runs")


def test_strip_pr_template_tail_single_line():
    """EHR template: `<PROC>  <CODE> - PR <DESCR>` -> `<PROC>  <CPT_MASK>`."""
    s = _make_scrubber()
    text = (
        "Procedure Performed (Required):\n"
        "CYSTOSCOPY WITH INSERTION STENT URETER  52332 - PR CYSTOSCOPY,INSERT URETERAL STENT\n"
        "Findings: stone dusted."
    )
    r = s.scrub(text)
    # The code is gone.
    assert "52332" not in r.text
    # The official CPT short description is gone.
    assert "CYSTOSCOPY,INSERT URETERAL STENT" not in r.text
    # The EHR-procedure-name narrative prefix is preserved.
    assert "CYSTOSCOPY WITH INSERTION STENT URETER" in r.text
    assert r.n_codes_redacted == 1
    assert r.n_pr_tails_stripped == 1
    print("PASS: test_strip_pr_template_tail_single_line")


def test_strip_pr_template_tail_multi_proc_collapsed():
    """Multiple procs on one collapsed line separated by 2+ spaces."""
    s = _make_scrubber()
    text = (
        "RELEASE ARTHROSCOPY RETINACULUM KNEE  29873 - PR KNEE SCOPE, W/LATERAL RELEASE"
        "    "
        "ARTHROSCOPY KNEE  29875 - PR KNEE SCOPE,PART SYNOVECT"
    )
    r = s.scrub(text)
    assert "29873" not in r.text
    assert "29875" not in r.text
    assert "W/LATERAL RELEASE" not in r.text
    assert "PART SYNOVECT" not in r.text
    # Both procedure-name prefixes survive.
    assert "RELEASE ARTHROSCOPY RETINACULUM KNEE" in r.text
    assert "ARTHROSCOPY KNEE" in r.text
    assert r.n_codes_redacted == 2
    assert r.n_pr_tails_stripped == 2
    print("PASS: test_strip_pr_template_tail_multi_proc_collapsed")


def test_no_pr_tail_when_pattern_absent():
    """Bare code mention with no '- PR <descr>' tail: mask code only."""
    s = _make_scrubber()
    text = "Billed 43235 and the patient tolerated it well."
    r = s.scrub(text)
    assert "43235" not in r.text
    assert MASK in r.text
    assert r.n_pr_tails_stripped == 0
    # Narrative text untouched.
    assert "tolerated it well" in r.text
    print("PASS: test_no_pr_tail_when_pattern_absent")


def test_mask_codes_with_alpha_suffix():
    """
    EHR templates frequently glue a 1-3 letter laterality / modifier
    suffix directly onto the code ("49621L", "59150T", "43281R",
    "49320PB").  The boundary-aware code regex must still redact the
    full token so the 5-digit leak is eliminated.
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    s = CodeLeakageScrubber(
        _TINY_KB_CODES | {"49621", "59150", "43281", "49320", "15823", "97598", "49255", "55700"}
    )
    cases = [
        ("49621L bilateral", ""),           # single-letter
        ("(43281R)", ""),                    # parenthesized + single
        ("59150T - PR TRAUMA LAPAROSCOPIC", ""),   # single + PR tail
        ("49320PB procedure", ""),           # 2-letter suffix
        ("OMENTECTOMY (CARCINOMATOSIS) (49255C) CPT", ""),  # in parens
        ("CPT [55700A] LATERALITY: N/A", ""),   # in square brackets
    ]
    for raw, _ in cases:
        r = s.scrub(raw)
        assert MASK in r.text, f"expected mask in {r.text!r} for input {raw!r}"
        # No 5-digit run should survive from the original code span.
        survived = re.search(r"(?<!\d)\d{5}(?!\d)", r.text)
        assert survived is None, (
            f"5-digit leak survived in {r.text!r} for input {raw!r}"
        )
    print("PASS: test_mask_codes_with_alpha_suffix")


def test_mask_codes_with_glued_prefix_letter():
    """
    EHR dictation sometimes glues 1-3 stray letters IN FRONT of a
    5-digit code with no separator.  Examples observed in production:
      * single letter   — ``C15820.11``, ``W52353``, ``A43235``
      * 2-3 letter word — ``eye66180``, ``CPT27197``, ``AGE12345``
    The full token must be masked so the 5-digit leak is eliminated.
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    s = CodeLeakageScrubber(
        _TINY_KB_CODES | {"15820", "15823", "43235", "66180", "27197"}
    )
    cases = [
        "BLEPHAROPLASTY C15820.11 - cosmetic lower lid",  # 1-letter
        "URETEROSCOPY W52353 - PR CYSTO",                  # 1-letter
        "Indication line: A43235 bilateral",               # 1-letter
        "CPT: C15823 bilateral",                           # 1-letter
        "reinforcement of the the left eye66180. SURGEON:",  # 3-letter
        "pelvic injury (CPT27197) FEMORAL",                  # 3-letter
    ]
    for text in cases:
        r = s.scrub(text)
        survived = re.search(r"(?<!\d)\d{5}(?!\d)", r.text)
        assert survived is None, (
            f"5-digit leak survived in {r.text!r} for input {text!r}"
        )
        assert MASK in r.text, r.text
    print("PASS: test_mask_codes_with_glued_prefix_letter")


def test_preserve_long_prefix_glued_to_code():
    """
    Policy: when 4–5 letters are glued to a code with no separator
    (``CLIPS58671``, ``ABCD43235``), Epic-concatenation in real EHR
    notes is overwhelmingly more common than legitimate identifier
    collisions, so the scrubber absorbs them.  The prefix cap is 5;
    a 6+ letter run is still preserved (it's almost certainly a real
    identifier or narrative word glued by a sectionizer bug rather
    than an Epic template crush).
    """
    s = _make_scrubber()
    # 4-letter prefix: absorbed (was preserved under the old {0,3} cap)
    r = s.scrub("legacy identifier ABCD43235 logged")
    assert "43235" not in r.text, r.text
    assert MASK in r.text
    # 6+ letter prefix: still preserved (out of the {0,5} window)
    r2 = s.scrub("legacy identifier ABCDEF43235 logged")
    assert "ABCDEF43235" in r2.text, r2.text
    assert MASK not in r2.text
    print("PASS: test_preserve_long_prefix_glued_to_code")


def test_real_data_epic_crush_patterns():
    """Regression tests for the specific leak patterns found by
    ``scripts/audit_cpt_leakage.py`` on the 20k/2k/2k/2k subsample:

      * ``22558MILF``  — 4-letter surgical abbrev glued as suffix
      * ``67113G25``, ``67113G23`` — CPT modifier digits after a
        1-letter alpha suffix
      * ``67108G25B``, ``67108G25BIL`` — same pattern as 67113G25 but
        with an extra 1-3 letter laterality tag glued AFTER the
        modifier digits
      * ``CLIPS58671`` — 5-letter word run straight into a code

    All must be fully redacted; the cleaned text must have no bare
    5-digit token that matches the audit's
    ``(?<!\\d)CODE(?!\\d)`` boundary.
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    s = CodeLeakageScrubber(
        _TINY_KB_CODES | {"22558", "67113", "67108", "58671"}
    )
    cases = [
        (
            "ETROPERITONEAL APPROACH MILF 22558MILF - PR ANTERIOR INTERBODY",
            {"22558"},
        ),
        (
            "7 GAUGE <CPT_MASK>(R) Code: 67113G25 - PR REPAIR",
            {"67113"},
        ),
        (
            "GAU BIL <CPT_MASK>(R) Code: 67113G23 - PR REPAIR",
            {"67113"},
        ),
        (
            # Exact pattern from train_eval.csv row 12608: alpha + digits
            # + trailing single letter glued to the modifier-digit tail.
            "25 GAU BIL CPT(R) Code: 67108G25B - PR REP RETINAL DETACHMENT",
            {"67108"},
        ),
        (
            # Same pattern with a 3-letter trailing tag.
            "25 GAU BIL CPT(R) Code: 67108G25BIL - PR REP RETINAL DETACHMENT",
            {"67108"},
        ),
        (
            "LAPAROSCOPIC TUBAL LIGATION WITH FILSHIE CLIPS58671) <CPT_MASK>",
            {"58671"},
        ),
    ]
    for text, must_vanish in cases:
        r = s.scrub(text)
        for code in must_vanish:
            # Use the audit script's exact digit-boundary regex.
            pat = re.compile(rf"(?<!\d){re.escape(code)}(?!\d)")
            assert pat.search(r.text) is None, (
                f"leak of {code!r} in scrubbed text: {r.text!r}\n"
                f"input was: {text!r}"
            )
    print("PASS: test_real_data_epic_crush_patterns")


def test_mask_codes_with_multiplicand_suffix():
    """
    ``CODExN`` notation ("code billed N times") is common in
    multi-procedure notes, e.g. ``15734x2``, ``15272x3``.  The full
    token must be masked so the 5-digit leak is eliminated.
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    s = CodeLeakageScrubber(_TINY_KB_CODES | {"15734", "15272"})
    cases = [
        "paraspinous muscle flaps (15734x2)",
        "(CPT: <CPT_MASK>, 15272x3, <CPT_MASK> x3)",
        "29875x10 total repetitions",               # 2-digit N
    ]
    for text in cases:
        r = s.scrub(text)
        survived = re.search(r"(?<!\d)\d{5}(?!\d)", r.text)
        assert survived is None, (
            f"5-digit leak survived in {r.text!r} for input {text!r}"
        )
    print("PASS: test_mask_codes_with_multiplicand_suffix")


def test_preserve_longer_alpha_runs_after_code():
    """
    Policy: 4-letter suffixes (``22558MILF``, ``52332LONG``) are
    absorbed — real EHR notes crush surgical abbreviations onto the
    trailing end of a code much more often than they produce a
    5-digit identifier genuinely ending in a letter run.  The suffix
    cap is 4; a 5+ letter run is still preserved.
    """
    s = _make_scrubber()
    # 4 trailing letters: absorbed (was preserved under the old {0,3} cap)
    r = s.scrub("Account 52332LONG was flagged.")
    assert MASK in r.text, r.text
    assert "52332" not in r.text
    # 5+ trailing letters: still preserved (out of the {0,4} window)
    r2 = s.scrub("Account 52332LONGER was flagged.")
    assert MASK not in r2.text, r2.text
    assert "52332LONGER" in r2.text
    print("PASS: test_preserve_longer_alpha_runs_after_code")


def test_strip_stray_letter_prefix_in_cpt_marker():
    """
    Dictation artifact: "CPT: C15823 bilateral".  The stray uppercase
    letter is dropped, then the 5-digit code is masked normally.
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    s = CodeLeakageScrubber(_TINY_KB_CODES | {"15823"})
    text = "Procedure: COSMETIC BLEPHAROPLASTY both upper eyelids CPT: C15823 bilateral"
    r = s.scrub(text)
    assert "15823" not in r.text, r.text
    assert "C15823" not in r.text, r.text
    assert MASK in r.text
    # The word "bilateral" (narrative) survives.
    assert "bilateral" in r.text
    # And a non-prefixed 5-digit token preceded by "CPT:" still works.
    r2 = s.scrub("... CPT: 15823 bilateral")
    assert "15823" not in r2.text
    assert MASK in r2.text
    print("PASS: test_strip_stray_letter_prefix_in_cpt_marker")


def test_stray_prefix_does_not_overreach():
    """
    The stray-letter pre-pass must NOT drop a real modifier word that
    happens to start with an uppercase letter near a CPT marker —
    e.g. 'CPT ROBOTIC 43235' has 'ROBOTIC' as a word, not a stray
    letter glued to digits, so it should survive untouched.
    """
    s = _make_scrubber()
    text = "Note mentions CPT ROBOTIC 43235 was used."
    r = s.scrub(text)
    assert "ROBOTIC" in r.text, r.text
    # 43235 is in KB, so it should be masked.
    assert "43235" not in r.text
    assert MASK in r.text
    print("PASS: test_stray_prefix_does_not_overreach")


def test_extra_codes_for_retired_gt_codes():
    """
    Retired CPT codes (e.g. 49560 / 49585 / 49568 pre-2023 hernia family)
    no longer live in the current KB but still appear in historical GT
    labels.  The scrubber must accept a user-supplied ``extra_codes``
    vocabulary to cover them.
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    text = (
        "Procedure:    REPAIR HERNIA VENTRAL  "
        "CPT(R) Code:  49560 - PR REPAIR INCISIONAL HERNIA,REDUCIBLE"
    )
    # KB-only: retired code slips through.
    s_kb_only = CodeLeakageScrubber(_TINY_KB_CODES)  # no 49560 in vocab
    assert "49560" in s_kb_only.scrub(text).text

    # With extra_codes: retired code is masked AND its PR-tail stripped.
    s_extended = CodeLeakageScrubber(_TINY_KB_CODES, extra_codes={"49560"})
    r = s_extended.scrub(text)
    assert "49560" not in r.text
    assert "INCISIONAL HERNIA,REDUCIBLE" not in r.text
    assert "REPAIR HERNIA VENTRAL" in r.text
    assert r.n_codes_redacted == 1
    assert r.n_pr_tails_stripped == 1
    print("PASS: test_extra_codes_for_retired_gt_codes")


def test_idempotence():
    s = _make_scrubber()
    text = (
        "Procedure Performed: EGD  43235 - PR EGD BIOPSY    SCREENING COLONOSCOPY  45378 - PR COLONOSCOPY"
    )
    once = s.scrub(text)
    twice = s.scrub(once.text)
    assert twice.text == once.text
    assert twice.n_codes_redacted == 0
    assert twice.n_pr_tails_stripped == 0
    print("PASS: test_idempotence")


def test_empty_and_nan_input():
    s = _make_scrubber()
    assert s.scrub("").text == ""
    assert s.scrub(None).text == ""  # type: ignore[arg-type]
    # pd.NA/NaN from a CSV should not crash.
    import math
    assert s.scrub(math.nan).text == ""  # type: ignore[arg-type]
    print("PASS: test_empty_and_nan_input")


def test_trie_regex_equivalent_to_flat_alternation():
    """The trie-compressed code pattern (speed path) must match exactly
    what the old flat ``"|".join(sorted(codes, key=(-len, c)))``
    alternation matched — same spans, same counts — on every tricky
    surface form this suite pins, plus shared-prefix branching cases."""
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
        _trie_regex,
    )

    codes = set(_TINY_KB_CODES) | {"29874", "2987F", "C9739", "0075T"}
    trie_scrubber = CodeLeakageScrubber(codes)

    # Reference: the pre-trie flat alternation, embedded in the SAME
    # surrounding prefix/suffix/multiplicand regex.
    sorted_codes = sorted(codes, key=lambda c: (-len(c), c))
    flat_alt = "|".join(re.escape(c) for c in sorted_codes)
    flat_re = re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"[A-Za-z]{{0,5}}?"
        rf"(?:{flat_alt})"
        rf"[A-Za-z]{{0,4}}"
        rf"(?:(?<=[A-Za-z])\d{{1,2}}[A-Za-z]{{0,3}})?"
        rf"(?:x\d{{1,2}})?"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )

    probes = [
        "EGD  43235 - PR EGD BIOPSY    COLO 45378 done",
        "codes C15820 W52353 eye66180 CPT27197 CLIPS58671",
        "49621L 49320PB 97598A 22558MILF 67113G25 67108G25BIL",
        "15734x2 and 15272x3 multiplicands",
        "phone 5233-2100, MRN 452332, zip 74420-1234, [Z98.890]",
        "g0121 lowercase hcpcs, J0585 mixed",
        "prefix-of-code cases: 29873 29874 2987F 0075T C9739",
        "ABCDEF43235 must keep long prefixes; 52332LONGER keeps tails",
        "52332 35 stray number must not fire the modifier slot",
        "",
        "no codes at all here",
    ]
    for text in probes:
        expect, n_expect = flat_re.subn(MASK, text)
        got = trie_scrubber._code_re.subn(MASK, text)
        assert got == (expect, n_expect), (
            f"trie != flat on {text!r}: {got} vs {(expect, n_expect)}"
        )
    # And the pattern really is trie-shaped (shared prefixes, not 14 branches).
    assert _trie_regex(codes).count("|") < len(codes) - 1
    print("PASS: test_trie_regex_equivalent_to_flat_alternation")


def test_scrub_series_parallel_matches_serial():
    """workers>1 must reproduce the serial output exactly (values, counts,
    index, and row order) — this is what licenses --workers in R3."""
    s = _make_scrubber()
    texts = [
        f"note {i}: EGD 43235 - PR EGD BIOPSY   then 45378, call 5233-2100"
        if i % 3 == 0 else f"note {i}: no codes, MRN 452332"
        for i in range(200)
    ]
    texts[7] = ""  # empty + shuffled index must survive the chunking
    series = pd.Series(texts, index=[f"r{i}" for i in range(200)], name="NOTE_TEXT")

    t1, c1, p1 = s.scrub_series(series, workers=1)
    t2, c2, p2 = s.scrub_series(series, workers=4)
    pd.testing.assert_series_equal(t1, t2)
    pd.testing.assert_series_equal(c1, c2)
    pd.testing.assert_series_equal(p1, p2)
    print("PASS: test_scrub_series_parallel_matches_serial")


# ---------------------------------------------------------------------------
# CLI roundtrip (synthetic KB + synthetic notes)
# ---------------------------------------------------------------------------
def test_cli_roundtrip_synthetic():
    """Run `cptrec-scrub-leakage` end-to-end via subprocess on tiny fixtures."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Synthetic KB compatible with CodeKnowledgeBase.from_csv.
        kb_df = pd.DataFrame(
            [
                {"code": "43235", "code_description": "EGD diagnostic",
                 "code_lay_term": "EGD", "code_system": "CPT"},
                {"code": "45378", "code_description": "Colonoscopy diagnostic",
                 "code_lay_term": "Colonoscopy", "code_system": "CPT"},
                {"code": "52332", "code_description": "Cystoscopy w/ stent",
                 "code_lay_term": "Stent", "code_system": "CPT"},
                {"code": "J0585", "code_description": "Botulinum toxin A",
                 "code_lay_term": "Botox", "code_system": "HCPCS"},
            ]
        )
        # Need at least code_range_1 columns present (optional, defaulted).
        for lvl in range(1, 7):
            kb_df[f"code_range_{lvl}"] = ""
            kb_df[f"code_range_{lvl}_description"] = ""
        kb_path = tmp / "kb.csv"
        kb_df.to_csv(kb_path, index=False)

        # Synthetic input CSV.
        notes_df = pd.DataFrame(
            [
                {
                    "NOTE_ID": "n1",
                    "NOTE_TEXT": (
                        "Procedure Performed (Required):\n"
                        "COLONOSCOPY  45378 - PR COLONOSCOPY DIAGNOSTIC\n"
                        "Findings: normal."
                    ),
                    "CPT_CODES": "45378",
                },
                {
                    "NOTE_ID": "n2",
                    "NOTE_TEXT": "Phone 615-936-1000. Received J0585 today.",
                    "CPT_CODES": "J0585",
                },
                {
                    "NOTE_ID": "n3",
                    "NOTE_TEXT": "Nothing to redact here — no codes in KB.",
                    "CPT_CODES": "",
                },
            ]
        )
        in_path = tmp / "notes.csv"
        out_path = tmp / "notes_clean.csv"
        notes_df.to_csv(in_path, index=False)

        # Invoke CLI module directly via `python -m` so PYTHONPATH=src works.
        cmd = [
            sys.executable,
            "-m",
            "cpt_rec.common.preprocess.code_leakage_scrubber",
            "--input", str(in_path),
            "--output", str(out_path),
            "--kb", str(kb_path),
            "--log-level", "WARNING",
        ]
        env_src = Path("src").resolve()
        env = {"PYTHONPATH": str(env_src)}
        # Inherit parent environment, just layer PYTHONPATH.
        import os
        full_env = {**os.environ, **env}
        res = subprocess.run(
            cmd, capture_output=True, text=True, env=full_env, check=False
        )
        assert res.returncode == 0, f"CLI failed:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}"

        out = pd.read_csv(out_path)
        assert list(out["NOTE_ID"]) == ["n1", "n2", "n3"]
        # Row 1: code + PR tail both gone, procedure-name prefix kept.
        t1 = out.loc[out["NOTE_ID"] == "n1", "NOTE_TEXT"].item()
        assert "45378" not in t1
        assert "COLONOSCOPY DIAGNOSTIC" not in t1  # tail stripped
        assert "COLONOSCOPY" in t1                  # narrative prefix kept
        assert MASK in t1
        # Row 2: J0585 masked, phone preserved (not in KB).
        t2 = out.loc[out["NOTE_ID"] == "n2", "NOTE_TEXT"].item()
        assert "J0585" not in t2
        assert "615-936-1000" in t2
        # Row 3: unchanged.
        t3 = out.loc[out["NOTE_ID"] == "n3", "NOTE_TEXT"].item()
        assert t3 == notes_df.loc[2, "NOTE_TEXT"]
        # Stats columns present and correct.
        assert "N_CODES_REDACTED" in out.columns
        assert "N_PR_TAILS_STRIPPED" in out.columns
        assert int(out.loc[out["NOTE_ID"] == "n1", "N_CODES_REDACTED"].item()) == 1
        assert int(out.loc[out["NOTE_ID"] == "n1", "N_PR_TAILS_STRIPPED"].item()) == 1
        assert int(out.loc[out["NOTE_ID"] == "n2", "N_CODES_REDACTED"].item()) == 1
        assert int(out.loc[out["NOTE_ID"] == "n3", "N_CODES_REDACTED"].item()) == 0
    print("PASS: test_cli_roundtrip_synthetic")


# ---------------------------------------------------------------------------
# Vocab shape-filter: reject non-procedure-code tokens (modifiers, prose, ...)
# ---------------------------------------------------------------------------
def test_vocab_shape_filter_drops_non_procedure_codes():
    """Only real CPT/HCPCS procedure-code shapes should survive the
    ``CodeLeakageScrubber.__init__`` shape gate.

    Short HCPCS modifier codes (``MA``, ``ME``, ``MD``, ``GD``) and stray
    prose (``CPT``, ``645``, ``44405-1``) must NOT enter the alternation
    — if they did, the IGNORECASE regex would eat English words whose
    letters happen to contain those sequences (``Name``, ``male``,
    ``MAIN``, ``Same`` …).
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    raw_vocab = {
        # --- real procedure codes (must survive) ---
        "00100",    # CPT Cat I
        "99213",    # CPT Cat I
        "0001T",    # CPT Cat III
        "1234F",    # CPT Cat II
        "0002U",    # PLA
        "0003M",    # MAAA
        "0141A",    # COVID-19 vaccine admin (Cat-II-ish, A-suffix)
        "J1885",    # HCPCS Level II
        "G0121",    # HCPCS Level II
        # --- must be filtered out ---
        "MA", "MB", "ME", "MD", "GD", "QQ",  # 2-char HCPCS modifiers
        "645",           # 3-digit stray
        "CPT",           # code-system tag leaked into code col
        "HCPCS",
        "",              # empty (already filtered by prior guard but harmless)
        "44405-1",       # versioned annotation (deleted_codes.csv residue)
        "G0279-1",
        "00100-2",
        "To report, use 10060",  # prose
        "0001",          # 4 digits only, missing letter
        "AA001",         # letter first but 3 digits
        "00XXX",         # digits first but letters
        "J18852",        # HCPCS-shape but 5 digits → not valid
    }
    expected = {"00100", "99213", "0001T", "1234F", "0002U", "0003M",
                "0141A", "J1885", "G0121"}

    s = CodeLeakageScrubber(raw_vocab)
    assert s._codes == expected, (
        f"shape filter kept {sorted(s._codes)}; expected {sorted(expected)}"
    )
    # Also assert it raises cleanly when EVERY input is garbage.
    import pytest
    try:
        CodeLeakageScrubber({"MA", "ME", "CPT", "645"})
    except ValueError as e:
        assert "shape filter" in str(e), e
    else:
        raise AssertionError("expected ValueError when all vocab is invalid")

    print("PASS: test_vocab_shape_filter_drops_non_procedure_codes")


def test_short_modifier_tokens_do_not_mask_english_words():
    """Regression for the sample-run over-masking reported by the user.

    Before the shape filter was added, feeding 2-char HCPCS modifier codes
    (``MA``, ``ME``, ``MD``, ``GD``) into the scrubber vocabulary caused
    ``Name`` → ``<CPT_MASK>``, ``female`` → ``<CPT_MASK>``, ``MAIN`` →
    ``<CPT_MASK>``, ``Same`` → ``<CPT_MASK>``, ``Sex`` (via ``MD`` +
    suffix) → ``<CPT_MASK>``, ``media`` → ``<CPT_MASK>``, and so on.

    With the filter in place, those modifier tokens are dropped from the
    vocab at construction time, and the text below must pass through
    completely unchanged — while real 5-digit codes on the same line
    still get masked.
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )

    # Include the exact short-code "HCPCS modifier" tokens that were
    # observed in the pre-cleanup history CSVs.  The scrubber MUST drop
    # them; if a future regression re-enables them, this test fails.
    vocab = (
        _TINY_KB_CODES
        | {"MA", "MB", "MC", "MD", "ME", "MF", "MG", "MH", "GD", "645", "CP"}
    )
    s = CodeLeakageScrubber(vocab)

    text = (
        "Patient Name: Frank Molina  Sex: male  Hansen, MD  "
        "Same diagnosis  MAIN OR  female patient  media  primary  "
        "September 26  EGD and Colonoscopy"
    )
    r = s.scrub(text)
    assert r.text == text, (
        f"short-modifier tokens caused over-masking.\n"
        f"  in : {text!r}\n"
        f"  out: {r.text!r}"
    )
    assert r.n_codes_redacted == 0

    # Companion check: a real code on the same line is still masked.
    text2 = (
        "Patient Name: Frank  Sex: male  Code: 43235 billed.  "
        "Hansen, MD  female patient"
    )
    r2 = s.scrub(text2)
    assert "43235" not in r2.text, r2.text
    assert MASK in r2.text
    assert "Name" in r2.text, "`Name` must survive (not an HCPCS modifier match)"
    assert "female" in r2.text
    assert "MD" in r2.text
    assert r2.n_codes_redacted == 1
    print("PASS: test_short_modifier_tokens_do_not_mask_english_words")


def test_cli_with_polluted_history_still_scrubs_only_real_codes():
    """End-to-end CLI regression: even if the history CSVs handed to
    ``cptrec-scrub-leakage`` contain 2-char HCPCS modifier rows (pre-cleanup
    state OR a future ingestion regression), the scrubber must filter
    them at vocab-build time and leave English words alone.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Synthetic KB — only legitimate procedure-code shapes.
        kb_df = pd.DataFrame(
            [
                {"code": "43235", "code_description": "EGD",
                 "code_lay_term": "EGD", "code_system": "CPT"},
            ]
        )
        for lvl in range(1, 7):
            kb_df[f"code_range_{lvl}"] = ""
            kb_df[f"code_range_{lvl}_description"] = ""
        kb_path = tmp / "kb.csv"
        kb_df.to_csv(kb_path, index=False)

        # Intentionally polluted history CSVs: mix real codes with the
        # exact modifier rows observed in the pre-cleanup data.
        changes_cols = [
            "code_system", "year", "category", "change_type", "code",
            "description", "advice", "crosswalk_code", "new_descriptor",
        ]
        deleted_cols = [
            "code", "description", "deleted_date", "substitute_code",
            "archived_info", "code_system",
        ]
        changes_df = pd.DataFrame(
            [
                {"code_system": "CPT", "year": 2023, "category": "",
                 "change_type": "Deleted", "code": "49560",
                 "description": "Hernia", "advice": "",
                 "crosswalk_code": "49591", "new_descriptor": ""},
                # Pollution — these should be dropped by the shape filter.
                {"code_system": "HCPCS", "year": 2021, "category": "",
                 "change_type": "Deleted", "code": "MA",
                 "description": "modifier", "advice": "",
                 "crosswalk_code": "", "new_descriptor": ""},
                {"code_system": "HCPCS", "year": 2021, "category": "",
                 "change_type": "Deleted", "code": "ME",
                 "description": "modifier", "advice": "",
                 "crosswalk_code": "", "new_descriptor": ""},
                {"code_system": "HCPCS", "year": 2021, "category": "",
                 "change_type": "Deleted", "code": "MD",
                 "description": "modifier", "advice": "",
                 "crosswalk_code": "", "new_descriptor": ""},
            ],
            columns=changes_cols,
        )
        deleted_df = pd.DataFrame(
            [
                {"code": "49560", "description": "Hernia",
                 "deleted_date": "1-Jan-23", "substitute_code": "49591",
                 "archived_info": "", "code_system": "CPT"},
                # Pollution.
                {"code": "GD", "description": "modifier",
                 "deleted_date": "1-Jan-21", "substitute_code": "",
                 "archived_info": "", "code_system": "HCPCS"},
                {"code": "645", "description": "stray",
                 "deleted_date": "1-Jan-21", "substitute_code": "",
                 "archived_info": "", "code_system": "HCPCS"},
            ],
            columns=deleted_cols,
        )
        changes_path = tmp / "code_changes.csv"
        deleted_path = tmp / "deleted_codes.csv"
        changes_df.to_csv(changes_path, index=False)
        deleted_df.to_csv(deleted_path, index=False)

        # A note whose text, without the shape filter, would be obliterated:
        # 'Name', 'male', 'female', 'MAIN', 'MD', 'Same', 'media' all match
        # short HCPCS modifier entries under IGNORECASE.
        in_path = tmp / "notes.csv"
        out_path = tmp / "notes_clean.csv"
        pristine_text = (
            "Patient Name: Frank  Sex: male  Hansen, MD  Same diagnosis  "
            "MAIN OR  female patient  media  primary  "
            "Prior 49560 repair.  EGD 43235 today."
        )
        pd.DataFrame([
            {"NOTE_ID": "n1", "NOTE_TEXT": pristine_text, "CPT_CODES": "43235"},
        ]).to_csv(in_path, index=False)

        cmd = [
            sys.executable, "-m",
            "cpt_rec.common.preprocess.code_leakage_scrubber",
            "--input", str(in_path),
            "--output", str(out_path),
            "--kb", str(kb_path),
            "--history-changes", str(changes_path),
            "--history-deleted", str(deleted_path),
            "--log-level", "INFO",
        ]
        import os
        env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        assert res.returncode == 0, (
            f"CLI failed:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}"
        )
        # INFO log should announce the filter kicking in.
        assert "filtered" in res.stderr or "filtered" in res.stdout, (
            f"expected a 'filtered ... non-procedure-code token' INFO line, "
            f"got:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}"
        )

        out = pd.read_csv(out_path)
        t = out.loc[out["NOTE_ID"] == "n1", "NOTE_TEXT"].item()

        # English words must all survive.
        for word in ("Name", "Frank", "Sex", "male", "Hansen", "MD",
                     "Same", "diagnosis", "MAIN", "female", "patient",
                     "media", "primary"):
            assert word in t, (
                f"expected {word!r} to survive scrubbing but it's gone.\n"
                f"  in : {pristine_text!r}\n"
                f"  out: {t!r}"
            )

        # Real codes must still be masked.
        assert "43235" not in t, "active-KB code must still be masked"
        assert "49560" not in t, "retired code from history must still be masked"
        assert t.count(MASK) == 2, f"expected 2 masks, got: {t!r}"
    print("PASS: test_cli_with_polluted_history_still_scrubs_only_real_codes")


# ---------------------------------------------------------------------------
# History-expansion: CodeHistory adds retired/deleted codes to the scrub vocab
# ---------------------------------------------------------------------------
def _write_tiny_history_csvs(tmp: Path) -> tuple[Path, Path]:
    """Build the two history CSVs with the exact schemas used in production."""
    changes_cols = [
        "code_system", "year", "category", "change_type", "code",
        "description", "advice", "crosswalk_code", "new_descriptor",
    ]
    deleted_cols = [
        "code", "description", "deleted_date", "substitute_code",
        "archived_info", "code_system",
    ]
    # 49560 is a well-known retired hernia code whose substitute (also
    # retired by 2023) is modeled here by 99991 just to keep the fixture
    # self-contained.  The key behaviour: 49560 is NOT in the active KB
    # but MUST still be scrubbed.
    changes_df = pd.DataFrame(
        [{
            "code_system": "CPT", "year": 2023, "category": "",
            "change_type": "Deleted", "code": "49560",
            "description": "Hernia repair", "advice": "",
            "crosswalk_code": "49591", "new_descriptor": "",
        }],
        columns=changes_cols,
    )
    deleted_df = pd.DataFrame(
        [{
            "code": "49560", "description": "Hernia repair",
            "deleted_date": "1-Jan-23", "substitute_code": "49591",
            "archived_info": "", "code_system": "CPT",
        }],
        columns=deleted_cols,
    )
    c_path = tmp / "code_changes.csv"
    d_path = tmp / "deleted_codes.csv"
    changes_df.to_csv(c_path, index=False)
    deleted_df.to_csv(d_path, index=False)
    return c_path, d_path


def test_history_codes_expands_vocab():
    """Direct unit test of _run_one's history_codes= path.

    49560 is NOT in our tiny active KB — so without history, the scrubber
    would miss it.  Passing history_codes={"49560", ...} via _run_one
    must add it to the vocabulary and redact it.
    """
    from cpt_rec.common.preprocess.code_leakage_scrubber import _run_one

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_path = tmp / "notes.csv"
        out_path = tmp / "out.csv"
        pd.DataFrame([
            {
                "NOTE_ID": "n1",
                "NOTE_TEXT": "Procedure Performed: 49560 hernia repair.",
                # NB: GT column blank so the default _harvest_extra_codes
                # path can't rescue us — the code must come from history.
                "CPT_CODES": "",
            },
        ]).to_csv(in_path, index=False)

        # Active KB that does NOT contain 49560.
        kb_codes = {"43235", "45378"}

        _run_one(
            input_csv=in_path,
            output_csv=out_path,
            kb_codes=kb_codes,
            history_codes={"49560"},
        )
        out = pd.read_csv(out_path)
        t = out.loc[out["NOTE_ID"] == "n1", "NOTE_TEXT"].item()
        assert "49560" not in t, f"expected 49560 masked, got: {t}"
        assert MASK in t
        assert int(out.loc[out["NOTE_ID"] == "n1", "N_CODES_REDACTED"].item()) == 1
    print("PASS: test_history_codes_expands_vocab")


def test_cli_roundtrip_with_history_flags():
    """End-to-end CLI call with --history-changes and --history-deleted.

    Proves that a retired code (49560) which is absent from both the
    active KB and the input CSV's CPT_CODES column still gets masked
    because CodeHistory was loaded from the two ledger CSVs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Synthetic KB — 49560 NOT in it.
        kb_df = pd.DataFrame(
            [
                {"code": "43235", "code_description": "EGD",
                 "code_lay_term": "EGD", "code_system": "CPT"},
                {"code": "49591", "code_description": "Anterior hernia",
                 "code_lay_term": "Hernia", "code_system": "CPT"},
            ]
        )
        for lvl in range(1, 7):
            kb_df[f"code_range_{lvl}"] = ""
            kb_df[f"code_range_{lvl}_description"] = ""
        kb_path = tmp / "kb.csv"
        kb_df.to_csv(kb_path, index=False)

        changes_path, deleted_path = _write_tiny_history_csvs(tmp)

        in_path = tmp / "notes.csv"
        out_path = tmp / "notes_clean.csv"
        pd.DataFrame([
            {
                "NOTE_ID": "n1",
                "NOTE_TEXT": "History: prior 49560 repair.  EGD 43235 today.",
                "CPT_CODES": "43235",  # 49560 deliberately absent here
            },
        ]).to_csv(in_path, index=False)

        cmd = [
            sys.executable, "-m",
            "cpt_rec.common.preprocess.code_leakage_scrubber",
            "--input", str(in_path),
            "--output", str(out_path),
            "--kb", str(kb_path),
            "--history-changes", str(changes_path),
            "--history-deleted", str(deleted_path),
            "--log-level", "WARNING",
        ]
        import os
        env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        assert res.returncode == 0, (
            f"CLI failed:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}"
        )

        out = pd.read_csv(out_path)
        t = out.loc[out["NOTE_ID"] == "n1", "NOTE_TEXT"].item()
        assert "49560" not in t, (
            f"49560 should have been masked via CodeHistory vocab, got: {t}"
        )
        assert "43235" not in t, "active KB code must still be masked too"
        # The note should now contain two <CPT_MASK> tokens.
        assert t.count(MASK) == 2, f"expected 2 masks, got: {t!r}"
    print("PASS: test_cli_roundtrip_with_history_flags")


# ---------------------------------------------------------------------------
# End-to-end check on the real sample DEID file
# ---------------------------------------------------------------------------
def test_post_scrub_audit_on_sample_deid():
    """
    Apply the scrubber to the corpus at CPT_REC_SAMPLE_CSV using the
    real KB and assert that zero ground-truth CPT codes appear verbatim in
    any cleaned NOTE_TEXT.

    Pre-scrub baseline (measured by scripts/audit_cpt_leakage.py):
        144/341 codes leak, 103/200 notes with any leak.
    Acceptance criterion: post-scrub, 0/341 codes leak in any note.
    """
    if not SAMPLE_CSV.exists():
        skip(f"needs a note corpus at {SAMPLE_CSV}")
        return
    if not KB_CSV.exists():
        skip(f"needs a KB at {KB_CSV}")
        return

    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
    from cpt_rec.common.preprocess.code_leakage_scrubber import (
        CodeLeakageScrubber,
    )
    from cpt_rec.common.preprocess.code_utils import parse_proc_codes

    kb = CodeKnowledgeBase.from_csv(KB_CSV, build_index=False)
    df = pd.read_csv(SAMPLE_CSV)
    # Harvest extra codes from GT labels — mirrors the CLI's defensive
    # behavior so retired CPT codes absent from the 2025 KB are also
    # scrubbed (e.g. 49560 / 49585 / 49568 pre-2023 hernia-family codes).
    extra: set[str] = set()
    for raw in df["CPT_CODES"].dropna().astype(str):
        extra.update(parse_proc_codes(raw))
    scrubber = CodeLeakageScrubber(kb.codes, extra_codes=extra)

    total_codes = 0
    leaked_after = 0
    leaked_notes_after = 0
    examples: list[str] = []

    for _, row in df.iterrows():
        gt_codes = parse_proc_codes(row["CPT_CODES"])
        if not gt_codes:
            continue
        cleaned = scrubber.scrub(row["NOTE_TEXT"] if isinstance(row["NOTE_TEXT"], str) else "").text
        any_leak_this_row = False
        for code in gt_codes:
            total_codes += 1
            # Use the *audit script's* stricter digit-only boundary so that
            # suffixed codes like "49621L" (which attach a laterality
            # modifier directly to the digits) still count as leaks.  The
            # scrubber must handle these; a weaker alphanumeric-boundary
            # check here would hide them from this regression test.
            pat = re.compile(rf"(?<!\d){re.escape(code)}(?!\d)")
            m = pat.search(cleaned)
            if m:
                leaked_after += 1
                any_leak_this_row = True
                if len(examples) < 3:
                    ctx = cleaned[max(0, m.start() - 60):m.end() + 60]
                    examples.append(f"code={code}  ...{ctx.replace(chr(10), ' ')}...")
        if any_leak_this_row:
            leaked_notes_after += 1

    print(f"  post-scrub: {leaked_after}/{total_codes} codes still leak "
          f"across {leaked_notes_after} notes")
    if examples:
        print("  first few residual leaks:")
        for ex in examples:
            print(f"    {ex}")
    assert leaked_after == 0, (
        f"Expected zero ground-truth codes in cleaned text, got "
        f"{leaked_after}/{total_codes}."
    )
    # Also sanity: placeholder appears in at least some rows.
    any_cleaned = df["NOTE_TEXT"].fillna("").astype(str).map(
        lambda t: scrubber.scrub(t).text
    )
    assert any_cleaned.str.contains(re.escape(MASK)).any(), (
        "Sample DEID has known leaks — expected at least one <CPT_MASK> placeholder."
    )
    print("PASS: test_post_scrub_audit_on_sample_deid")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Code-Leakage Scrubber — Test Suite")
    print("=" * 60)
    print()

    print("--- Unit: masking ---")
    test_mask_basic_cpt_codes()
    test_mask_hcpcs_codes()
    test_preserve_non_kb_5digit_tokens()
    test_preserve_icd_bracketed_codes()
    test_preserve_embedded_digit_runs()
    print()

    print("--- Unit: PR-tail stripping ---")
    test_strip_pr_template_tail_single_line()
    test_strip_pr_template_tail_multi_proc_collapsed()
    test_no_pr_tail_when_pattern_absent()
    print()

    print("--- Unit: alpha-suffix / stray-prefix handling ---")
    test_mask_codes_with_alpha_suffix()
    test_mask_codes_with_glued_prefix_letter()
    test_preserve_long_prefix_glued_to_code()
    test_real_data_epic_crush_patterns()
    test_mask_codes_with_multiplicand_suffix()
    test_preserve_longer_alpha_runs_after_code()
    test_strip_stray_letter_prefix_in_cpt_marker()
    test_stray_prefix_does_not_overreach()
    print()

    print("--- Unit: invariants ---")
    test_extra_codes_for_retired_gt_codes()
    test_idempotence()
    test_empty_and_nan_input()
    test_trie_regex_equivalent_to_flat_alternation()
    test_scrub_series_parallel_matches_serial()
    print()

    print("--- CLI roundtrip (synthetic) ---")
    test_cli_roundtrip_synthetic()
    print()

    print("--- Vocab shape filter (reject modifiers / prose) ---")
    test_vocab_shape_filter_drops_non_procedure_codes()
    test_short_modifier_tokens_do_not_mask_english_words()
    test_cli_with_polluted_history_still_scrubs_only_real_codes()
    print()

    print("--- History-expansion (retired codes via CodeHistory) ---")
    test_history_codes_expands_vocab()
    test_cli_roundtrip_with_history_flags()
    print()

    print("--- End-to-end: sample DEID post-scrub audit ---")
    test_post_scrub_audit_on_sample_deid()
    print()

    print("=" * 60)
    print("ALL LEAKAGE-SCRUBBER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
