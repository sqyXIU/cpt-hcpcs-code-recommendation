#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Tests for code parsing, label normalization, and the code knowledge base.

Run from project root:
    PYTHONPATH=src python3 tests/test_codes_and_kb.py

The knowledge-base tests run against ``tests/fixtures/mini_kb.csv``, a small
synthetic KB that ships with this repository, so they pass on a clean clone
with no licensed data present.  Tests that need a real corpus of operative
notes skip when that corpus is absent — see ``docs/DATA.md``.
"""

from __future__ import annotations

from _skip import skip

import logging
import os
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Synthetic KB that ships with the repo.  Resolved relative to this file so the
# tests pass no matter which directory pytest is invoked from.
FIXTURE_KB = Path(__file__).resolve().parent / "fixtures" / "mini_kb.csv"

# The CMS public-domain HCPCS Level II knowledge base that ships in data/kb/.
PUBLIC_KB = Path(__file__).resolve().parents[1] / "data" / "kb" / "hcpcs_level2_public.csv"

# A corpus of operative notes, which this repository does not distribute.  Point
# CPT_REC_SAMPLE_CSV at your own normalized corpus to exercise the tests below
# that need real notes; they skip when it is unset or missing.
SAMPLE_CSV = Path(os.environ.get("CPT_REC_SAMPLE_CSV", "data/notes/sample_notes.csv"))

# The KB to normalize that corpus against.  Defaults to the shipped public KB;
# point CPT_REC_KB_CSV at a licensed build (see scripts/build_kb.py) to resolve
# CPT Level I codes as well.
CORPUS_KB = Path(os.environ.get("CPT_REC_KB_CSV", str(PUBLIC_KB)))


# ===================================================================
# code_utils
# ===================================================================

def test_parse_proc_codes():
    from cpt_rec.common.preprocess.code_utils import parse_proc_codes

    # Basic extraction
    assert parse_proc_codes("43235, 45378") == ["43235", "45378"]
    assert parse_proc_codes("52332, 52353, 74420") == ["52332", "52353", "74420"]
    # HCPCS
    assert parse_proc_codes("J0585") == ["J0585"]
    assert parse_proc_codes("L8680, L8687") == ["L8680", "L8687"]
    # Cat III
    assert parse_proc_codes("0479T") == ["0479T"]
    # PLA / MAAA
    assert parse_proc_codes("0001U") == ["0001U"]
    assert parse_proc_codes("0002M") == ["0002M"]
    # Deduplication
    assert parse_proc_codes("43235, 43235, 45378") == ["43235", "45378"]
    # Edge cases
    assert parse_proc_codes("") == []
    assert parse_proc_codes(None) == []
    assert parse_proc_codes(float("nan")) == []
    # Should NOT extract from modifier-appended codes (not standalone)
    assert parse_proc_codes("52235BL") == []  # 52235 is embedded, BL follows
    assert parse_proc_codes("67036G25") == []  # G25 looks like HCPCS but isn't standalone
    print("PASS: test_parse_proc_codes")


def test_strip_modifier():
    from cpt_rec.common.preprocess.code_utils import strip_modifier

    # Trailing alpha modifiers on 5-digit CPT
    assert strip_modifier("52235BL") == "52235"
    assert strip_modifier("27130A") == "27130"
    assert strip_modifier("45395R") == "45395"
    assert strip_modifier("50545R") == "50545"
    assert strip_modifier("55866R") == "55866"
    assert strip_modifier("61598A") == "61598"
    assert strip_modifier("63048M") == "63048"

    # Multi-char modifier
    assert strip_modifier("33405TR") == "33405"

    # G-prefixed modifier (e.g. G25 = HCPCS modifier appended)
    assert strip_modifier("67036G25") == "67036"

    # Dot-separated (quantity/unit notation)
    assert strip_modifier("66999.11") == "66999"

    # Already valid — return as-is
    assert strip_modifier("43235") == "43235"
    assert strip_modifier("0479T") == "0479T"
    assert strip_modifier("J0585") == "J0585"
    assert strip_modifier("L8680") == "L8680"
    assert strip_modifier("S2068") == "S2068"
    assert strip_modifier("0001U") == "0001U"

    # Category III codes that look like 5-digit + T
    # "35820T" — 35820 is valid 5-digit CPT, T is modifier -> strip to 35820
    assert strip_modifier("35820T") == "35820"
    # "49000T" — 49000 is valid 5-digit CPT, T is modifier -> strip to 49000
    assert strip_modifier("49000T") == "49000"
    # "61154T" — 61154 is valid 5-digit CPT
    assert strip_modifier("61154T") == "61154"

    # HCPCS with trailing digit (A44320 -> A4432)
    assert strip_modifier("A44320") == "A4432"

    # Non-code tokens -> None
    assert strip_modifier("PBHANDEXT") is None
    assert strip_modifier("ROBOT39") is None
    assert strip_modifier("") is None

    print("PASS: test_strip_modifier")


def test_normalize_code_list():
    from cpt_rec.common.preprocess.code_utils import normalize_code_list

    # Basic case
    acc, unr, drp = normalize_code_list("43235, 45378")
    assert acc == ["43235", "45378"]
    assert unr == []
    assert drp == []

    # With modifiers
    acc, unr, drp = normalize_code_list("52235BL, 27130A, J0585")
    assert acc == ["52235", "27130", "J0585"]
    assert drp == []

    # With non-parseable tokens
    acc, unr, drp = normalize_code_list("11012, PBHANDEXT")
    assert acc == ["11012"]
    assert drp == ["PBHANDEXT"]

    # With KB validation
    fake_kb = {"43235", "45378"}
    acc, unr, drp = normalize_code_list("43235, 45378, 99999", valid_codes=fake_kb)
    assert acc == ["43235", "45378"]
    assert unr == ["99999"]

    # Deduplication after modifier stripping
    acc, unr, drp = normalize_code_list("27130, 27130A")
    assert acc == ["27130"]  # second is duplicate after stripping

    # Edge cases
    acc, unr, drp = normalize_code_list("")
    assert acc == [] and unr == [] and drp == []
    acc, unr, drp = normalize_code_list(None)
    assert acc == [] and unr == [] and drp == []

    print("PASS: test_normalize_code_list")


def test_is_valid_code_format():
    from cpt_rec.common.preprocess.code_utils import is_valid_code_format

    assert is_valid_code_format("43235") is True
    assert is_valid_code_format("0479T") is True
    assert is_valid_code_format("2029F") is True
    assert is_valid_code_format("0001U") is True
    assert is_valid_code_format("0002M") is True
    assert is_valid_code_format("J0585") is True
    assert is_valid_code_format("A0428") is True
    assert is_valid_code_format("52235BL") is False
    assert is_valid_code_format("PBHANDEXT") is False
    assert is_valid_code_format("") is False
    print("PASS: test_is_valid_code_format")


# ===================================================================
# label_normalizer (needs a note corpus)
# ===================================================================

def test_normalize_split_sample():
    """Run normalize_split over a real note corpus and verify the output schema.

    Skips unless CPT_REC_SAMPLE_CSV points at a corpus; this repository does not
    distribute clinical notes.
    """
    import pandas as pd
    from cpt_rec.common.preprocess.label_normalizer import (
        normalize_split,
        compute_code_frequency_stats,
    )
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    if not SAMPLE_CSV.exists():
        skip(f"needs a note corpus at {SAMPLE_CSV}; set CPT_REC_SAMPLE_CSV")
        return

    kb = CodeKnowledgeBase.from_csv(CORPUS_KB, build_index=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "sample_normalized.csv"
        counters = normalize_split(
            input_csv=SAMPLE_CSV,
            output_csv=out_csv,
            kb_codes=kb.codes,
            min_tokens=100,
            token_col="NOTE_TEXT_TOKENS",
            code_col="CPT_CODES",
        )
        print(f"  Normalization counters: {counters}")

        assert out_csv.exists(), "Output CSV not created"
        df = pd.read_csv(out_csv, dtype=str, nrows=20)

        # Check expected columns exist
        for col in ["CPT_CODES_RAW", "proc_codes", "n_codes", "codes_unresolved", "codes_dropped"]:
            assert col in df.columns, f"Missing column: {col}"

        # Verify proc_codes are pipe-separated valid codes
        for _, row in df.iterrows():
            codes = [c for c in str(row["proc_codes"]).split("|") if c]
            assert len(codes) > 0, "Row with 0 accepted codes should have been filtered"
            for c in codes:
                assert c in kb.codes or len(c) == 5, \
                    f"Code {c} not in KB (from proc_codes: {row['proc_codes']})"

        print(f"  Output rows: {counters['rows_out']}, sample verified OK")

        # Test stats
        stats_csv = Path(tmpdir) / "stats.csv"
        stats_df = compute_code_frequency_stats(out_csv, stats_csv)
        assert len(stats_df) > 0
        assert set(stats_df.columns) >= {"code", "frequency", "cumulative_freq", "rank", "bin"}
        assert set(stats_df["bin"].unique()).issubset({"head", "torso", "tail"})
        print(f"  Stats: {len(stats_df)} unique codes, bins: {stats_df['bin'].value_counts().to_dict()}")

    print("PASS: test_normalize_split_sample")


# ===================================================================
# code_kb
# ===================================================================

def test_code_kb_loading():
    """CodeKnowledgeBase loading, lookups, and hierarchy on the shipped fixture."""
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    kb = CodeKnowledgeBase.from_csv(FIXTURE_KB, build_index=False)
    assert len(kb) == 24, f"fixture should hold 24 unique codes, got {len(kb)}"
    print(f"  Loaded {len(kb)} unique codes from {FIXTURE_KB.name}")

    # Membership
    assert "DEMO022" in kb
    assert "J0585" in kb
    assert "ZZZZZ" not in kb

    # Description
    desc = kb.description("DEMO022")
    assert desc is not None
    assert "polyp" in desc.lower()
    print(f"  DEMO022 desc: {desc[:80]}...")

    # Short description is the first sentence, never longer than the full text
    short = kb.short_description("DEMO022")
    assert short is not None
    assert len(short) <= len(desc)

    # Code system, and the deprecated ``category`` alias for it
    assert kb.system("DEMO022") == "DEMO"
    assert kb.system("J0585") == "HCPCS"
    assert kb.category("J0585") == kb.system("J0585")

    # Hierarchy: broadest range first, most specific last
    h = kb.hierarchy("DEMO022")
    assert len(h) == 2, f"expected 2 hierarchy levels, got {h}"
    assert h[0][0] == "DEMO001-DEMO099"
    assert h[1][0] == "DEMO020-DEMO029"
    print(f"  DEMO022 hierarchy ({len(h)} levels):")
    for rng, desc_h in h:
        print(f"    {rng}: {desc_h[:60]}")

    # The real HCPCS rows carry a hierarchy too
    assert len(kb.hierarchy("J0585")) >= 1

    # Family = deepest range; family_codes = every code sharing it
    fam = kb.family("DEMO022")
    assert fam == "DEMO020-DEMO029"
    fam_codes = kb.family_codes("DEMO022")
    assert fam_codes == {"DEMO020", "DEMO021", "DEMO022", "DEMO023"}, fam_codes
    print(f"  DEMO022 family '{fam}': {len(fam_codes)} codes")

    # A code whose only range is the top level still resolves to a family
    assert kb.family("DEMO150") == "DEMO100-DEMO199"

    # Unknown codes return empty, not an exception
    assert kb.description("ZZZZZ") is None
    assert kb.hierarchy("ZZZZZ") == []
    assert kb.family_codes("ZZZZZ") == set()

    print("PASS: test_code_kb_loading")


def test_code_kb_tfidf_search():
    """TF-IDF similarity search over the shipped fixture."""
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    kb = CodeKnowledgeBase.from_csv(FIXTURE_KB, build_index=True)

    results = kb.search("lower endoscopy removal of polyp by snare technique", top_k=5)
    assert len(results) == 5
    codes_found = [code for code, score in results]
    print(f"  Search 'polyp snare': {results[:3]}")
    assert "DEMO022" in codes_found, f"expected DEMO022 in top 5, got {codes_found}"

    # Self-retrieval: a code's own description must retrieve that code first
    for probe in ("DEMO022", "DEMO110", "J0585"):
        desc = kb.description(probe)
        assert desc is not None
        top = [c for c, _ in kb.search(desc, top_k=3)]
        assert top[0] == probe, f"self-retrieval for {probe} returned {top}"
    print("  Self-retrieval holds for DEMO022, DEMO110, J0585")

    # Batch search returns one result list per query, with non-negative scores
    queries = ["upper endoscopy with biopsy", "excision of skin lesion", "chest radiograph"]
    batch_results = kb.batch_search(queries, top_k=3)
    assert len(batch_results) == len(queries)
    for query, res in zip(queries, batch_results):
        assert len(res) == 3, f"expected 3 results for '{query}'"
        print(f"  Batch '{query}': {res[:2]}")
        for code, score in res:
            assert score >= 0, f"negative score for '{query}': {code}={score}"

    print("PASS: test_code_kb_tfidf_search")


def test_public_hcpcs_kb_is_free_of_licensed_descriptors():
    """The shipped KB must stay HCPCS Level II only.

    CPT Level I descriptors are AMA copyright and are not distributed with this
    repository.  Every code in data/kb/ therefore has to be a letter-prefixed
    HCPCS Level II code, and the commercial lay-term column has to be empty.
    This test is the guard that keeps a licensed build from being committed by
    accident; see docs/DATA.md.
    """
    import pandas as pd
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    assert PUBLIC_KB.exists(), f"public KB missing: {PUBLIC_KB}"
    df = pd.read_csv(PUBLIC_KB, dtype=str).fillna("")

    numeric = df[~df["code"].str.match(r"^[A-V]", na=False)]
    assert numeric.empty, \
        f"{len(numeric)} non-HCPCS code(s) in the public KB: {list(numeric['code'][:10])}"

    assert (df["code_system"] == "HCPCS").all(), \
        f"unexpected code_system values: {sorted(set(df['code_system']))}"

    lay = df["code_lay_term"].str.strip()
    assert (lay == "").all(), \
        f"{int((lay != '').sum())} row(s) still carry a licensed lay term"

    kb = CodeKnowledgeBase.from_csv(PUBLIC_KB, build_index=False)
    assert len(kb) >= 7000, f"expected >= 7000 HCPCS codes, got {len(kb)}"
    print(f"  Public KB: {len(kb)} HCPCS Level II codes, no licensed descriptors")

    print("PASS: test_public_hcpcs_kb_is_free_of_licensed_descriptors")


def test_make_op_note_snippet_still_works():
    """Verify that the refactored make_op_note_snippet still imports correctly."""
    from cpt_rec.common.retriever.make_op_note_snippet import (
        parse_proc_codes,
        build_code_mask_pattern,
        sent_tokenize_clinical,
        make_overlapping_sentence_windows,
    )
    # Quick smoke test
    codes = parse_proc_codes("43235, 45378")
    assert codes == ["43235", "45378"]

    pat = build_code_mask_pattern(codes)
    assert pat is not None
    assert pat.sub("[CODE]", "performed 43235 and 45378") == "performed [CODE] and [CODE]"

    sents = sent_tokenize_clinical("Patient was prepped. The scope was inserted. No complications.")
    assert len(sents) >= 2

    windows = make_overlapping_sentence_windows(sents, max_words=50, overlap_words=10)
    assert len(windows) >= 1

    print("PASS: test_make_op_note_snippet_still_works")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Codes and knowledge base — test suite")
    print("=" * 60)
    print()

    print("--- code_utils ---")
    test_parse_proc_codes()
    test_strip_modifier()
    test_normalize_code_list()
    test_is_valid_code_format()
    print()

    print("--- label_normalizer ---")
    test_normalize_split_sample()
    print()

    print("--- code_kb ---")
    test_code_kb_loading()
    test_code_kb_tfidf_search()
    test_public_hcpcs_kb_is_free_of_licensed_descriptors()
    print()

    print("--- Regression: make_op_note_snippet import ---")
    test_make_op_note_snippet_still_works()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
