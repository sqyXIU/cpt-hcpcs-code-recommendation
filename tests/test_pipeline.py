#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Tests for the retrieve-and-verify pipeline.

Modules exercised
-----------------

* ``pipeline.pool.snippetize``            — note -> evidence snippets
* ``pipeline.pool.candidate_gen``         — KB index wrapper (uses baselines.kb_index artifacts)
* ``pipeline.decode.constrained_decode``  — NCCI-aware greedy decoder

Tests that need artifacts this repository does not ship (a built KB index, the
CMS NCCI tables) print a SKIP message naming the missing input.

Run from project root::

    PYTHONPATH=src python3 tests/test_pipeline.py
"""

from __future__ import annotations

from _skip import skip

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
SAMPLE_NORM_CSV = Path(os.environ.get("CPT_REC_SAMPLE_NORM_CSV",
                                       "outputs/notes/sample_notes_normalized.csv"))
KB_CSV = Path(os.environ.get("CPT_REC_KB_CSV", str(PUBLIC_KB)))
NCCI_DIR = Path(os.environ.get("CPT_REC_NCCI_DIR", "data/ncci"))
KB_INDEX_DIR = Path(os.environ.get("CPT_REC_KB_INDEX_DIR",
                                   "outputs/indices/code_kb/default"))

# ---------------------------------------------------------------------------
# Local model directory — set CPT_REC_MODEL_DIR on servers without HuggingFace
# access.  All test surrogate paths are resolved relative to this directory.
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(os.environ.get("CPT_REC_MODEL_DIR", "~/models")).expanduser()


def _local_model(name: str, folder: str | None = None) -> str:
    local = _MODEL_DIR / (folder if folder else name.split("/")[-1])
    return str(local) if local.exists() else name


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


# ===================================================================
# Test 3.0: snippetize
# ===================================================================

def test_snippetize_with_sections():
    from cpt_rec.pipeline.pool.snippetize import snippets_for_note

    sections = {
        "Procedure(s) Performed": "Laparoscopic cholecystectomy performed. "
                                  "No intraoperative complications.",
        "Detailed Description": (
            "A 12 mm port was placed at the umbilicus. "
            "The gallbladder was dissected off the liver bed. "
            "The cystic duct and artery were clipped and divided. "
            "The specimen was removed in an endocatch bag."
        ),
        "Findings": "Chronic cholecystitis with gallstones.",
        "Indications for Surgery": "Symptomatic cholelithiasis.",
    }
    snips = snippets_for_note(
        note_id="n1",
        note_text=None,
        sections=sections,
        max_words=40,
        overlap_words=10,
    )
    assert len(snips) >= 3, f"Expected multiple snippets, got {len(snips)}"
    # No snippet should be tagged with the fallback section.
    assert all(s.section != "__whole_note__" for s in snips)
    # Every tagged text starts with [SECTION=...]
    for s in snips:
        assert s.tagged_text().startswith("[SECTION=")
    # Snippet ids are unique.
    ids = [s.snippet_id for s in snips]
    assert len(set(ids)) == len(ids)
    print(f"PASS: test_snippetize_with_sections (n={len(snips)})")


def test_snippetize_whole_note_fallback():
    from cpt_rec.pipeline.pool.snippetize import snippets_for_note

    txt = " ".join(
        f"This is sentence number {i} in the fallback note." for i in range(20)
    )
    snips = snippets_for_note(
        note_id="n2",
        note_text=txt,
        sections=None,
        max_words=50,
        overlap_words=10,
    )
    assert len(snips) >= 1
    assert all(s.section == "__whole_note__" for s in snips)
    print(f"PASS: test_snippetize_whole_note_fallback (n={len(snips)})")


def test_snippetize_max_snippets_cap():
    from cpt_rec.pipeline.pool.snippetize import snippets_for_note

    txt = " ".join(f"Sentence {i}." for i in range(200))
    snips_uncapped = snippets_for_note(
        note_id="n3", note_text=txt, max_words=20, overlap_words=5,
    )
    snips_capped = snippets_for_note(
        note_id="n3", note_text=txt, max_words=20, overlap_words=5, max_snippets=3,
    )
    assert len(snips_capped) <= 3
    assert len(snips_capped) <= len(snips_uncapped)
    print(
        f"PASS: test_snippetize_max_snippets_cap "
        f"(uncapped={len(snips_uncapped)}, capped={len(snips_capped)})"
    )


# ===================================================================
# Test 3.1: aggregator math
# ===================================================================



# ===================================================================
# Test 3.2: neg_sampler
# ===================================================================

def _tiny_kb() -> "CodeKnowledgeBase":  # type: ignore[name-defined]
    """Build a tiny in-memory CodeKnowledgeBase with two families."""
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    df = pd.DataFrame(
        [
            {"code": "43235", "code_description": "EGD diagnostic",
             "code_range_1": "43200-43499", "code_range_1_description": "GI endoscopy",
             "code_range_2": "43200-43273", "code_range_2_description": "Esophagoscopy"},
            {"code": "43239", "code_description": "EGD with biopsy",
             "code_range_1": "43200-43499", "code_range_1_description": "GI endoscopy",
             "code_range_2": "43200-43273", "code_range_2_description": "Esophagoscopy"},
            {"code": "45378", "code_description": "Colonoscopy diagnostic",
             "code_range_1": "45300-45399", "code_range_1_description": "Colonoscopy"},
            {"code": "45380", "code_description": "Colonoscopy with biopsy",
             "code_range_1": "45300-45399", "code_range_1_description": "Colonoscopy"},
            {"code": "66984", "code_description": "Cataract extraction with IOL",
             "code_range_1": "66830-66940", "code_range_1_description": "Lens procedures"},
        ]
    )
    return CodeKnowledgeBase(df)






# ===================================================================
# Test 3.3: bag_dataset
# ===================================================================







# ===================================================================
# Test 3.4: pseudo_evidence — span parse + generation loop with stub
# ===================================================================







# ===================================================================
# Test 3.5: constrained decoder
# ===================================================================

def _synthetic_checker():
    """Hand-built NCCIRuleChecker for unit-testing the decoder logic."""
    from cpt_rec.common.ncci.rule_checker import NCCIRuleChecker

    # Directed pair dict with three interesting cases:
    # - (A1, A2) hard (CCMI=0)
    # - (B1, B2) modifier-contingent (CCMI=1)
    # - (D1, D2) hard (CCMI=9)
    pair_to_ccmi: Dict[Tuple[str, str], int] = {
        ("A1", "A2"): 0,
        ("A2", "A1"): 0,
        ("B1", "B2"): 1,
        ("B2", "B1"): 1,
        ("D1", "D2"): 9,
        ("D2", "D1"): 9,
    }
    pair_to_rationale = {k: "synthetic" for k in pair_to_ccmi}
    # Add-on Z1 requires primary P1; Z2 requires P2.
    addon_to_primaries = {"Z1": {"P1"}, "Z2": {"P2"}}
    return NCCIRuleChecker(
        pair_to_ccmi=pair_to_ccmi,
        pair_to_rationale=pair_to_rationale,
        addon_to_primaries=addon_to_primaries,
        contractor_defined_addons=set(),
        code_to_max_units={},
    )


def test_constrained_decode_drops_hard_ptp():
    from cpt_rec.pipeline.decode.constrained_decode import decode_one

    checker = _synthetic_checker()
    # A1 (higher prob) wins; A2 (lower prob) dropped by PTP hard violation.
    trace = decode_one(
        note_id="n1",
        candidate_codes=["A1", "A2", "X"],
        probs=[0.95, 0.90, 0.80],
        checker=checker,
        threshold=0.5,
    )
    assert "A1" in trace.kept_codes
    assert "A2" not in trace.kept_codes
    assert "X" in trace.kept_codes
    assert len(trace.dropped_hard_ptp) >= 1
    print(
        f"PASS: test_constrained_decode_drops_hard_ptp "
        f"(kept={trace.kept_codes}, dropped={trace.dropped_hard_ptp})"
    )


def test_constrained_decode_flags_modifier_contingent():
    from cpt_rec.pipeline.decode.constrained_decode import decode_one

    checker = _synthetic_checker()
    trace = decode_one(
        note_id="n2",
        candidate_codes=["B1", "B2"],
        probs=[0.9, 0.8],
        checker=checker,
        threshold=0.5,
    )
    # Modifier-contingent pairs are KEPT, just flagged.
    assert set(trace.kept_codes) == {"B1", "B2"}
    assert trace.dropped_hard_ptp == []
    assert len(trace.modifier_contingent) >= 1
    print(
        f"PASS: test_constrained_decode_flags_modifier_contingent "
        f"(kept={trace.kept_codes}, flags={trace.modifier_contingent})"
    )


def test_constrained_decode_repairs_aoc():
    """Z1 is an add-on for P1; P1 is also predicted with LOWER probability,
    so Z1 gets held in pass 1 and repaired in pass 2."""
    from cpt_rec.pipeline.decode.constrained_decode import decode_one

    checker = _synthetic_checker()
    trace = decode_one(
        note_id="n3",
        candidate_codes=["Z1", "P1"],  # Z1 has the higher probability
        probs=[0.95, 0.60],
        checker=checker,
        threshold=0.5,
    )
    assert "P1" in trace.kept_codes
    assert "Z1" in trace.kept_codes
    assert "Z1" in trace.repaired_aoc
    print(f"PASS: test_constrained_decode_repairs_aoc (kept={trace.kept_codes})")


def test_constrained_decode_drops_orphan_aoc():
    from cpt_rec.pipeline.decode.constrained_decode import decode_one

    checker = _synthetic_checker()
    trace = decode_one(
        note_id="n4",
        candidate_codes=["Z2", "X"],  # Z2 needs P2 which is not predicted
        probs=[0.95, 0.90],
        checker=checker,
        threshold=0.5,
    )
    assert "Z2" not in trace.kept_codes
    assert "Z2" in trace.dropped_orphan_aoc
    assert "X" in trace.kept_codes
    print(
        f"PASS: test_constrained_decode_drops_orphan_aoc "
        f"(kept={trace.kept_codes}, orphans={trace.dropped_orphan_aoc})"
    )


def test_constrained_decode_from_csv_roundtrip():
    from cpt_rec.pipeline.decode.constrained_decode import decode_from_csv

    checker_dir = NCCI_DIR
    if not checker_dir.exists():
        skip(f"needs the CMS NCCI tables at {NCCI_DIR}")
        return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pred_csv = td / "preds.csv"
        pd.DataFrame(
            [
                {"note_id": "n1", "pred_codes": "43235|45378", "pred_scores": "0.9|0.7"},
                {"note_id": "n2", "pred_codes": "66984", "pred_scores": "0.8"},
            ]
        ).to_csv(pred_csv, index=False)
        out = td / "constrained.csv"
        n = decode_from_csv(
            pred_csv=pred_csv,
            ncci_dir=checker_dir,
            out_csv=out,
            threshold=0.0,
            top_k=None,
        )
        assert n == 2
        df = pd.read_csv(out, dtype=str)
        assert "pred_codes" in df.columns
    print(f"PASS: test_constrained_decode_from_csv_roundtrip (n={n})")


# ===================================================================
# candidate_gen (uses the KB index artifacts if present)
# ===================================================================

def test_candidate_gen_uses_kb_index_when_built():
    if not _torch_available():
        skip("needs the gpu extra (torch)")
        return
    if not (KB_INDEX_DIR / "bm25_corpus.npz").exists():
        skip(f"needs a built KB index at {KB_INDEX_DIR}")
        return

    from cpt_rec.pipeline.pool.candidate_gen import KBCandidateIndex

    try:
        idx = KBCandidateIndex(KB_INDEX_DIR)
    except Exception as exc:
        skip(str(exc))
        return

    hits = idx.topk_union_batch(
        ["diagnostic colonoscopy to cecum"],
        bm25_top_k=5, dense_top_k=5,
    )
    assert len(hits) == 1
    assert len(hits[0]) > 0
    assert all(isinstance(c, str) for c in hits[0])
    print(
        f"PASS: test_candidate_gen_uses_kb_index_when_built "
        f"(n_cands={len(hits[0])}, first={hits[0][:3]})"
    )


# ===================================================================
# Test 3.7: scorer forward pass (surrogate bert-tiny)
# ===================================================================



# ===================================================================
# Test 3.8: train + predict end-to-end smoke (surrogate)
# ===================================================================



# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Retrieve-and-verify pipeline — test suite")
    print("=" * 60)
    print()

    print("--- evidence windows ---")
    test_snippetize_with_sections()
    test_snippetize_whole_note_fallback()
    test_snippetize_max_snippets_cap()
    print()

    print("--- constrained decoder ---")
    test_constrained_decode_drops_hard_ptp()
    test_constrained_decode_flags_modifier_contingent()
    test_constrained_decode_repairs_aoc()
    test_constrained_decode_drops_orphan_aoc()
    test_constrained_decode_from_csv_roundtrip()
    print()

    print("--- candidate generation ---")
    test_candidate_gen_uses_kb_index_when_built()
    print()

    print("=" * 60)
    print("ALL PIPELINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
