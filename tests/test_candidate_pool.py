#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Unit tests for ``crossencoder_verifier.pool.candidate_pool`` — the candidate-pool
assembler and its source-tagged φ(n, c) features.

These tests lock the *behaviour* of the assembler so the readability refactor
(source-builder extraction, ``_CandidateSignals``, ``AssemblerConfig``) is
provably value-preserving: every φ feature here is hand-computed from a tiny
synthetic KB index with 2-D dense vectors, so the expected numbers can be read
straight off the formulas in :meth:`PoolAssembler._compute_feats`.

No torch / HuggingFace / KB-index artifacts: the real :class:`KBCandidateIndex` is
replaced by a duck-typed ``_FakeKBIndex`` exposing only the four members the
assembler actually touches (``codes``, ``code_to_row``, ``dense_vecs``,
``encode_queries``, ``bm25_topk_scored_batch``).

Run from project root::

    PYTHONPATH=src python3 tests/test_candidate_pool.py
    # or
    PYTHONPATH=src pytest tests/test_candidate_pool.py -q
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.pipeline.pool.candidate_pool import (
    AssemblerConfig,
    CandidateRecord,
    FEATURE_ORDERS,
    NotePool,
    feature_dim,
    features_to_vector,
)
from cpt_rec.pipeline.pool.snippetize import Snippet

_SQRT_HALF = 1.0 / np.sqrt(2.0)  # cos(45°) — the C33333 evidence cosine


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _tiny_kb() -> CodeKnowledgeBase:
    """Five-code KB: four CPT codes plus one HCPCS code (for ``is_hcpcs``)."""
    df = pd.DataFrame(
        [
            {"code": "11111", "code_description": "EGD diagnostic", "code_system": "CPT"},
            {"code": "22222", "code_description": "EGD with biopsy", "code_system": "CPT"},
            {"code": "33333", "code_description": "Colonoscopy diagnostic procedure",
             "code_system": "CPT"},
            {"code": "44444", "code_description": "Cataract extraction", "code_system": "CPT"},
            {"code": "A0021", "code_description": "Ambulance outside state", "code_system": "HCPCS"},
        ]
    )
    return CodeKnowledgeBase(df)


class _FakeKBIndex:
    """Minimal stand-in for :class:`KBCandidateIndex`.

    ``encode_queries`` returns a fixed snippet-embedding matrix (ignoring the
    text), and ``bm25_topk_scored_batch`` returns preset ``(code_row, score)``
    rows — so dense retrieval, best-evidence selection, and BM25 are all fully
    deterministic and the φ features become hand-computable.
    """

    def __init__(
        self,
        codes: List[str],
        dense_vecs: np.ndarray,
        snip_embs: np.ndarray,
        bm25_per_snippet: List[List[Tuple[int, float]]],
    ) -> None:
        self.codes = list(codes)
        self.code_to_row = {c: i for i, c in enumerate(self.codes)}
        self.dense_vecs = np.asarray(dense_vecs, dtype=np.float32)
        self._snip_embs = np.asarray(snip_embs, dtype=np.float32)
        self._bm25_per_snippet = bm25_per_snippet

    def encode_queries(self, queries: Sequence[str], batch_size: int = 64) -> np.ndarray:
        assert len(queries) == self._snip_embs.shape[0], "snippet count mismatch"
        return self._snip_embs

    def bm25_topk_scored_batch(
        self, query_texts: Sequence[str], top_k: int, query_chunk: int = 512
    ) -> List[List[Tuple[int, float]]]:
        assert len(query_texts) == len(self._bm25_per_snippet)
        return [row[:top_k] for row in self._bm25_per_snippet]


def _one_snippet(note_id: str = "n1") -> List[Snippet]:
    """A single whole-note snippet (its text is irrelevant — the fake index
    ignores it and returns a fixed embedding)."""
    return [
        Snippet(
            note_id=note_id,
            section="__whole_note__",
            snippet_index=0,
            start_sentence=0,
            end_sentence=1,
            n_words=5,
            text="colonoscopy to cecum with biopsy",
        )
    ]


def _build_pool(feature_version: str = "v1b") -> NotePool:
    """Assemble the canonical synthetic note used across the φ assertions.

    Layout (1 snippet, embedding ``[1, 0]``; ``bm25_top_k = dense_top_k = 2``):

    =======  ============  ========  =======  ========  =====
    code     dense vec     cos→snip  in dense in bm25   other
    =======  ============  ========  =======  ========  =====
    11111    [1, 0]        1.000     yes(r0)  yes(r0)   —
    22222    [0, 1]        0.000     no       yes(r1)   nbr
    33333    [s, s]        0.707     yes(r1)  no        nbr, llm
    44444    [-1, 0]      -1.000     no       no        —      (never pooled)
    A0021    [0, 1]        0.000     no       no        llm
    =======  ============  ========  =======  ========  =====

    so the pool is exactly ``{11111, 22222, 33333, A0021}`` (``44444`` is
    surfaced by no source and must be absent).
    """
    kb = _tiny_kb()
    codes = ["11111", "22222", "33333", "44444", "A0021"]
    dense_vecs = np.array(
        [[1.0, 0.0], [0.0, 1.0], [_SQRT_HALF, _SQRT_HALF], [-1.0, 0.0], [0.0, 1.0]]
    )
    snip_embs = np.array([[1.0, 0.0]])               # one snippet on the C11111 axis
    bm25_per_snippet = [[(0, 5.0), (1, 2.0)]]         # row0 best (score 5), row1 (score 2)
    index = _FakeKBIndex(codes, dense_vecs, snip_embs, bm25_per_snippet)

    cfg = AssemblerConfig(bm25_top_k=2, dense_top_k=2, neighbor_top_k=25, llm_norm_k=50)
    assembler = cfg.build_assembler(kb, index, feature_version=feature_version)

    pool = assembler.assemble_note(
        note_id="n1",
        snippets=_one_snippet("n1"),
        neighbor_ranked=[
            ("nbrA", {"22222", "33333"}),  # rank 0
            ("nbrB", {"33333"}),           # rank 1
        ],
        llm_map={
            "33333": (3, 0.9),
            "A0021": (5, 0.7),
            "99999": (1, 0.5),   # not in KB -> dropped by _llm_source
        },
    )
    assert pool is not None
    return pool


def _rec(pool: NotePool, code: str) -> CandidateRecord:
    matches = [r for r in pool.records if r.code == code]
    assert matches, f"code {code} not in pool {pool.codes()}"
    return matches[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pool_membership_is_source_union():
    """The pool is the union over dense ∪ bm25 ∪ neighbor ∪ llm; a code no
    source surfaces (44444) and an out-of-KB llm code (99999) are excluded."""
    pool = _build_pool()
    assert pool.note_id == "n1"
    assert pool.codes() == ["11111", "22222", "33333", "A0021"]  # sorted union
    assert "44444" not in pool.codes()
    assert "99999" not in pool.codes()
    print(f"PASS: test_pool_membership_is_source_union (pool={pool.codes()})")


def test_sources_tuple_per_candidate():
    """``sources`` is reported in canonical bm25,dense,neighbor,llm order and
    lists exactly the sources that surfaced each code."""
    pool = _build_pool()
    assert _rec(pool, "11111").sources == ("bm25", "dense")
    assert _rec(pool, "22222").sources == ("bm25", "neighbor")
    assert _rec(pool, "33333").sources == ("dense", "neighbor", "llm")
    assert _rec(pool, "A0021").sources == ("llm",)
    print("PASS: test_sources_tuple_per_candidate")


def test_phi_features_multi_source_code():
    """Hand-computed φ for 33333 — present in dense(rank1) + neighbor(rank0,
    both neighbours) + llm(rank3, conf0.9), absent from bm25."""
    pool = _build_pool()
    rec = _rec(pool, "33333")
    f = rec.feats

    # record-level evidence
    assert rec.best_snippet_idx == 0
    assert np.isclose(rec.best_snippet_cos, _SQRT_HALF)

    # bm25 absent -> present 0, rank sentinel 1.0, score 0.0
    assert f["kb_bm25_present"] == 0.0
    assert np.isclose(f["kb_bm25_rank"], 1.0)
    assert f["kb_bm25_score"] == 0.0
    # dense present at union-rank 1 of dense_top_k 2 -> 0.5; score is the cosine
    assert f["kb_dense_present"] == 1.0
    assert np.isclose(f["kb_dense_rank"], 0.5)
    assert np.isclose(f["kb_dense_score"], _SQRT_HALF)
    # neighbour: best rank 0 / 25 = 0.0; carried by 2 of 2 used neighbours -> 1.0
    assert f["nbr_present"] == 1.0
    assert np.isclose(f["nbr_rank"], 0.0)
    assert np.isclose(f["nbr_frac"], 1.0)
    # evidence + structure
    assert np.isclose(f["best_snip_cos"], _SQRT_HALF)
    assert f["best_snip_pos"] == 0.0           # single snippet
    assert np.isclose(f["n_sources"], 3.0 / 4.0)
    assert f["is_hcpcs"] == 0.0                 # CPT
    # llm: present, rank 3 / norm 50 = 0.06, confidence passthrough
    assert f["llm_present"] == 1.0
    assert np.isclose(f["llm_rank"], 3.0 / 50.0)
    assert np.isclose(f["llm_confidence"], 0.9)
    print("PASS: test_phi_features_multi_source_code")


def test_phi_features_bm25_and_dense_code():
    """Hand-computed φ for 11111 — top of both bm25 and dense (rank 0 each),
    best BM25 score in the note (-> normalized 1.0), cosine 1.0."""
    pool = _build_pool()
    f = _rec(pool, "11111").feats
    assert f["kb_bm25_present"] == 1.0
    assert np.isclose(f["kb_bm25_rank"], 0.0)
    assert np.isclose(f["kb_bm25_score"], 1.0)   # 5.0 / per-note max 5.0
    assert f["kb_dense_present"] == 1.0
    assert np.isclose(f["kb_dense_rank"], 0.0)
    assert np.isclose(f["kb_dense_score"], 1.0)
    assert f["nbr_present"] == 0.0
    assert f["llm_present"] == 0.0
    assert np.isclose(f["n_sources"], 2.0 / 4.0)
    print("PASS: test_phi_features_bm25_and_dense_code")


def test_phi_features_llm_only_hcpcs_code():
    """Hand-computed φ for A0021 — surfaced by llm only, and HCPCS so the
    ``is_hcpcs`` flag fires."""
    pool = _build_pool()
    f = _rec(pool, "A0021").feats
    assert f["kb_bm25_present"] == 0.0
    assert f["kb_dense_present"] == 0.0
    assert f["kb_dense_score"] == 0.0            # absent from dense -> 0, not the cosine
    assert f["nbr_present"] == 0.0
    assert f["llm_present"] == 1.0
    assert np.isclose(f["llm_rank"], 5.0 / 50.0)
    assert np.isclose(f["llm_confidence"], 0.7)
    assert f["is_hcpcs"] == 1.0
    assert np.isclose(f["n_sources"], 1.0 / 4.0)
    print("PASS: test_phi_features_llm_only_hcpcs_code")


def test_feats_dict_is_version_independent_superset():
    """``_compute_feats`` always emits the full superset (incl. ``llm_*``)
    regardless of feature_version; the version only selects the column order
    used to vectorize it."""
    pool_v1a = _build_pool(feature_version="v1a")
    f = _rec(pool_v1a, "33333").feats
    for key in ("llm_present", "llm_rank", "llm_confidence"):
        assert key in f, f"{key} missing from v1a feats superset"

    vec_a = features_to_vector(f, FEATURE_ORDERS["v1a"])
    vec_b = features_to_vector(f, FEATURE_ORDERS["v1b"])
    assert vec_a.shape == (feature_dim("v1a"),) == (14,)
    assert vec_b.shape == (feature_dim("v1b"),) == (17,)
    # v1b is v1a + 3 llm columns; the shared prefix must be identical.
    assert np.allclose(vec_a, vec_b[:14])
    print("PASS: test_feats_dict_is_version_independent_superset")


def test_empty_snippets_returns_none():
    """A note that yields no snippets produces no pool."""
    kb = _tiny_kb()
    index = _FakeKBIndex(
        ["11111"], np.array([[1.0, 0.0]]), np.zeros((0, 2)), []
    )
    assembler = AssemblerConfig().build_assembler(kb, index)
    assert assembler.assemble_note(note_id="n1", snippets=[]) is None
    print("PASS: test_empty_snippets_returns_none")


def test_neighbor_self_leakage_guard():
    """``exclude_note_id`` drops the note from its own BM25 neighbour list (the
    train index contains it), so its gold cannot leak in as a neighbour vote."""
    kb = _tiny_kb()
    codes = ["11111", "22222", "33333", "44444", "A0021"]
    dense_vecs = np.array(
        [[1.0, 0.0], [0.0, 1.0], [_SQRT_HALF, _SQRT_HALF], [-1.0, 0.0], [0.0, 1.0]]
    )
    index = _FakeKBIndex(codes, dense_vecs, np.array([[1.0, 0.0]]), [[(0, 5.0)]])
    assembler = AssemblerConfig(
        bm25_top_k=1, dense_top_k=1, neighbor_top_k=25
    ).build_assembler(kb, index, feature_version="v1a")

    # "n1" is its own top neighbour and is the *only* carrier of 33333.
    neighbor_ranked = [("n1", {"33333"}), ("nbrB", {"22222"})]

    kept = assembler.assemble_note(
        note_id="n1", snippets=_one_snippet("n1"),
        neighbor_ranked=neighbor_ranked, exclude_note_id=False,
    )
    dropped = assembler.assemble_note(
        note_id="n1", snippets=_one_snippet("n1"),
        neighbor_ranked=neighbor_ranked, exclude_note_id=True,
    )
    assert kept is not None and dropped is not None
    # With the self-neighbour kept, 33333 enters via the neighbour source...
    assert "neighbor" in _rec(kept, "33333").sources
    # ...with it excluded, 33333 is no longer surfaced by any source.
    assert "33333" not in dropped.codes()
    print("PASS: test_neighbor_self_leakage_guard")


def test_assembler_config_roundtrip_and_defaults():
    """``AssemblerConfig`` is the single source of truth for train↔predict
    parity: ``to_config_dict`` -> ``from_config_dict`` round-trips, and a
    config.json missing keys falls back to the historical defaults."""
    cfg = AssemblerConfig(
        bm25_top_k=30, dense_top_k=40, neighbor_top_k=15, llm_norm_k=100,
        max_snippets=16, snippet_max_words=200, snippet_overlap_words=50,
        encode_batch_size=128,
    )
    restored = AssemblerConfig.from_config_dict({"assembler": cfg.to_config_dict()})
    assert restored == cfg

    # Missing/empty -> all defaults (matches a pre-field config.json loading).
    defaults = AssemblerConfig.from_config_dict({})
    assert defaults == AssemblerConfig()
    # Partial override leaves the rest at defaults.
    partial = AssemblerConfig.from_config_dict({"assembler": {"bm25_top_k": 7}})
    assert partial.bm25_top_k == 7
    assert partial.dense_top_k == AssemblerConfig().dense_top_k
    print("PASS: test_assembler_config_roundtrip_and_defaults")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# S2 evidence-snippet selection
# ---------------------------------------------------------------------------


class _LenientFakeKBIndex(_FakeKBIndex):
    """``_FakeKBIndex`` that tolerates a shrinking snippet list.

    S2 selection drops snippets *between* ``encode_queries`` and the BM25 call,
    so the strict length asserts in the base fake would fire on a selecting run
    even though the assembler is behaving correctly.
    """

    def encode_queries(self, queries, batch_size: int = 64):
        return self._snip_embs[: len(queries)]

    def bm25_topk_scored_batch(self, query_texts, top_k: int, query_chunk: int = 512):
        return [row[:top_k] for row in self._bm25_per_snippet[: len(query_texts)]]


def _five_snippets() -> List[Snippet]:
    """Five windows: #1 and #3 are peaked on a code, the rest are boilerplate."""
    return [
        Snippet(
            note_id="n1", section="__whole_note__", snippet_index=i,
            start_sentence=i, end_sentence=i + 1,
            # boilerplate is made LONGEST on purpose: a length-ranked selector
            # would keep exactly the windows a relevance-ranked one discards.
            n_words=100 if i in (0, 4) else 10,
            text=f"window {i}",
        )
        for i in range(5)
    ]


def _selecting_pool(mode: str, select_k: int) -> NotePool:
    kb = _tiny_kb()
    codes = ["11111", "22222", "33333", "44444", "A0021"]
    # Code directions spread around the circle, and deliberately NOT sharing an
    # axis with each other.  The module-level layout used elsewhere puts 33333
    # at 45° and gives A0021 the same vector as 22222; both are fatal here:
    #   * a code at 45° means the "boilerplate" snippets sit exactly ON a code
    #     axis, so cosmax separated them from the peaked windows by ~6e-8 of
    #     float32 rounding rather than by anything real;
    #   * two codes sharing the [0,1] axis doubles that row's mean, which is
    #     what cosmargin (max - mean) subtracts -- so window 3 scored LOWEST on
    #     the very selector meant to rank it top.
    # Local to this fixture: the other tests need the shared layout.
    dense_vecs = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
         [_SQRT_HALF, -_SQRT_HALF]]
    )
    # Snippets 1 and 3 sit ON a code axis (11111 and 22222).  0/2/4 sit at 45°
    # between two codes and on none of them: moderately close to several, peaked
    # on nothing -- generic surgical boilerplate.
    snip_embs = np.array(
        [[_SQRT_HALF, _SQRT_HALF], [1.0, 0.0], [_SQRT_HALF, _SQRT_HALF],
         [0.0, 1.0], [_SQRT_HALF, _SQRT_HALF]]
    )
    bm25 = [[(0, 5.0), (1, 2.0)] for _ in range(5)]
    index = _LenientFakeKBIndex(codes, dense_vecs, snip_embs, bm25)
    cfg = AssemblerConfig(
        bm25_top_k=2, dense_top_k=2, snippet_select=mode, max_snippets=select_k
    )
    return cfg.build_assembler(kb, index).assemble_note(
        note_id="n1", snippets=_five_snippets(), select_k=select_k
    )


def test_snippet_select_length_is_a_noop():
    """The historical path must not select inside the assembler."""
    pool = _selecting_pool("length", select_k=2)
    assert len(pool.snippet_texts) == 5, (
        "snippet_select='length' must leave the snippet list untouched — "
        "down-selection belongs upstream in snippets_for_note"
    )
    print("  OK  snippet_select='length' is a no-op inside assemble_note")


def test_snippet_select_keeps_peaked_windows_not_longest():
    """cosmargin must reject boilerplate that a length ranker would keep."""
    for mode in ("cosmax", "cosmargin"):
        pool = _selecting_pool(mode, select_k=2)
        assert len(pool.snippet_texts) == 2, f"{mode}: expected 2 snippets"
        kept = {t.split("] ")[-1] for t in pool.snippet_texts}
        assert kept == {"window 1", "window 3"}, (
            f"{mode}: kept {kept}; expected the two peaked windows. The "
            f"boilerplate windows (0, 2, 4) are LONGER, so this failing means "
            f"selection fell back to length ranking."
        )
    print("  OK  cosmax/cosmargin keep peaked evidence over longer boilerplate")


def test_best_snippet_idx_reindexed_after_selection():
    """Every ``best_snippet_idx`` must index the SELECTED list, not the original.

    This is the one way selection can corrupt the ledger: an index left over
    from the pre-selection list would point at the wrong evidence, or off the
    end of ``snippet_texts`` entirely.
    """
    pool = _selecting_pool("cosmargin", select_k=2)
    n = len(pool.snippet_texts)
    for rec in pool.records:
        assert 0 <= rec.best_snippet_idx < n, (
            f"{rec.code}: best_snippet_idx={rec.best_snippet_idx} out of range "
            f"for {n} selected snippets"
        )
    print("  OK  best_snippet_idx re-indexed onto the selected snippets")


def test_snippet_select_is_persisted_and_validated():
    cfg = AssemblerConfig(snippet_select="cosmargin", snippet_pool_cap=128)
    rt = AssemblerConfig.from_config_dict({"assembler": cfg.to_config_dict()})
    assert rt == cfg, "snippet_select/snippet_pool_cap must survive config.json"
    # a config.json written before the flag existed must still load
    legacy = AssemblerConfig.from_config_dict({"assembler": {"bm25_top_k": 25}})
    assert legacy.snippet_select == "length", "legacy configs must default to length"
    try:
        AssemblerConfig(snippet_select="bogus").build_assembler(_tiny_kb(), None)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown snippet_select must raise")
    print("  OK  snippet_select persists, back-compats, and validates")


def main() -> None:
    print("=" * 60)
    print("candidate_pool — Unit Test Suite")
    print("=" * 60)
    test_pool_membership_is_source_union()
    test_sources_tuple_per_candidate()
    test_phi_features_multi_source_code()
    test_phi_features_bm25_and_dense_code()
    test_phi_features_llm_only_hcpcs_code()
    test_feats_dict_is_version_independent_superset()
    test_empty_snippets_returns_none()
    test_neighbor_self_leakage_guard()
    test_assembler_config_roundtrip_and_defaults()
    test_snippet_select_length_is_a_noop()
    test_snippet_select_keeps_peaked_windows_not_longest()
    test_best_snippet_idx_reindexed_after_selection()
    test_snippet_select_is_persisted_and_validated()
    print("=" * 60)
    print("ALL candidate_pool TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
