# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Unit tests for evidence-ledger emission (``--emit-ledger``).

Exercises :func:`build_ledger_entries` (the pure mapping the predict loop calls
on the emit path) without torch, plus the ledger JSONL round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.pipeline.pool.candidate_pool import (
    CandidateRecord,
    NotePool,
)
from cpt_rec.pipeline.crossencoder.predict_verifier import (
    build_ledger_entries,
)
from cpt_rec.evidence.ledger import read_ledger, write_ledger


def _write_kb(path):
    rows = [
        ("29881", "Knee arthroscopy w/ meniscectomy", "CPT", "29800-29999", "Knee arthroscopy"),
        ("29882", "Knee arthroscopy w/ repair",       "CPT", "29800-29999", "Knee arthroscopy"),
        ("43235", "EGD diagnostic",                   "CPT", "43200-43499", "Esophagoscopy"),
    ]
    pd.DataFrame(
        rows,
        columns=["code", "code_description", "code_system",
                 "code_range_1", "code_range_1_description"],
    ).to_csv(path, index=False)


def _pool():
    return NotePool(
        note_id="n1",
        snippet_texts=["snippet A", "snippet B"],
        records=[
            CandidateRecord(
                code="29881", description="Knee arthroscopy",
                best_snippet_idx=0, best_snippet_cos=0.5,
                sources=("bm25", "dense"),
                feats={"kb_bm25_rank": 0.1, "kb_bm25_score": 0.9,
                       "kb_dense_rank": 0.2, "llm_confidence": 0.7},
            ),
            CandidateRecord(
                code="43235", description="EGD",
                best_snippet_idx=5, best_snippet_cos=0.1,  # out-of-range -> None
                sources=("llm",),
                feats={"llm_rank": 0.3, "llm_confidence": 0.4},
            ),
        ],
    )


def test_build_ledger_entries_reconcile_1to1(tmp_path):
    kb_csv = tmp_path / "kb.csv"
    _write_kb(kb_csv)
    kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)

    pool = _pool()
    probs = np.array([0.8, 0.2])
    keep_idx = [0]  # only the first candidate cleared the threshold

    entries = build_ledger_entries(pool, probs, keep_idx, kb)
    assert len(entries) == len(pool.records)  # 1:1 with the pool / NPZ

    e0, e1 = entries
    assert e0.code == "29881"
    assert e0.candidate_sources == {"bm25": True, "dense": True}
    assert e0.source_ranks == {"kb_bm25": 0.1, "kb_dense": 0.2}
    assert e0.source_confidences["kb_bm25"] == 0.9
    assert e0.source_confidences["llm"] == 0.7
    assert e0.best_snippet == "snippet A"
    assert e0.code_description == "Knee arthroscopy"
    assert e0.code_family == "29800-29999"
    assert e0.verifier_score == 0.8
    assert e0.selected_pre_rule is True

    # second candidate: out-of-range snippet idx -> None; not kept.
    assert e1.best_snippet is None
    assert e1.source_ranks == {"llm": 0.3}
    assert e1.verifier_score == 0.2
    assert e1.selected_pre_rule is False


def test_ledger_jsonl_round_trip(tmp_path):
    kb_csv = tmp_path / "kb.csv"
    _write_kb(kb_csv)
    kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)

    entries = build_ledger_entries(_pool(), np.array([0.8, 0.2]), [0], kb)
    path = tmp_path / "ledger.jsonl"
    assert write_ledger(entries, path) == 2

    back = list(read_ledger(path))
    assert [b.code for b in back] == ["29881", "43235"]
    assert back[0].source_ranks == {"kb_bm25": 0.1, "kb_dense": 0.2}
    assert back[0].selected_pre_rule is True
    assert back[1].best_snippet is None
