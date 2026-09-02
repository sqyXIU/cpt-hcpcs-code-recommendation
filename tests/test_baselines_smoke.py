#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Smoke tests for the B-series baselines.

These tests stand up the full CLI surface against tiny synthetic fixtures
(no GPU, no LLM API).  Heavy components — bi-encoder, cross-encoder,
Azure OpenAI — are stubbed via monkey-patching or replaced with the
``EchoBackend``.

Run from project root:
    PYTHONPATH=src python3 tests/test_baselines_smoke.py
    PYTHONPATH=src pytest tests/test_baselines_smoke.py -q
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PYTHON = sys.executable


def _shared_env() -> dict:
    return {**os.environ, "PYTHONPATH": str(Path("src").resolve())}


def _read_predictions(path: Path) -> pd.DataFrame:
    """
    Read a baseline predictions CSV with all string columns.

    Predictions CSVs have two columns where pandas type inference can
    bite us: ``pred_codes`` and ``pred_scores``.  When every non-empty
    value in a column happens to look numeric (e.g. all codes are
    5-digit CPTs and one row is empty), pandas infers ``float64`` and
    the codes come back as ``"43235.0"`` instead of ``"43235"``.  The
    canonical fix used everywhere else in this codebase is to force
    ``dtype=str`` and ``fillna("")``.
    """
    return pd.read_csv(path, dtype=str).fillna("")


def _make_kb_csv(tmp: Path) -> Path:
    df = pd.DataFrame([
        {"code": "43235", "code_description": "EGD diagnostic",
         "code_lay_term": "EGD", "code_system": "CPT"},
        {"code": "45378", "code_description": "Colonoscopy diagnostic",
         "code_lay_term": "Colonoscopy", "code_system": "CPT"},
        {"code": "52332", "code_description": "Cystoscopy w/ stent",
         "code_lay_term": "Stent", "code_system": "CPT"},
        {"code": "0141A", "code_description": "COVID-19 vax admin",
         "code_lay_term": "COVID admin", "code_system": "CPT"},
        {"code": "J0585", "code_description": "Botulinum toxin A",
         "code_lay_term": "Botox", "code_system": "HCPCS"},
    ])
    for lvl in range(1, 7):
        df[f"code_range_{lvl}"] = ""
        df[f"code_range_{lvl}_description"] = ""
    p = tmp / "kb.csv"
    df.to_csv(p, index=False)
    return p


def _make_train_csv(tmp: Path) -> Path:
    df = pd.DataFrame([
        {"NOTE_ID": "T1", "PROCEDURE_DATE": "2024-03-01",
         "NOTE_TEXT": "EGD performed with biopsy of stomach.",
         "proc_codes": "43235"},
        {"NOTE_ID": "T2", "PROCEDURE_DATE": "2024-04-15",
         "NOTE_TEXT": "Routine screening colonoscopy completed.",
         "proc_codes": "45378"},
        {"NOTE_ID": "T3", "PROCEDURE_DATE": "2024-05-10",
         "NOTE_TEXT": "Cystoscopy with placement of ureteral stent.",
         "proc_codes": "52332"},
        {"NOTE_ID": "T4", "PROCEDURE_DATE": "2024-06-01",
         "NOTE_TEXT": "Patient received Botox injection per protocol.",
         "proc_codes": "J0585"},
        # Polluted gold cell — shape filter must drop ``MA``.
        {"NOTE_ID": "T5", "PROCEDURE_DATE": "2024-06-15",
         "NOTE_TEXT": "Repeat EGD for surveillance biopsy.",
         "proc_codes": "43235|MA"},
    ])
    p = tmp / "train.csv"
    df.to_csv(p, index=False)
    return p


def _make_test_csv(tmp: Path) -> Path:
    df = pd.DataFrame([
        {"NOTE_ID": "Q1", "PROCEDURE_DATE": "2025-01-01",
         "NOTE_TEXT": "EGD with biopsy was performed for the patient.",
         "proc_codes": "43235"},
        {"NOTE_ID": "Q2", "PROCEDURE_DATE": "2025-02-01",
         "NOTE_TEXT": "Patient received Botox injection.",
         "proc_codes": "J0585"},
    ])
    p = tmp / "test.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# M1: BM25 KNN-vote
# ---------------------------------------------------------------------------
def test_m1_build_predict_tune_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train_csv = _make_train_csv(tmp)
        test_csv = _make_test_csv(tmp)
        idx = tmp / "m1.pkl"
        out = tmp / "m1_test.csv"
        thr = tmp / "m1_thr.json"

        # 1) build
        cmd = [PYTHON, "-m", "cpt_rec.baselines.m1_bm25_knn",
               "build-index", "--train", str(train_csv),
               "--index-out", str(idx)]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             env=_shared_env(), check=False)
        assert res.returncode == 0, res.stderr

        # 2) tune-threshold (val == test for the smoke test)
        cmd = [PYTHON, "-m", "cpt_rec.baselines.m1_bm25_knn",
               "tune-threshold", "--notes", str(test_csv),
               "--index", str(idx), "--out-json", str(thr),
               "--top-k", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             env=_shared_env(), check=False)
        assert res.returncode == 0, res.stderr
        with open(thr) as f:
            payload = json.load(f)
        assert "best_threshold" in payload
        assert payload["best_micro_f1"] >= 0.0

        # 3) predict using sidecar threshold
        cmd = [PYTHON, "-m", "cpt_rec.baselines.m1_bm25_knn",
               "predict", "--notes", str(test_csv),
               "--index", str(idx), "--out", str(out),
               "--threshold-json", str(thr), "--top-k", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             env=_shared_env(), check=False)
        assert res.returncode == 0, res.stderr

        df = _read_predictions(out)
        assert list(df["note_id"]) == ["Q1", "Q2"]
        # Q1 must contain a non-empty pred_codes (fallback floor guarantees it).
        assert df.loc[0, "pred_codes"] != ""
        # MA pollution must NOT have entered the index — `MA` should never
        # appear as a predicted code.
        for cell in df["pred_codes"]:
            for tok in cell.split("|"):
                assert tok != "MA", f"MA leaked into predictions: {cell!r}"
    print("PASS: test_m1_build_predict_tune_roundtrip")


# ---------------------------------------------------------------------------
# kb_index: hybrid retrieve-and-rerank — stub the bi-encoder + cross-encoder so we
# don't have to download any HF model on a smoke run.
# ---------------------------------------------------------------------------
class _FakeBi:
    """Deterministic fake bi-encoder returning a tiny dense matrix."""
    def __init__(self, *args, **kwargs):
        self.dim = 16

    def encode(self, texts, batch_size=64, show_progress=False,
               max_length_override=None):
        import hashlib
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.md5(str(t).encode("utf-8")).digest()
            vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            vec = vec[: self.dim] / 255.0
            norm = np.linalg.norm(vec)
            out[i] = vec / norm if norm > 0 else vec
        return out


class _FakeCross:
    def __init__(self, *args, **kwargs):
        pass

    def score(self, pairs, batch_size=32):
        # Score 0.99 if any token in text_a appears in text_b, else 0.10.
        scores = np.full(len(pairs), 0.10, dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            tokens_a = {t for t in a.lower().split() if t.isalpha()}
            tokens_b = {t for t in b.lower().split() if t.isalpha()}
            if tokens_a & tokens_b:
                scores[i] = 0.99
        return scores


@contextlib.contextmanager
def _stubbed_kb_index():
    """Yield the kb_index module with its two HF encoders swapped for fakes."""
    from cpt_rec.baselines import kb_index as KBI

    orig_bi, orig_cx = KBI.BiEncoder, KBI.CrossEncoder
    KBI.BiEncoder = _FakeBi
    KBI.CrossEncoder = _FakeCross
    try:
        yield KBI
    finally:
        KBI.BiEncoder = orig_bi
        KBI.CrossEncoder = orig_cx


def test_kb_index_build_predict_with_stubs():
    with _stubbed_kb_index() as KBI, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kb_csv = _make_kb_csv(tmp)
        test_csv = _make_test_csv(tmp)
        idx_dir = tmp / "kb_index"
        out = tmp / "kb_test.csv"

        KBI.build_index(kb_csv=kb_csv, out_dir=idx_dir)
        with open(idx_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest.get("format_version", 1) == KBI.INDEX_FORMAT_VERSION

        KBI.predict_b3(
            notes_csv=test_csv,
            kb_csv=None,        # v2 index has descriptions baked in
            index_dir=idx_dir,
            out_csv=out,
            bm25_top_k=5, dense_top_k=5,
            rerank_threshold=0.5,
            bi_query_max_length=32,
        )
        df = _read_predictions(out)
        assert list(df["note_id"]) == ["Q1", "Q2"]
        # The rerank stub flags any code-description sharing a word
        # with the query, so Q1 ("EGD with biopsy") should at least
        # surface 43235 ("EGD diagnostic").
        q1_codes = df.loc[0, "pred_codes"].split("|")
        assert "43235" in q1_codes, q1_codes
    print("PASS: test_kb_index_build_predict_with_stubs")


def test_kb_index_chunks_let_the_cross_encoder_see_past_its_window():
    """The reranker judges a whole note through a ~350-token window.

    `--rerank-max-length 384` is the budget for the PACKED PAIR, and HF
    truncates `longest_first`, so a 10–30 token code description survives
    whole and the note absorbs every truncation step. It cannot be raised
    to fit the notes: every BERT-family reranker caps at 512 learned
    position embeddings, and operative notes run 3–5x that. `--rerank-chunks
    N` is the only way past it — score each piece, keep each candidate's
    best. N=1 must stay byte-identical.

    The fake below scores on the first 5 words of text_a only, which is
    exactly what the real window does. Evidence for 52332 sits at word ~20,
    outside that window.
    """
    from cpt_rec.baselines import kb_index as KBI

    WINDOW = 5

    class _WindowedCross:
        """Word-overlap scorer that can only see the first WINDOW words."""
        def __init__(self, *a, **k):
            pass

        def score(self, pairs, batch_size=32):
            out = np.full(len(pairs), 0.10, dtype=np.float32)
            for i, (a, b) in enumerate(pairs):
                seen = {t for t in a.lower().split()[:WINDOW] if t.isalpha()}
                if seen & {t for t in b.lower().split() if t.isalpha()}:
                    out[i] = 0.99
            return out

    orig_bi, orig_cx = KBI.BiEncoder, KBI.CrossEncoder
    KBI.BiEncoder, KBI.CrossEncoder = _FakeBi, _WindowedCross
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            kb_csv = _make_kb_csv(tmp)
            # "cystoscopy" lands at word ~20 — past the window, inside chunk 4.
            note = ("Preoperative note header boilerplate follows here. "
                    "Patient identification and facility information block. "
                    "The operative detail begins now: cystoscopy with stent "
                    "placement was performed without complication.")
            pd.DataFrame([{
                "NOTE_ID": "L1", "PROCEDURE_DATE": "2025-01-01",
                "NOTE_TEXT": note, "proc_codes": "52332",
            }]).to_csv(tmp / "long.csv", index=False)

            idx = tmp / "idx"
            KBI.build_index(kb_csv=kb_csv, out_dir=idx)

            def _run(n_chunks, name):
                out = tmp / f"{name}.csv"
                KBI.predict_b3(
                    notes_csv=tmp / "long.csv", kb_csv=None, index_dir=idx,
                    out_csv=out, bm25_top_k=5, dense_top_k=5,
                    rerank_threshold=0.5, bi_query_max_length=32,
                    rerank_chunks=n_chunks,
                )
                return _read_predictions(out).loc[0, "pred_codes"].split("|")

            single = _run(1, "chunks1")
            chunked = _run(4, "chunks4")

            # N=1: the window never reaches the operative detail, nothing
            # clears the threshold, and the min-keep floor decides the answer.
            assert "52332" not in single[:1], (
                f"52332 should not lead on a head-only window: {single}")
            # N=4: the piece carrying the evidence scores it, and max-over-
            # chunks promotes it to the top.
            assert chunked[0] == "52332", (
                f"chunked rerank should surface 52332 first: {chunked}")
    finally:
        KBI.BiEncoder, KBI.CrossEncoder = orig_bi, orig_cx
    print("PASS: test_kb_index_chunks_let_the_cross_encoder_see_past_its_window")


def _make_history_csvs(tmp: Path) -> tuple[Path, Path]:
    """A two-event history for 45378: born in 2000, deleted 15-Jan-2025.

    The fixture notes straddle that deletion — Q1 is dated 2025-01-01 and
    Q2 2025-02-01 — so a correct per-note restriction keeps 45378 in Q1's
    candidate pool and drops it from Q2's.  A restriction that reads a
    constant date, the wrong row, or no row at all fails one half or the
    other.  The other four KB codes have no events and stay active
    throughout, which is what makes the single drop attributable.
    """
    changes = pd.DataFrame([{
        "code_system": "CPT", "year": "2000", "category": "",
        "change_type": "new", "code": "45378", "description": "",
        "advice": "", "crosswalk_code": "", "new_descriptor": "",
    }])
    deleted = pd.DataFrame([{
        "code": "45378", "description": "Colonoscopy diagnostic",
        "deleted_date": "15-Jan-25", "substitute_code": "",
        "archived_info": "", "code_system": "CPT",
    }])
    c_path = tmp / "code_changes.csv"
    d_path = tmp / "deleted_codes.csv"
    changes.to_csv(c_path, index=False)
    deleted.to_csv(d_path, index=False)
    return c_path, d_path


def _pool_from_npz(path: Path) -> dict:
    """note_id -> candidate codes, read back from a --dump-scores-npz file.

    The NPZ holds the *post-restriction* union, before --rerank-threshold,
    so it is the direct observable for what history did or didn't remove.
    """
    dat = np.load(path, allow_pickle=True)
    return {
        str(n): [str(c) for c in codes]
        for n, codes in zip(dat["note_ids"], dat["candidate_codes"])
    }


def test_kb_index_history_restricts_candidate_pool():
    """--history-* must drop per-note-inactive codes from the KB index pool.

    Regression test for ``NameError: name 'row' is not defined``, which sat
    behind ``if history is not None`` from a696c08 (2026-04-25) until
    The index advertised the history flags and crashed on the first
    note the moment they were passed, so no result from it had ever been
    produced with the date restriction M1/M3/M4 apply.  The A/B below is
    the assertion that was missing — the same run with and without the
    flags, compared on the pool rather than on the thresholded output.
    """
    with _stubbed_kb_index() as KBI, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kb_csv = _make_kb_csv(tmp)
        test_csv = _make_test_csv(tmp)
        changes_csv, deleted_csv = _make_history_csvs(tmp)
        idx_dir = tmp / "kb_index"
        KBI.build_index(kb_csv=kb_csv, out_dir=idx_dir)

        # Identical in every respect except the two history flags.
        shared = dict(
            notes_csv=test_csv,
            kb_csv=None,
            index_dir=idx_dir,
            bm25_top_k=5, dense_top_k=5,
            rerank_threshold=0.5,
            bi_query_max_length=32,
        )
        off_npz = tmp / "pool_off.npz"
        on_npz = tmp / "pool_on.npz"
        KBI.predict_b3(out_csv=tmp / "off.csv",
                      dump_scores_npz=off_npz, **shared)
        KBI.predict_b3(out_csv=tmp / "on.csv", dump_scores_npz=on_npz,
                      history_changes=changes_csv,
                      history_deleted=deleted_csv, **shared)

        off = _pool_from_npz(off_npz)
        on = _pool_from_npz(on_npz)
        # Control: with history off, 45378 is a candidate for both notes,
        # so the drop below can only be the restriction.
        assert "45378" in off["Q1"], off["Q1"]
        assert "45378" in off["Q2"], off["Q2"]
        # Q1 (2025-01-01) predates the deletion; Q2 (2025-02-01) follows it.
        assert "45378" in on["Q1"], on["Q1"]
        assert "45378" not in on["Q2"], on["Q2"]
        # Exactly one code moved: history restricts, it doesn't reshuffle.
        assert set(on["Q1"]) == set(off["Q1"]), (on["Q1"], off["Q1"])
        assert set(on["Q2"]) == set(off["Q2"]) - {"45378"}, (on["Q2"], off["Q2"])
    print("PASS: test_kb_index_history_restricts_candidate_pool")


# ---------------------------------------------------------------------------
# M3: GPT zero-shot via EchoBackend
# ---------------------------------------------------------------------------
def test_m3_with_echo_backend():
    from cpt_rec.baselines.m3_zeroshot_llm import predict_b4
    from cpt_rec.baselines.llm import EchoBackend
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kb_csv = _make_kb_csv(tmp)
        test_csv = _make_test_csv(tmp)
        out = tmp / "m3.csv"

        kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
        # EchoBackend echoes any "- CODE: ..." lines from the prompt; M3
        # builds no candidate block, so the echo returns no codes.
        predict_b4(
            notes_csv=test_csv,
            kb=kb,
            out_csv=out,
            backend=EchoBackend(),
            max_workers=2,
        )
        df = _read_predictions(out)
        assert list(df["note_id"]) == ["Q1", "Q2"]
        # No candidate block → echo returns []; cells should be empty
        # but the harness contract still holds.
        for cell in df["pred_codes"]:
            assert cell == "", f"unexpected echo prediction: {cell!r}"
    print("PASS: test_m3_with_echo_backend")


# ---------------------------------------------------------------------------
# M4: GPT + RAG via EchoBackend.  Exemplar block has "- CODE: ..." lines so
# Echo returns the first k of them; predictions are then constrained to the
# candidate union.
# ---------------------------------------------------------------------------
def test_m4_with_echo_backend():
    from cpt_rec.baselines.m1_bm25_knn import build_index as m1_build
    from cpt_rec.baselines.m4_exemplar_rag import predict_b5
    from cpt_rec.baselines.bm25_index import TrainNoteBM25Index
    from cpt_rec.baselines.llm import EchoBackend
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train_csv = _make_train_csv(tmp)
        test_csv = _make_test_csv(tmp)
        kb_csv = _make_kb_csv(tmp)
        idx_path = tmp / "m1.pkl"
        out = tmp / "m4.csv"

        m1_build(train_csv=train_csv, index_out=idx_path)
        index = TrainNoteBM25Index.load(idx_path)
        kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)

        predict_b5(
            notes_csv=test_csv,
            index=index,
            kb=kb,
            out_csv=out,
            backend=EchoBackend(k=10),
            top_k=3,
            max_workers=2,
        )
        df = _read_predictions(out)
        assert list(df["note_id"]) == ["Q1", "Q2"]
        # Predictions must be a subset of the candidate-union, which is a
        # subset of the KB.  No `MA` (filtered by shape filter at index
        # build time) and no out-of-KB code should ever appear.
        kb_codes = set(kb.codes)
        for cell in df["pred_codes"]:
            for tok in cell.split("|"):
                if tok:
                    assert tok in kb_codes, f"out-of-KB token: {tok!r}"
                    assert tok != "MA"
    print("PASS: test_m4_with_echo_backend")


# ---------------------------------------------------------------------------
# Common helpers (apply_seed_and_limit, log_prediction_stats)
# ---------------------------------------------------------------------------
def test_apply_seed_and_limit_is_deterministic():
    from cpt_rec.baselines.common import apply_seed_and_limit

    df = pd.DataFrame([
        {"note_id": "B", "note_text": "b"},
        {"note_id": "A", "note_text": "a"},
        {"note_id": "C", "note_text": "c"},
    ])
    out = apply_seed_and_limit(df, seed=42, limit=2)
    # Sorted -> A, B, C; limit 2 -> A, B.
    assert list(out["note_id"]) == ["A", "B"], list(out["note_id"])
    print("PASS: test_apply_seed_and_limit_is_deterministic")


def test_log_prediction_stats_returns_expected_dict():
    from cpt_rec.baselines.common import (
        Prediction, log_prediction_stats,
    )

    preds = [
        Prediction(note_id="A", codes=["43235", "45378"]),
        Prediction(note_id="B", codes=[]),
        Prediction(note_id="C", codes=["52332"]),
    ]
    stats = log_prediction_stats(preds, label="smoke")
    assert stats["n_notes"] == 3.0
    assert stats["total_codes"] == 3.0
    assert stats["n_empty_predictions"] == 1.0
    print("PASS: test_log_prediction_stats_returns_expected_dict")


def test_budget_fill_audit_is_opt_in_and_flags_a_short_run():
    """--shortlist-k runs must report achieved cardinality, not just ask for k.

    Regression guard for the 2026-08-25 finding: every
    single-sample generative baseline answered ``--shortlist-k 10`` with ~4
    codes, and nothing in the pipeline noticed, because R@8/R@10/R@20 all
    read as the same saturated number.
    """
    from cpt_rec.baselines.common import (
        Prediction, log_prediction_stats,
    )

    # Omitting budget_k must leave the returned dict exactly as it was.
    preds = [Prediction(note_id=str(i), codes=["43235"]) for i in range(4)]
    base = log_prediction_stats(preds, label="smoke")
    assert not any(k.startswith("budget") for k in base), base
    assert "n_notes_at_budget" not in base

    # A short run is measured, not merely warned about.
    short = log_prediction_stats(preds, label="smoke", budget_k=10)
    assert short["budget_k"] == 10.0
    assert short["budget_fill_rate"] == 0.1
    assert short["n_notes_at_budget"] == 0.0
    assert short["pct_notes_at_budget"] == 0.0

    # A run that fills the budget reports fill 1.0 and every note at k.
    full = log_prediction_stats(
        [Prediction(note_id=str(i), codes=[str(90000 + j) for j in range(10)])
         for i in range(4)],
        label="smoke", budget_k=10,
    )
    assert full["budget_fill_rate"] == 1.0
    assert full["pct_notes_at_budget"] == 100.0

    # Notes above k still count as reaching it (trimming happens upstream).
    over = log_prediction_stats(
        [Prediction(note_id="A", codes=[str(90000 + j) for j in range(12)])],
        label="smoke", budget_k=10,
    )
    assert over["n_notes_at_budget"] == 1.0
    print("PASS: test_budget_fill_audit_is_opt_in_and_flags_a_short_run")



# ---------------------------------------------------------------------------
# Budget-matched shortlists: self-consistency, padding, truncation recovery
# ---------------------------------------------------------------------------
def test_self_consistency_scores_are_agreement_frequencies():
    from cpt_rec.baselines.common import rank_by_self_consistency

    codes, scores = rank_by_self_consistency([
        ["44950", "49505"],
        ["44950", "49505"],
        ["44950", "49505"],
        ["44950", "47562"],
        ["44950"],
    ])
    assert codes[0] == "44950"
    assert abs(scores[0] - 1.0) < 1e-9      # 5/5
    assert abs(scores[1] - 0.6) < 1e-9      # 3/5
    assert abs(scores[2] - 0.2) < 1e-9      # 1/5
    assert scores == sorted(scores, reverse=True)
    # equal frequency -> the code named earlier on average wins
    assert rank_by_self_consistency([["B", "A"], ["B", "A"]])[0] == ["B", "A"]
    assert rank_by_self_consistency([]) == ([], [])
    print("PASS: test_self_consistency_scores_are_agreement_frequencies")


def test_complete_shortlist_never_pads_implicitly():
    from cpt_rec.baselines.common import complete_shortlist

    # No pool -> the shortfall is REPORTED, not filled.
    codes, scores = complete_shortlist(["X"], [1.0], 5)
    assert codes == ["X"] and len(scores) == 1

    # With a pool -> exactly k, generated code first, pad strictly below it
    # and internally ordered by pool position.
    codes, scores = complete_shortlist(
        ["X"], [0.4], 5, pad_pool=["X", "P1", "P2", "P3", "P4", "P5"]
    )
    assert codes == ["X", "P1", "P2", "P3", "P4"]
    assert all(x < 0.4 for x in scores[1:])
    assert scores[1] > scores[2] > scores[3] > scores[4]

    # Over-long input is still truncated to k.
    codes, _ = complete_shortlist(list("ABCDEFG"), [1.0] * 7, 3)
    assert codes == ["A", "B", "C"]
    print("PASS: test_complete_shortlist_never_pads_implicitly")


def test_m3_self_consistency_fills_the_budget():
    """N=1 must stay byte-identical; N>1 must fill k and carry real scores."""
    import random
    from cpt_rec.baselines.m3_zeroshot_llm import _score_one_note
    from cpt_rec.baselines.llm import LLMBackend

    filler = [f"4{n:04d}" for n in range(200)]
    kb = set(filler) | {"44950", "49505"}

    class _Varying(LLMBackend):
        """Two codes it always names, plus a tail that varies by sample."""
        def __init__(self):
            self.calls = 0

        def chat(self, system, user):
            self.calls += 1
            random.seed(self.calls)
            codes = ["44950", "49505"] + random.sample(
                filler, random.randint(0, 4)
            )
            return '{"selected": [%s]}' % ",".join(f'"{c}"' for c in codes)

    b = _Varying()
    single = _score_one_note("n1", "note", b, kb, 2048, shortlist_k=10,
                             self_consistency=1)
    assert b.calls == 1
    assert single.scores is None, "N=1 must not start emitting scores"
    assert len(single.codes) < 10, "a single call cannot fill k=10"

    b = _Varying()
    multi = _score_one_note("n2", "note", b, kb, 2048, shortlist_k=10,
                            self_consistency=10)
    assert b.calls == 10
    assert len(multi.codes) == 10, "N=10 must fill the budget"
    assert multi.scores == sorted(multi.scores, reverse=True)
    assert all(0.0 < x <= 1.0 for x in multi.scores)
    # the codes every sample agreed on lead, at probability 1.0
    assert set(multi.codes[:2]) == {"44950", "49505"}
    assert abs(multi.scores[0] - 1.0) < 1e-9
    print("PASS: test_m3_self_consistency_fills_the_budget")


def test_m4_candidate_filter_has_three_levels_and_defaults_to_candidates():
    """M4's response filter is far tighter than M3's; --candidate-filter
    measures that asymmetry instead of assuming it.

    M3 filters the response against the whole KB (~10^4 codes); M4 filters
    against the retrieved candidate block (~10^1). So part of M4's
    precision advantage over M3 could be filter strength rather than
    better reading. Level 'kb' makes the two filters identical, which
    turns that into a measurable single-variable comparison; 'none'
    removes the filter entirely. Default stays 'candidates' so every
    published M4 run reproduces byte-for-byte.
    """
    from cpt_rec.baselines.m4_exemplar_rag import (
        CANDIDATE_FILTER_LEVELS, _apply_response_filter,
        log_response_filter_tally, reset_response_filter_tally,
    )

    assert CANDIDATE_FILTER_LEVELS == ("candidates", "kb", "none")
    cands = {"44950"}
    kb = {"44950", "49505"}
    # 49505 is real but retrieval missed it; 99999 is a hallucination.
    named = ["44950", "49505", "99999"]

    reset_response_filter_tally()
    assert _apply_response_filter(named, cands, kb, "candidates") == ["44950"]
    reset_response_filter_tally()
    assert _apply_response_filter(named, cands, kb, "kb") == ["44950", "49505"]
    reset_response_filter_tally()
    kept = _apply_response_filter(named, cands, kb, "none")
    assert kept == named and kept is not named

    # every level is tallied whichever one is active
    reset_response_filter_tally()
    _apply_response_filter(named, cands, kb, "candidates")
    st = log_response_filter_tally(level="candidates")
    assert st["candidate_filter_level"] == "candidates"
    assert st["codes_named_by_model"] == 3.0
    assert st["codes_outside_candidates"] == 2.0     # 49505 + 99999
    assert st["codes_outside_kb"] == 1.0             # 99999 only
    assert st["note_samples_with_a_drop"] == 1.0

    print("PASS: test_m4_candidate_filter_has_three_levels_and_defaults_to_candidates")


def test_kb_filter_is_default_on_and_can_be_switched_off():
    """M3's response filter is a scored aid; --no-kb-filter measures its cost.

    M3 silently drops any code the model names that is absent from the KB
    vocabulary. That is an out-of-vocabulary safety net no unaided
    zero-shot deployment has: it converts a hallucination into a
    non-event rather than a false positive. The default stays ON (every
    result to date was produced with it on and must stay reproducible),
    but the tally is kept either way, so the size of the aid is reported
    even when the aid is in force.
    """
    from cpt_rec.baselines.m3_zeroshot_llm import (
        _apply_kb_filter, log_kb_filter_tally, reset_kb_filter_tally,
    )

    kb = {"44950", "49505"}
    named = ["44950", "99999", "49505", "00000"]

    reset_kb_filter_tally()
    on = _apply_kb_filter(named, kb, enabled=True)
    assert on == ["44950", "49505"], "filter ON must drop out-of-KB codes"
    stats = log_kb_filter_tally(enabled=True)
    assert stats["kb_filter_enabled"] == 1.0
    assert stats["codes_named_by_model"] == 4.0
    assert stats["codes_not_in_kb"] == 2.0
    assert stats["pct_codes_not_in_kb"] == 50.0
    assert stats["note_samples_with_a_drop"] == 1.0

    reset_kb_filter_tally()
    off = _apply_kb_filter(named, kb, enabled=False)
    assert off == named, "filter OFF must keep the model's own vocabulary"
    assert off is not named, "must not alias the caller's list"
    stats = log_kb_filter_tally(enabled=False)
    assert stats["kb_filter_enabled"] == 0.0
    # counted even though nothing was removed -- that is the whole point
    assert stats["codes_not_in_kb"] == 2.0

    reset_kb_filter_tally()
    assert log_kb_filter_tally()["codes_named_by_model"] == 0.0
    clean = _apply_kb_filter(["44950"], kb, enabled=True)
    assert clean == ["44950"]
    assert log_kb_filter_tally()["note_samples_with_a_drop"] == 0.0
    print("PASS: test_kb_filter_is_default_on_and_can_be_switched_off")


def test_empty_completion_at_token_cap_is_retried_not_swallowed():
    """The M3 --shortlist-k 10 defect: reasoning spends the whole cap, the
    SDK raises nothing, and the note silently scores zero codes."""
    import types
    from cpt_rec.baselines.llm import AzureOpenAIBackend

    def _resp(content, finish_reason):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )])

    class _Completions:
        def __init__(self, threshold):
            self.caps = []
            self.threshold = threshold

        def create(self, **kw):
            cap = kw.get("max_completion_tokens") or kw.get("max_tokens")
            self.caps.append(cap)
            if cap < self.threshold:
                return _resp("", "length")
            return _resp('{"selected": ["44950"]}', "stop")

    def _backend(cap, retries, threshold):
        o = AzureOpenAIBackend.__new__(AzureOpenAIBackend)
        o.rate_limiter = None
        o.max_retries = 4
        o.max_tokens = cap
        o.deployment_name = "gpt-5.3-sol"
        o.temperature = 0.0
        # The self-correcting request-shape flags that __init__ seeds.  This
        # fixture bypasses __init__ via __new__, so every flag added to
        # AzureOpenAIBackend has to be mirrored here or _build_kwargs raises
        # AttributeError -- which is exactly how _send_response_format broke
        # this test when it landed.
        o._use_completion_tokens = True
        o._send_temperature = False
        o._send_response_format = True
        o.truncation_retries = retries
        o.n_truncation_retries = 0
        o.n_truncation_lost = 0
        comp = _Completions(threshold)
        o.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=comp)
        )
        return o, comp

    # recovers by escalating the cap
    o, comp = _backend(512, 2, 2000)
    assert o.chat("s", "u").strip(), "must recover a real completion"
    assert comp.caps == [512, 2048]
    assert o.n_truncation_retries == 1 and o.n_truncation_lost == 0

    # a genuinely exhausted budget is counted, never silent
    o, comp = _backend(8, 1, 2000)
    assert o.chat("s", "u") == ""
    assert o.n_truncation_lost == 1

    # a response that never truncated makes exactly one call
    o, comp = _backend(4096, 2, 2000)
    assert o.chat("s", "u").strip()
    assert comp.caps == [4096] and o.n_truncation_retries == 0
    print("PASS: test_empty_completion_at_token_cap_is_retried_not_swallowed")


def test_server_side_n_is_used_when_supported_and_never_faked():
    """Self-consistency needs n DISTINCT samples.  An endpoint that ignores
    `n` returns one choice, which would make every agreement score 1.0 -- a
    vacuous ranking that looks perfectly healthy in the output."""
    import types
    from cpt_rec.baselines.llm import LocalOpenAIBackend

    def _resp(texts, finish_reason="stop"):
        return types.SimpleNamespace(choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=t),
                finish_reason=finish_reason,
            ) for t in texts
        ])

    def _backend(handler):
        o = LocalOpenAIBackend.__new__(LocalOpenAIBackend)
        o.rate_limiter = None
        o.extra_body = {}
        o.max_retries = 3
        o.max_tokens = 512
        o.model = "m5-sft"
        o.temperature = 0.7
        o._use_json_mode = False
        o._use_completion_tokens = False
        o._send_temperature = True
        o._use_server_side_n = True
        o.truncation_retries = 2
        o.n_truncation_retries = 0
        o.n_truncation_lost = 0
        calls = []

        class _Completions:
            def create(self, **kw):
                calls.append(kw)
                return handler(kw, len(calls))

        o.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())
        )
        return o, calls

    # honoured: ONE request carries the whole batch, at the sampling temperature
    o, calls = _backend(lambda kw, i: _resp(["s%d" % j for j in range(kw["n"])]))
    assert o.chat_n("s", "u", 5) == ["s0", "s1", "s2", "s3", "s4"]
    assert len(calls) == 1 and calls[0]["n"] == 5
    assert calls[0]["temperature"] == 0.7, "sampling temperature must reach the server"

    # silently ignored: detected, filled sequentially, and never re-attempted
    o, calls = _backend(lambda kw, i: _resp(["only%d" % i]))
    assert len(o.chat_n("s", "u", 4)) == 4
    assert o._use_server_side_n is False
    o.chat_n("s", "u", 4)
    assert all("n" not in kw for kw in calls[1:]), "fallback must be sticky"

    # rejected outright: same fallback, via the exception path
    def _rejecting(kw, i):
        if "n" in kw:
            raise ValueError("unsupported parameter: 'n'")
        return _resp(["seq%d" % i])

    o, calls = _backend(_rejecting)
    assert len(o.chat_n("s", "u", 3)) == 3 and o._use_server_side_n is False

    # n=1 is the historical single-call path and must not send `n` at all
    o, calls = _backend(lambda kw, i: _resp(["one"]))
    assert o.chat_n("s", "u", 1) == ["one"] and "n" not in calls[0]

    # an all-empty batch defers to the sequential path, which owns the
    # token-cap escalation ladder
    def _empties(kw, i):
        if "n" in kw:
            return _resp([""] * kw["n"], "length")
        return _resp(["recovered"])

    o, calls = _backend(_empties)
    assert o.chat_n("s", "u", 3) == ["recovered"] * 3
    print("PASS: test_server_side_n_is_used_when_supported_and_never_faked")



def test_sectionized_csv_still_carries_the_whole_raw_note():
    """`*_eval_sectioned.csv` is a SUPERSET of the raw export, not a
    replacement for it — so pointing a baseline's --notes at that file does
    NOT make it read sections.

    `split_op_notes.write_wide_output` does `out_base_df = df.copy()` and
    APPENDS the section columns, so NOTE_TEXT survives verbatim.  M4 had no
    --sectionized-csv and `_TEXT_CANDIDATES` resolves NOTE_TEXT first, so M4
    read the whole raw note while M3 and M5 read three sections.  This test
    pins that fact, and pins that M4's --sectionized-csv is what actually
    equalises the input across systems.
    """
    from cpt_rec.common.sectionizer.split_op_notes import (
        write_wide_output)
    from cpt_rec.baselines.common import load_notes_for_prediction
    from cpt_rec.baselines.m4_exemplar_rag import (
        _sectionized_text_lookup)

    note = (
        "PATIENT IDENTIFICATION: subject A.\n"
        "PRE-OPERATIVE DIAGNOSIS: condition B.\n"
        "PROCEDURE(S) PERFORMED: procedure X.\n"
        "DETAILED DESCRIPTION: narrative C.\n"
        "FINDINGS: finding D.\n"
        "DISPOSITION: disposition E.\n"
    )
    # The pattern config used for the paper's runs was induced from that corpus
    # and is not distributed; `cptrec-extract-headers` induces one from any
    # corpus.  A minimal config covering this note's headers is enough here.
    cfg = {"version": 1, "patterns": [
        {"section": "Patient Identification", "regex": r"\bPATIENT\s+IDENTIFICATION\s*:\s*"},
        {"section": "Pre-operative Diagnosis", "regex": r"\bPRE-OPERATIVE\s+DIAGNOSIS\s*:\s*"},
        {"section": "Procedure(s) Performed", "regex": r"\bPROCEDURE\(S\)\s+PERFORMED\s*:\s*"},
        {"section": "Detailed Description", "regex": r"\bDETAILED\s+DESCRIPTION\s*:\s*"},
        {"section": "Findings", "regex": r"\bFINDINGS\s*:\s*"},
        {"section": "Disposition", "regex": r"\bDISPOSITION\s*:\s*"},
    ]}
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "eval.csv"
        sec = Path(d) / "eval_sectioned.csv"
        cfg_path = Path(d) / "patterns.json"
        cfg_path.write_text(json.dumps(cfg))
        pd.DataFrame(
            {"NOTE_ID": ["n1"], "NOTE_TEXT": [note], "CPT_CODES": ["44950"]}
        ).to_csv(raw, index=False)
        write_wide_output(str(raw), str(sec), "NOTE_TEXT", str(cfg_path))

        # the sectioned file still holds the untouched note ...
        wide = pd.read_csv(sec, dtype=str)
        assert wide["NOTE_TEXT"].iloc[0] == note

        # ... and the shared loader hands it straight back
        loaded = load_notes_for_prediction(sec)
        assert loaded["note_text"].iloc[0] == note, (
            "reading *_eval_sectioned.csv must yield the RAW note; if this "
            "ever changes, every archived M4 number changes with it"
        )

        # the new flag is what swaps in the three evidence sections
        ev = _sectionized_text_lookup(sec, "test")["n1"]
        assert "procedure X" in ev and "narrative C" in ev and "finding D" in ev
        assert "subject A" not in ev and "condition B" not in ev
        assert len(ev) < len(note)

    print("PASS: test_sectionized_csv_still_carries_the_whole_raw_note")



@contextlib.contextmanager
def _capture_logs(logger_name="cpt_rec.baselines.common"):
    """Collect a logger's output so a test can assert on what a run REPORTS,
    not only on what it returns."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    lg = logging.getLogger(logger_name)
    prev_level, prev_prop = lg.level, lg.propagate
    lg.addHandler(handler); lg.setLevel(logging.INFO); lg.propagate = False
    try:
        yield buf
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level); lg.propagate = prev_prop


def test_unified_note_budget_is_audited_not_asserted():
    """The benchmark's input policy has to be checkable from a log, because
    "every row read the same thing" is the claim a benchmark table makes.

    4096 is not arbitrary: it is M2's Longformer native position count, the
    largest budget every component can hold. A run that deviates must SAY so
    in its own log, so `grep DEVIATES-FROM-POLICY` is the audit.
    """
    from cpt_rec.baselines.common import (
        UNIFIED_NOTE_BUDGET, log_note_budget)

    assert UNIFIED_NOTE_BUDGET == 4096

    ok = log_note_budget("M3", 4096, "whole-note")
    assert ok["matches_policy"] is True

    # right budget, wrong text
    assert log_note_budget("M3", 4096, "sections:3")["matches_policy"] is False
    # right text, wrong budget
    assert log_note_budget("M4", 1000, "whole-note")["matches_policy"] is False
    # never set at all
    assert log_note_budget("B?", None, "whole-note")["matches_policy"] is False

    # and the deviation has to be visible in the LOG, not just the return value
    with _capture_logs() as buf:
        log_note_budget("M5", 2048, "sections:3")
    text = buf.getvalue()
    assert "DEVIATES-FROM-POLICY" in text
    assert "not the whole note" in text

    with _capture_logs() as buf:
        log_note_budget("M3", 4096, "whole-note")
    assert "DEVIATES-FROM-POLICY" not in buf.getvalue()

    print("PASS: test_unified_note_budget_is_audited_not_asserted")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Baselines — Smoke Test Suite")
    print("=" * 60)
    test_apply_seed_and_limit_is_deterministic()
    test_log_prediction_stats_returns_expected_dict()
    test_budget_fill_audit_is_opt_in_and_flags_a_short_run()
    test_m1_build_predict_tune_roundtrip()
    test_kb_index_build_predict_with_stubs()
    test_kb_index_history_restricts_candidate_pool()
    test_kb_index_chunks_let_the_cross_encoder_see_past_its_window()
    test_m3_with_echo_backend()
    test_m4_with_echo_backend()
    test_self_consistency_scores_are_agreement_frequencies()
    test_complete_shortlist_never_pads_implicitly()
    test_m3_self_consistency_fills_the_budget()
    test_m4_candidate_filter_has_three_levels_and_defaults_to_candidates()
    test_sectionized_csv_still_carries_the_whole_raw_note()
    test_unified_note_budget_is_audited_not_asserted()
    test_kb_filter_is_default_on_and_can_be_switched_off()
    test_empty_completion_at_token_cap_is_retried_not_swallowed()
    test_server_side_n_is_used_when_supported_and_never_faked()
    print("=" * 60)
    print("ALL BASELINE-SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
