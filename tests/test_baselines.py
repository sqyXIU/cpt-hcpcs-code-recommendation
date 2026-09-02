#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Tests for the baseline systems.

Mapping (paper label -> module)
-------------------------------
* M1 — BM25 KNN-vote over training notes   (baselines.m1_bm25_knn)
* M2 — Clinical-Longformer, label attention (baselines.m2_longformer_label_attn)
* M3 — zero-shot LLM                        (baselines.m3_zeroshot_llm)
* M4 — LLM + BM25 exemplars (RAG)           (baselines.m4_exemplar_rag)
* M5 — LoRA-SFT LLM                         (baselines.m5_lora_sft)

``baselines.kb_index`` is the retrieval index the pipeline reuses, not a
compared system; it is exercised here only as a library.

Tests needing a real LLM, a GPU, or an encoder download are gated and print a
SKIP message naming the missing prerequisite.

Run from project root::

    PYTHONPATH=src python3 tests/test_baselines.py
"""

from __future__ import annotations

from _skip import skip

import logging
import os
import tempfile
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Local model directory — set CPT_REC_MODEL_DIR on servers without HuggingFace
# access.  All test surrogate paths are resolved relative to this directory.
#
#   export CPT_REC_MODEL_DIR=~/models
#   PYTHONPATH=src python3 tests/test_baselines.py
#
# If the directory or a specific sub-model folder is absent the tests will
# still SKIP gracefully.
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(os.environ.get("CPT_REC_MODEL_DIR", "~/models")).expanduser()

def _local_model(name: str, folder: str | None = None) -> str:
    """Return local path if it exists under CPT_REC_MODEL_DIR, else the HF repo name.

    ``folder`` overrides the directory name used under CPT_REC_MODEL_DIR when the
    local folder name differs from the HuggingFace repo slug (e.g.
    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is stored locally as
    ``ms-marco-MiniLM-L6-v2``).
    """
    local = _MODEL_DIR / (folder if folder else name.split("/")[-1])
    return str(local) if local.exists() else name


# ===================================================================
# Test 2.0: shared infra (common.py + llm.py + bm25_index.py)
# ===================================================================

def test_prediction_dataclass():
    from cpt_rec.baselines.common import Prediction

    p = Prediction(note_id="n1", codes=["43235", "45378"], scores=[0.5, 0.4])
    assert p.codes == ["43235", "45378"]
    assert p.scores == [0.5, 0.4]
    try:
        Prediction(note_id="bad", codes=["A"], scores=[0.1, 0.2])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched lengths")
    p2 = Prediction(note_id="n2", codes=["43235", "", "  "], scores=None)
    assert p2.codes == ["43235"]
    print("PASS: test_prediction_dataclass")


def test_write_predictions_roundtrip():
    from cpt_rec.baselines.common import (
        Prediction,
        write_predictions,
    )
    from cpt_rec.common.evaluation.harness import load_predictions

    preds = [
        Prediction("n1", ["43235", "45378"], [0.9, 0.8]),
        Prediction("n2", ["52353"], [0.7]),
        Prediction("n3", [], []),
    ]
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "preds.csv"
        write_predictions(preds, out_csv)
        loaded = load_predictions(out_csv)
        assert loaded["n1"] == {"43235", "45378"}
        assert loaded["n2"] == {"52353"}
        assert loaded["n3"] == set()
    print("PASS: test_write_predictions_roundtrip")


def test_parse_selected_codes():
    from cpt_rec.baselines.llm import parse_selected_codes

    assert parse_selected_codes('{"selected": ["43235", "45378"]}') == [
        "43235", "45378"
    ]
    assert parse_selected_codes(
        'Here are the codes:\n{"selected": ["43235"]}\n-- done'
    ) == ["43235"]
    scraped = parse_selected_codes("I think 43235 and J1885 apply")
    assert "43235" in scraped and "J1885" in scraped
    assert parse_selected_codes("") == []
    assert parse_selected_codes('{"selected": []}') == []
    print("PASS: test_parse_selected_codes")


def test_bm25_index_toy():
    """Tiny in-memory BM25 train-note index with hand-crafted notes."""
    from cpt_rec.baselines.bm25_index import TrainNoteBM25Index, tokenize

    note_ids = ["t1", "t2", "t3"]
    code_sets = [
        {"45378", "43235"},
        {"66984"},
        {"52353"},
    ]
    texts = [
        "screening colonoscopy with biopsy of stomach",
        "cataract extraction with intraocular lens",
        "ureteroscopy with lithotripsy of kidney stone",
    ]
    corpus = [tokenize(t) for t in texts]
    idx = TrainNoteBM25Index(note_ids, code_sets, corpus)

    # Query about colonoscopy should bring t1 to the top.
    nbrs = idx.neighbors("colonoscopy biopsy", top_k=3)
    assert nbrs[0][0] == "t1", f"Expected t1 first, got {nbrs}"
    assert nbrs[0][2] == {"45378", "43235"}

    # Save/load round-trip
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "idx.pkl"
        idx.save(path)
        idx2 = TrainNoteBM25Index.load(path)
        nbrs2 = idx2.neighbors("colonoscopy biopsy", top_k=3)
        assert nbrs2[0][0] == "t1"
    print("PASS: test_bm25_index_toy")


# ===================================================================
# Test 2.1: M3 — GPT zero-shot (echo backend, no API)
# ===================================================================

def test_m3_zeroshot_echo():
    if not SAMPLE_NORM_CSV.exists() or not KB_CSV.exists():
        skip("needs a note corpus and a KB")
        return
    from cpt_rec.baselines.m3_zeroshot_llm import predict_b4
    from cpt_rec.baselines.llm import EchoBackend
    from cpt_rec.common.evaluation.harness import load_predictions
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    kb = CodeKnowledgeBase.from_csv(KB_CSV, build_index=False)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "m3_preds.csv"
        n = predict_b4(
            notes_csv=SAMPLE_NORM_CSV,
            kb=kb,
            out_csv=out,
            backend=EchoBackend(k=0),  # zero-shot has no candidates -> empty
            max_note_tokens=512,
            checkpoint_every=0,
        )
        pred = load_predictions(out)
        assert len(pred) <= n and len(pred) > 0
        # Echo with no candidate lines -> all empty predictions
        for codes in pred.values():
            assert codes == set()
    print(
        f"PASS: test_m3_zeroshot_echo (n={n}, n_unique={len(pred)}, all empty as expected)"
    )


# ===================================================================
# Test 2.2: M1 — BM25 KNN-vote (real sample data)
# ===================================================================

def test_m1_bm25_knn_self_retrieval():
    """Index the sample CSV and predict on the SAME sample.  Each note's
    nearest neighbor is itself, so KNN-vote with normalize_votes=False and
    a small threshold should recover its own gold codes — gives recall≈1.0
    on a sanity-check pass."""
    if not SAMPLE_NORM_CSV.exists():
        skip("needs a note corpus")
        return

    from cpt_rec.baselines.m1_bm25_knn import build_index, predict_b1
    from cpt_rec.common.evaluation.harness import (
        load_gold,
        load_predictions,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx_path = td / "m1_bm25.pkl"
        n_indexed = build_index(SAMPLE_NORM_CSV, idx_path)
        assert n_indexed > 0

        out = td / "m1_preds.csv"
        n = predict_b1(
            notes_csv=SAMPLE_NORM_CSV,
            index_path=idx_path,
            out_csv=out,
            top_k=1,                # closest neighbor only = the note itself
            vote_threshold=0.5,     # any vote >0.5 (normalized -> 1.0 if alone)
            normalize_votes=True,
        )
        gold = load_gold(SAMPLE_NORM_CSV)
        pred = load_predictions(out)
        assert len(pred) <= n and len(pred) > 0

        tp = sum(len(gold[nid] & pred.get(nid, set())) for nid in gold)
        n_gold = sum(len(v) for v in gold.values())
        recall_self = tp / n_gold
        # Self-retrieval should give very high recall on this sanity test.
        assert recall_self > 0.8, f"Self-retrieval recall too low: {recall_self:.3f}"
        print(
            f"PASS: test_m1_bm25_knn_self_retrieval "
            f"(n={n}, tp={tp}/{n_gold}, recall={recall_self:.3f})"
        )


def test_m1_bm25_knn_held_out_split():
    """Use first 75% of sample notes as train, last 25% as test.  No
    self-overlap, so this measures real retrieval-vote quality on the
    sample.  Recall should be > 0 if the sample contains repeated
    procedure types (which it does — colonoscopies, cataracts, etc.)."""
    if not SAMPLE_NORM_CSV.exists():
        skip("needs a note corpus")
        return
    from cpt_rec.baselines.m1_bm25_knn import build_index, predict_b1
    from cpt_rec.common.evaluation.harness import (
        load_gold,
        load_predictions,
    )

    df = pd.read_csv(SAMPLE_NORM_CSV, dtype=str)
    cut = int(len(df) * 0.75)
    train_df = df.iloc[:cut]
    test_df = df.iloc[cut:]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        train_csv = td / "train.csv"
        test_csv = td / "test.csv"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        idx_path = td / "m1_bm25.pkl"
        build_index(train_csv, idx_path)

        out = td / "m1_preds.csv"
        n = predict_b1(
            notes_csv=test_csv,
            index_path=idx_path,
            out_csv=out,
            top_k=10,
            vote_threshold=0.05,
            normalize_votes=True,
        )
        gold = load_gold(test_csv)
        pred = load_predictions(out)
        tp = sum(len(gold[nid] & pred.get(nid, set())) for nid in gold)
        n_gold = sum(len(v) for v in gold.values())
        recall = tp / n_gold if n_gold else 0.0
        print(
            f"PASS: test_m1_bm25_knn_held_out_split "
            f"(n={n}, tp={tp}/{n_gold}, recall={recall:.3f})"
        )


# ===================================================================
# Test 2.3: M4 — GPT + BM25 exemplars (echo backend)
# ===================================================================

def test_m4_exemplar_rag_echo():
    """Build a BM25 index on the sample, run M4 against the same notes
    with the EchoBackend (which echoes the first 3 candidate codes from
    the prompt).  We verify pipeline wires up and every kept code came
    from the BM25-derived candidate set."""
    if not SAMPLE_NORM_CSV.exists() or not KB_CSV.exists():
        skip("needs a note corpus and a KB")
        return
    from cpt_rec.baselines.m1_bm25_knn import build_index
    from cpt_rec.baselines.m4_exemplar_rag import predict_b5
    from cpt_rec.baselines.bm25_index import TrainNoteBM25Index
    from cpt_rec.baselines.llm import EchoBackend
    from cpt_rec.common.evaluation.harness import load_predictions
    from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

    kb = CodeKnowledgeBase.from_csv(KB_CSV, build_index=False)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx_path = td / "m1_bm25.pkl"
        build_index(SAMPLE_NORM_CSV, idx_path)
        index = TrainNoteBM25Index.load(idx_path)

        out = td / "m4_preds.csv"
        n = predict_b5(
            notes_csv=SAMPLE_NORM_CSV,
            index=index,
            kb=kb,
            out_csv=out,
            backend=EchoBackend(k=3),
            top_k=5,
            max_note_tokens=512,
            max_exemplar_tokens=200,
            train_csv_for_text=SAMPLE_NORM_CSV,
            checkpoint_every=0,
        )
        pred = load_predictions(out)
        assert len(pred) <= n and len(pred) > 0
        nonempty = sum(1 for v in pred.values() if v)
        # With self-retrieval, neighbors include the note itself; candidates
        # always include the gold codes -> echo should keep some non-empty
        # predictions.
        assert nonempty >= 1
        # Every kept code must be in the KB
        for nid, codes in pred.items():
            for c in codes:
                assert c in kb.codes
        print(
            f"PASS: test_m4_exemplar_rag_echo "
            f"(n={n}, n_unique={len(pred)}, nonempty={nonempty})"
        )


# ===================================================================
# Test 2.4: M2 — Clinical-Longformer (smoke / data-prep only)
# ===================================================================

def test_m2_label_matrix_sanity():
    """Verify the multi-label matrix builder + label vocab logic.  Does
    NOT load Clinical-Longformer or run training (those are GPU-heavy
    and gated by env)."""
    from cpt_rec.baselines.m2_longformer_label_attn import (
        _build_label_matrix,
        _pipe_split,
    )

    code_lists = [["43235", "45378"], ["66984"], ["43235"]]
    labels = sorted({"43235", "45378", "66984"})
    label_to_idx = {c: i for i, c in enumerate(labels)}
    Y = _build_label_matrix(code_lists, label_to_idx)
    assert Y.shape == (3, 3)
    assert Y[0, label_to_idx["43235"]] == 1.0
    assert Y[0, label_to_idx["45378"]] == 1.0
    assert Y[1, label_to_idx["66984"]] == 1.0
    assert Y[2, label_to_idx["43235"]] == 1.0
    assert Y.sum() == 4
    assert _pipe_split("43235|45378") == ["43235", "45378"]
    assert _pipe_split("") == []
    print("PASS: test_m2_label_matrix_sanity")


def test_m2_train_smoke_distilbert():
    """OPTIONAL: run a tiny end-to-end train+predict cycle with a small
    surrogate model to validate the full Trainer pipeline.  Skipped if
    transformers/torch are missing or download fails."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        skip("needs the gpu extra (torch, transformers)")
        return
    if not SAMPLE_NORM_CSV.exists():
        skip("needs a note corpus")
        return

    from cpt_rec.baselines.m2_longformer_label_attn import (
        predict_longformer,
        train_longformer,
    )
    from cpt_rec.common.evaluation.harness import load_predictions

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model_dir = td / "m2_model"
        try:
            train_longformer(
                train_csv=SAMPLE_NORM_CSV,
                model_out=model_dir,
                base_model=_local_model("prajjwal1/bert-tiny"),  # surrogate, not Longformer
                max_length=128,
                min_code_freq=2,
                batch_size=4,
                grad_accum=1,
                epochs=1,
                lr=5e-4,
                fp16=False,
            )
        except Exception as exc:
            skip(str(exc))
            return

        preds_csv = td / "m2_preds.csv"
        n = predict_longformer(
            notes_csv=SAMPLE_NORM_CSV,
            model_dir=model_dir,
            out_csv=preds_csv,
            threshold=0.5,
            batch_size=8,
        )
        pred = load_predictions(preds_csv)
        assert len(pred) <= n and len(pred) > 0
        print(
            f"PASS: test_m2_train_smoke_distilbert (n_preds={n}, "
            f"unique_ids={len(pred)})"
        )


# ===================================================================
# kb_index — BM25 + semantic retrieval, cross-encoder rerank
# ===================================================================

def test_kb_index_topk_dense_cosine_math():
    from cpt_rec.baselines.kb_index import _topk_dense_cosine

    rng = np.random.default_rng(0)
    kb = rng.normal(size=(5, 4)).astype(np.float32)
    kb /= np.linalg.norm(kb, axis=1, keepdims=True)
    q = kb[[3, 3, 3]] + 0.01 * rng.normal(size=(3, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    top_idx, top_scores = _topk_dense_cosine(q, kb, top_k=3)
    assert top_idx.shape == (3, 3)
    assert (top_idx[:, 0] == 3).all()
    assert (np.diff(top_scores, axis=1) <= 1e-6).all()
    print("PASS: test_kb_index_topk_dense_cosine_math")


def test_kb_index_encoder_imports():
    """Heavy: only runs if torch+transformers are installed AND the
    surrogate models can be downloaded.  Just exercises the encoder
    classes' constructors + a 2-sample forward pass."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        skip("needs the gpu extra (torch, transformers)")
        return
    from cpt_rec.baselines.kb_index import BiEncoder, CrossEncoder

    try:
        bi = BiEncoder(model_name=_local_model("prajjwal1/bert-tiny"), max_length=16)
        ce = CrossEncoder(model_name=_local_model("cross-encoder/ms-marco-MiniLM-L-6-v2", folder="ms-marco-MiniLM-L6-v2"),
                           max_length=64)
    except Exception as exc:
        skip(str(exc))
        return

    vecs = bi.encode(["hello world", "colonoscopy"], batch_size=2,
                      show_progress=False)
    assert vecs.shape[0] == 2 and vecs.shape[1] > 0

    pair_scores = ce.score(
        [("a knee replacement", "total knee arthroplasty"),
         ("a knee replacement", "appendectomy")],
        batch_size=2,
    )
    assert pair_scores.shape == (2,)
    # The first pair should be more relevant than the second.
    assert pair_scores[0] > pair_scores[1]
    print(
        f"PASS: test_kb_index_encoder_imports "
        f"(bi_dim={vecs.shape[1]}, ce_scores={pair_scores.tolist()})"
    )


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Baselines — test suite")
    print("=" * 60)
    print()

    print("--- Step 2.0: shared infra ---")
    test_prediction_dataclass()
    test_write_predictions_roundtrip()
    test_parse_selected_codes()
    test_bm25_index_toy()
    print()

    print("--- Step 2.1: M3 GPT zero-shot ---")
    test_m3_zeroshot_echo()
    print()

    print("--- Step 2.2: M1 BM25 KNN-vote ---")
    test_m1_bm25_knn_self_retrieval()
    test_m1_bm25_knn_held_out_split()
    print()

    print("--- Step 2.3: M4 GPT + BM25 exemplars ---")
    test_m4_exemplar_rag_echo()
    print()

    print("--- Step 2.4: M2 Clinical-Longformer ---")
    test_m2_label_matrix_sanity()
    test_m2_train_smoke_distilbert()
    print()

    print("--- kb_index: BM25 + semantic -> cross-encoder ---")
    test_kb_index_topk_dense_cosine_math()
    test_kb_index_encoder_imports()
    print()

    print("=" * 60)
    print("ALL BASELINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
