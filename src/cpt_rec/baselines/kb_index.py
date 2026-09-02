#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Knowledge-base retrieval index (``kb_index``) — **not a reported system**.

It builds the BM25 + dense indexes over KB code descriptions that M6's
candidate generation reads, and additionally carries a retrieve-and-rerank
CLI that no reported table uses.

Two-stage pipeline over the CPT/HCPCS knowledge base:

1. **Candidate generation** — union of two retrievers indexed over code
   descriptions:

   * BM25 over tokenized descriptions (lexical recall).
   * Bi-encoder (default ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext``)
     dense top-k (semantic recall).

2. **Cross-encoder reranking** — for each ``(note, code_description)``
   candidate pair, score with a HuggingFace cross-encoder (default
   ``cross-encoder/ms-marco-MiniLM-L-6-v2``) and keep codes whose rerank
   score is at least ``--rerank-threshold``.

Two CLI subcommands
-------------------

::

    cptrec-build-kb-index build-index \\
        --kb data/kb/codes_with_ranges.csv \\
        --out-dir outputs/indices/code_kb_faiss/default/ \\
        --biencoder cambridgeltl/SapBERT-from-PubMedBERT-fulltext

    cptrec-build-kb-index predict \\
        --notes outputs/datasets/vumc/test_eval_sectioned.csv \\
        --kb data/kb/codes_with_ranges.csv \\
        --index-dir outputs/indices/code_kb_faiss/default/ \\
        --out outputs/baselines/kb_index/predictions/test.csv \\
        --bm25-top-k 50 --dense-top-k 50 \\
        --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --rerank-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from cpt_rec.baselines.bm25_index import _build_bm25, tokenize
from cpt_rec.baselines.common import (
    Prediction,
    apply_seed_and_limit,
    load_notes_for_prediction,
    log_note_budget,
    log_prediction_stats,
    stats_sidecar,
    maybe_load_code_history,
    parse_as_of,
    resolve_local_model,
    restrict_to_active,
    truncate_text_by_tokens,
    write_predictions,
    write_scores_npz,
)
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

LOGGER = logging.getLogger(__name__)


# Defaults refer to the basenames used for the offline cache layout
# (``models/SapBERT-from-PubMedBERT-fulltext/`` and
# ``models/ms-marco-MiniLM-L6-v2/``).  The canonical Hub repo ids are
# ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext`` and
# ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — pass them on a machine
# with internet access if you want to pull from the Hub directly.
DEFAULT_BIENCODER = "SapBERT-from-PubMedBERT-fulltext"
DEFAULT_RERANKER = "ms-marco-MiniLM-L6-v2"
INDEX_FORMAT_VERSION = 2  # bump when the artifact layout changes


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class BiEncoder:
    """Mean / CLS pooled bi-encoder, L2-normalized."""

    def __init__(
        self,
        model_name: str = DEFAULT_BIENCODER,
        max_length: int = 64,
        device: Optional[str] = None,
        pooling: str = "cls",
        local_models_dir: Optional[Path] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "kb_index requires `torch` and `transformers`: "
                "`pip install torch transformers`"
            ) from exc

        self.torch = torch
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        resolved = resolve_local_model(model_name, models_dir=local_models_dir)
        LOGGER.info("Loading bi-encoder %s (resolved=%s) on %s",
                    model_name, resolved, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModel.from_pretrained(resolved).to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 128,
               show_progress: bool = True,
               max_length_override: Optional[int] = None) -> np.ndarray:
        """
        Encode *texts* into L2-normalized embeddings.

        ``max_length_override`` lets the caller use a different sequence
        budget than the one fixed at construction — important for kb_index,
        where the *codes* side is encoded at ~64 tokens (descriptions
        are short) but the *query* side needs ~256+ tokens to capture an
        operative-note excerpt.  Without this knob, calling ``encode``
        on a 512-token note would silently truncate to ``max_length=64``
        and discard ~87 % of the input.
        """
        torch = self.torch
        eff_max_length = max_length_override or self.max_length
        out_vecs: List[np.ndarray] = []
        rng = range(0, len(texts), batch_size)
        if show_progress:
            rng = tqdm(rng, desc="bi-encoder",
                        total=(len(texts) + batch_size - 1) // batch_size)
        with torch.no_grad():
            for start in rng:
                batch = list(texts[start : start + batch_size])
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=eff_max_length, return_tensors="pt",
                ).to(self.device)
                hs = self.model(**enc).last_hidden_state
                if self.pooling == "cls":
                    pooled = hs[:, 0, :]
                else:  # mean
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out_vecs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(out_vecs) if out_vecs else np.zeros((0, 0), dtype=np.float32)


class CrossEncoder:
    """HF cross-encoder; scores ``(text_a, text_b)`` pairs into one logit."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        max_length: int = 256,
        device: Optional[str] = None,
        local_models_dir: Optional[Path] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("kb_index requires torch + transformers") from exc
        self.torch = torch
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        resolved = resolve_local_model(model_name, models_dir=local_models_dir)
        LOGGER.info("Loading cross-encoder %s (resolved=%s) on %s",
                    model_name, resolved, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved
        ).to(self.device)
        self.model.eval()

        # ``max_length`` is the budget for the PACKED PAIR
        # ``[CLS] note [SEP] description [SEP]`` -- not per side, and not a
        # number of pairs.  It cannot exceed the encoder's learned position
        # embeddings (512 for every BERT-family reranker, this one included),
        # so it can never be set from the note-length distribution: operative
        # notes run 3-5x that.  Say so out loud rather than letting a silent
        # default look like a tuned choice.
        ceiling = int(getattr(self.model.config, "max_position_embeddings", 0))
        if ceiling and self.max_length > ceiling:
            LOGGER.warning(
                "rerank max_length=%d exceeds the encoder's position-embedding "
                "ceiling of %d; it will be clipped. Use --rerank-chunks to "
                "cover a long note instead of raising this.",
                self.max_length, ceiling,
            )
        LOGGER.info(
            "Cross-encoder pair budget: %d tokens (encoder ceiling %s). "
            "HF truncates `longest_first`, so a ~10-30 token code description "
            "survives intact and the NOTE absorbs every truncation step.",
            self.max_length, ceiling or "unknown",
        )

    def score(self, pairs: Sequence[Tuple[str, str]],
              batch_size: int = 32) -> np.ndarray:
        """
        Score (text_a, text_b) pairs.  Output is in [0, 1] so threshold
        sweeps live in a stable, model-agnostic range:

        * 1-logit reranker (e.g. ``ms-marco-MiniLM-L-6-v2``) → sigmoid(logit).
        * 2-class reranker → softmax + take the "relevant" probability.
        """
        torch = self.torch
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = list(pairs[start : start + batch_size])
                a = [p[0] for p in batch]
                b = [p[1] for p in batch]
                enc = self.tokenizer(
                    a, b, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                if logits.shape[-1] == 1:
                    s = torch.sigmoid(logits.squeeze(-1))
                else:
                    s = torch.softmax(logits, dim=-1)[:, -1]
                out.append(s.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

_DEFAULT_KB_TEXT_FIELDS: Tuple[str, ...] = ("code_description",)


def _resolve_kb_field(row: pd.Series, field: str) -> str:
    """
    Resolve one indexable field for a KB row.  Supports:

    * any literal column name on the KB CSV (``code_description``,
      ``code_lay_term``, ``code_range_3_description``, …);
    * the synthetic field ``code_range_deepest``: walks from
      ``code_range_6_description`` down to ``code_range_1_description``
      and returns the first non-empty value (i.e. the most specific
      hierarchy node that this code actually has — necessary because
      not every code populates all 6 levels);
    * the synthetic field ``code_range_path``: joins every non-empty
      ``code_range_{i}_description`` from i=1..6 with `` > `` so the
      encoder sees a breadcrumb-style "Surgery > Digestive > Endoscopy
      > EGD" chain.
    """
    def _val(col: str) -> str:
        if col in row.index:
            v = row[col]
            if pd.isna(v):
                return ""
            return str(v).strip()
        return ""

    if field == "code_range_deepest":
        for lvl in range(6, 0, -1):
            v = _val(f"code_range_{lvl}_description")
            if v:
                return v
        return ""
    if field == "code_range_path":
        parts: List[str] = []
        for lvl in range(1, 7):
            v = _val(f"code_range_{lvl}_description")
            if v:
                parts.append(v)
        return " > ".join(parts)
    return _val(field)


def _kb_descriptions(
    kb: CodeKnowledgeBase,
    fields: Sequence[str] = _DEFAULT_KB_TEXT_FIELDS,
    field_separator: str = " | ",
) -> Tuple[List[str], List[str]]:
    """
    Build the indexable text for every KB code.

    ``fields`` controls which columns + synthetic fields contribute to
    each code's text.  Empty values for a given (code, field) are
    silently skipped, so a code that lacks ``code_range_6_description``
    still gets a clean concatenation of the non-empty fields.

    The first line is always ``"<code>. <code_description>"`` so the
    code itself is in the BM25 vocabulary even when the user passes
    only hierarchy fields.
    """
    codes: List[str] = []
    texts: List[str] = []
    df = kb._df  # access the underlying DataFrame for richer field lookup

    for code in sorted(kb.codes):
        codes.append(code)
        primary = (kb.description(code) or "").strip()
        # Header line: "<code>. <description>" — always present.
        line0 = f"{code}. {primary}" if primary else f"{code}."
        parts: List[str] = [line0]

        # Resolve each requested field; skip empties so a code with
        # only 3 hierarchy levels doesn't get padded with empty " | ".
        if code in df.index:
            row = df.loc[code]
            for field in fields:
                if field == "code_description":
                    # already covered by line0; skip to avoid duplication
                    continue
                v = _resolve_kb_field(row, field)
                if v and v.lower() != primary.lower():
                    parts.append(v)

        texts.append(field_separator.join(parts))
    return codes, texts


def build_index(
    kb_csv: Path,
    out_dir: Path,
    biencoder: str = DEFAULT_BIENCODER,
    bi_max_length: int = 64,
    bi_batch_size: int = 128,
    kb_text_fields: Sequence[str] = _DEFAULT_KB_TEXT_FIELDS,
    local_models_dir: Optional[Path] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
    codes, texts = _kb_descriptions(kb, fields=kb_text_fields)
    LOGGER.info(
        "Indexing %d KB codes with text fields: %s",
        len(codes), list(kb_text_fields),
    )
    # Quick stats on the resulting text-length distribution so the user
    # can sanity-check the indexing decision (e.g. tokens-per-code grew
    # from 8 to 30 after adding lay_term + hierarchy → we expect dense
    # recall to improve).
    char_lens = np.array([len(t) for t in texts], dtype=np.int64)
    LOGGER.info(
        "Indexed-text char length: min=%d, p50=%d, p95=%d, max=%d",
        int(char_lens.min()), int(np.percentile(char_lens, 50)),
        int(np.percentile(char_lens, 95)), int(char_lens.max()),
    )

    # 1) BM25 — tokenized corpus + the raw text strings, so predict time
    # doesn't need a second `CodeKnowledgeBase.from_csv` load just to
    # render reranker pairs.
    bm25_corpus = [tokenize(t) for t in texts]
    np.savez_compressed(
        out_dir / "bm25_corpus.npz",
        codes=np.array(codes, dtype=object),
        corpus=np.array(bm25_corpus, dtype=object),
        descriptions=np.array(texts, dtype=object),
    )
    LOGGER.info("Saved BM25 corpus -> %s", out_dir / "bm25_corpus.npz")

    # 2) Dense
    enc = BiEncoder(
        model_name=biencoder,
        max_length=bi_max_length,
        local_models_dir=local_models_dir,
    )
    vecs = enc.encode(texts, batch_size=bi_batch_size)
    np.savez_compressed(
        out_dir / "dense.npz",
        codes=np.array(codes, dtype=object),
        vectors=vecs,
        model_name=np.array([biencoder], dtype=object),
        max_length=np.array([bi_max_length], dtype=np.int32),
    )
    LOGGER.info("Saved dense KB index %s -> %s", vecs.shape, out_dir / "dense.npz")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "format_version": INDEX_FORMAT_VERSION,
                "biencoder": biencoder,
                "bi_max_length": bi_max_length,
                "n_codes": len(codes),
                "kb_csv": str(kb_csv),
                "kb_text_fields": list(kb_text_fields),
                "indexed_text_char_len_p50": int(np.percentile(char_lens, 50)),
                "indexed_text_char_len_p95": int(np.percentile(char_lens, 95)),
            },
            f, indent=2,
        )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str, n_chunks: int) -> List[str]:
    """
    Split *text* into ``n_chunks`` roughly equal-length pieces by word
    count.

    Used by both diagnose and predict to issue separate retrieval
    queries per chunk and union the results.  Catches codes that are
    only mentioned in one section of a long note — which is the failure
    mode the diagnostic exposed (gold codes for distinct procedures
    landing at ranks 150 / 471 / <500 in the same note because the
    full-note query averages over all procedures).

    Word-level splitting is intentional: it's tokenizer-independent so
    the chunk boundaries stay stable across BM25 and the bi-encoder.
    """
    if n_chunks <= 1:
        return [text]
    words = text.split()
    if not words:
        return [text]
    if n_chunks > len(words):
        return [text]
    chunk_size = max(1, len(words) // n_chunks)
    out: List[str] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            out.append(chunk)
    return out or [text]


def _topk_dense_cosine(q: np.ndarray, kb: np.ndarray, top_k: int):
    sims = q @ kb.T
    if top_k < sims.shape[1]:
        part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
        rows = np.arange(sims.shape[0])[:, None]
        ps = sims[rows, part]
        order = np.argsort(-ps, axis=1)
        idx = part[rows, order]
        scores = ps[rows, order]
    else:
        order = np.argsort(-sims, axis=1)
        idx = order[:, :top_k]
        rows = np.arange(sims.shape[0])[:, None]
        scores = sims[rows, idx]
    return idx, scores


def predict_b3(
    notes_csv: Path,
    kb_csv: Optional[Path],
    index_dir: Path,
    out_csv: Path,
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    reranker: str = DEFAULT_RERANKER,
    rerank_max_length: int = 384,
    rerank_batch_size: int = 32,
    rerank_threshold: float = 0.05,
    max_keep: Optional[int] = None,
    biencoder: Optional[str] = None,
    bi_max_length: Optional[int] = None,
    bi_query_max_length: int = 256,
    bi_batch_size: int = 64,
    max_note_tokens: int = 512,
    rerank_min_keep: int = 5,
    rerank_chunks: int = 1,
    rerank_mode: str = "cross",
    rrf_k: int = 60,
    query_chunks: int = 1,
    seed: int = 42,
    limit: Optional[int] = None,
    history_changes: Optional[Path] = None,
    history_deleted: Optional[Path] = None,
    as_of_col: str = "PROCEDURE_DATE",
    local_models_dir: Optional[Path] = None,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    dump_scores_npz: Optional[Path] = None,
) -> int:
    index_dir = Path(index_dir)
    with open(index_dir / "manifest.json") as f:
        manifest = json.load(f)
    fmt_version = int(manifest.get("format_version", 1))
    if fmt_version > INDEX_FORMAT_VERSION:
        raise ValueError(
            f"kb_index index at {index_dir} has format_version={fmt_version}, "
            f"but this code only supports up to {INDEX_FORMAT_VERSION}; "
            f"upgrade the package or rebuild the index."
        )
    bi_model = biencoder or manifest["biencoder"]
    # `bi_max_length` is the *KB-side* sequence budget — keep the value the
    # index was built with, since changing it would invalidate the cached
    # dense vectors.  The query-side budget is independent.
    bi_kb_max_length = bi_max_length or int(manifest.get("bi_max_length", 64))

    # Load indexes
    bm25_dat = np.load(index_dir / "bm25_corpus.npz", allow_pickle=True)
    bm25_codes = list(bm25_dat["codes"])
    bm25_corpus = [list(d) for d in bm25_dat["corpus"]]
    bm25 = _build_bm25(bm25_corpus)

    dense_dat = np.load(index_dir / "dense.npz", allow_pickle=True)
    dense_codes = list(dense_dat["codes"])
    dense_vecs = dense_dat["vectors"].astype(np.float32)
    if bm25_codes != dense_codes:
        raise ValueError("BM25 and dense indexes have inconsistent code orderings.")

    # Prefer the descriptions baked into the index artifact (format v2+);
    # fall back to a CodeKnowledgeBase re-load if the artifact is older.
    if "descriptions" in bm25_dat.files:
        descriptions = list(bm25_dat["descriptions"])
        code_to_desc = dict(zip(bm25_codes, descriptions))
    else:
        if kb_csv is None:
            raise ValueError(
                "kb_index index has no embedded descriptions (format_version<2); "
                "pass --kb to load them from the KB CSV."
            )
        kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
        code_to_desc = {c: (kb.description(c) or "") for c in bm25_codes}

    if rerank_mode not in ("cross", "fusion"):
        raise ValueError(
            f"rerank_mode must be 'cross' or 'fusion', got {rerank_mode!r}"
        )

    bi_enc = BiEncoder(
        model_name=bi_model,
        max_length=bi_kb_max_length,
        local_models_dir=local_models_dir,
    )
    # Skip the cross-encoder entirely in fusion mode — saves ~1 model
    # load + per-note inference cost, and avoids the OOD-on-medical-text
    # failure mode where MS-MARCO rerankers collapse to ~0 sigmoid for
    # every pair.
    cross_enc = None
    if rerank_mode == "cross":
        cross_enc = CrossEncoder(
            model_name=reranker,
            max_length=rerank_max_length,
            local_models_dir=local_models_dir,
        )

    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)
    LOGGER.info("Loaded %d notes from %s", len(df), notes_csv)

    # kb_index has the active KB code set in memory (baked into the retrieval
    # index), so we hand it to the history loader directly — no need to
    # re-read the KB CSV on the predict path, which means --kb is
    # genuinely optional (only needed for legacy v1 indexes).
    history, _ = maybe_load_code_history(
        history_changes=history_changes,
        history_deleted=history_deleted,
        kb_csv=kb_csv,
        active_kb_codes=bm25_codes,
    )
    if history is not None and as_of_col not in df.columns:
        LOGGER.warning(
            "kb_index history requested but as_of column %r not in CSV; "
            "candidate-pool restriction will be a no-op",
            as_of_col,
        )

    truncated = [
        truncate_text_by_tokens(t, max_tokens=max_note_tokens)
        for t in df["note_text"].tolist()
    ]

    # Build all per-note chunks once and encode them in a single batch.
    # query_chunks=1 reproduces today's behavior exactly (each note is
    # one query); larger N issues separate retrieval queries per chunk
    # and merges by best-rank, catching codes that are mentioned only
    # in one part of a multi-procedure note.
    LOGGER.info("kb_index query_chunks=%d rerank_chunks=%d max_note_tokens=%d",
                query_chunks, rerank_chunks, max_note_tokens)
    log_note_budget(
        "kb_index", max_note_tokens, "whole-note",
        covered_by=(
            f"retrieval {query_chunks} chunk(s), rerank {rerank_chunks} "
            f"chunk(s) -- the cross-encoder holds 512 positions, so a budget "
            f"above that is only honoured by chunking"
        ),
    )
    if max_note_tokens > 512 and rerank_chunks <= 1:
        LOGGER.warning(
            "kb_index: --max-note-tokens %d with --rerank-chunks 1 means the "
            "reranker still sees only its first window; pass "
            "--rerank-chunks %d to actually cover the budget.",
            max_note_tokens, max(2, -(-max_note_tokens // 512)),
        )
    note_to_chunk_rows: List[List[int]] = []
    chunk_texts: List[str] = []
    for t in truncated:
        chunks = _split_into_chunks(t, n_chunks=query_chunks)
        idx_list: List[int] = []
        for c in chunks:
            idx_list.append(len(chunk_texts))
            chunk_texts.append(c)
        note_to_chunk_rows.append(idx_list)
    # Query-side encoding uses its own (longer) max_length so the bi-encoder
    # actually sees the bulk of the operative-note excerpt.
    q_vecs = bi_enc.encode(
        chunk_texts,
        batch_size=bi_batch_size,
        max_length_override=bi_query_max_length,
    )

    predictions: List[Prediction] = []

    pool_rankings: List[Tuple[str, List[str], List[float]]] = []
    # Track score distribution across the first 50 notes for diagnostics —
    # the MS-MARCO cross-encoder produces wildly different sigmoid ranges
    # depending on whether the (note, code-desc) pair is in-distribution,
    # so logging the actual range lets the caller retune the threshold
    # from real data instead of guessing.
    sampled_score_quantiles: List[Tuple[float, float, float]] = []
    n_zero_after_threshold = 0

    LOGGER.info("kb_index rerank mode: %s", rerank_mode)

    note_ids = df["note_id"].astype(str).tolist()
    # Parallel to note_ids. [None] * len(df) when history is off or the column
    # is absent, which is exactly what the warning above promises. Same shape
    # as M1's as_of_raw; kb_index used to read a `row` that never existed in this
    # scope, so the restriction below had never run at all.
    as_of_raw = (
        df[as_of_col].tolist() if (history is not None and as_of_col in df.columns)
        else [None] * len(df)
    )
    for i, note_id in enumerate(
        tqdm(note_ids, total=len(df), desc=f"kb_index rerank ({rerank_mode})")
    ):
        text = truncated[i]
        chunk_rows = note_to_chunk_rows[i]
        chunks_for_note = [chunk_texts[r] for r in chunk_rows]

        # BM25 retrieval per chunk → merge by best (lowest) rank seen.
        # Codes that get a strong BM25 score in any single chunk surface
        # at that chunk's natural rank, even if they would have been
        # buried in a full-note query.
        bm25_best_rank: Dict[int, int] = {}
        bm25_best_score: Dict[int, float] = {}
        for chunk in chunks_for_note:
            scores = bm25.get_scores(tokenize(chunk))
            if bm25_top_k < len(scores):
                part = np.argpartition(-scores, bm25_top_k - 1)[:bm25_top_k]
                this_order = part[np.argsort(-scores[part])]
            else:
                this_order = np.argsort(-scores)
            for r, j in enumerate(this_order):
                jj = int(j)
                if jj not in bm25_best_rank or r < bm25_best_rank[jj]:
                    bm25_best_rank[jj] = r
                    bm25_best_score[jj] = float(scores[jj])
        bm25_idx = np.array(
            [j for j, _ in sorted(bm25_best_rank.items(), key=lambda kv: kv[1])],
            dtype=np.int64,
        )

        # Dense retrieval per chunk → merge by best (lowest) rank.
        dense_best_rank: Dict[int, int] = {}
        dense_best_score: Dict[int, float] = {}
        for r_idx in chunk_rows:
            d_idx_row, d_score_row = _topk_dense_cosine(
                q_vecs[r_idx : r_idx + 1], dense_vecs, top_k=dense_top_k
            )
            for r, j in enumerate(d_idx_row[0]):
                jj = int(j)
                if jj not in dense_best_rank or r < dense_best_rank[jj]:
                    dense_best_rank[jj] = r
                    dense_best_score[jj] = float(d_score_row[0][r])
        dense_idx = np.array(
            [j for j, _ in sorted(dense_best_rank.items(), key=lambda kv: kv[1])],
            dtype=np.int64,
        )
        dense_scores_aligned = np.array(
            [dense_best_score[int(j)] for j in dense_idx], dtype=np.float32
        )

        # Per-candidate provenance: rank in BM25 list, rank in dense list,
        # raw dense cosine, raw BM25 score.  Used by both fusion ranking
        # and any future explainability surface.
        cand_info: Dict[int, Dict[str, float]] = {}
        for r, j in enumerate(bm25_idx):
            jj = int(j)
            cand_info.setdefault(jj, {})["bm25_rank"] = r
            cand_info[jj]["bm25_score"] = float(bm25_best_score.get(jj, 0.0))
        for r, j in enumerate(dense_idx):
            jj = int(j)
            cand_info.setdefault(jj, {})["dense_rank"] = r
            cand_info[jj]["dense_score"] = float(dense_scores_aligned[r])

        cand_ids = sorted(cand_info.keys())
        cand_codes = [bm25_codes[j] for j in cand_ids]
        # Per-note CodeHistory restriction: drop candidates that aren't
        # active on the note's PROCEDURE_DATE.  No-op when history is None.
        if history is not None:
            as_of = parse_as_of(as_of_raw[i])
            if as_of is not None:
                allowed = restrict_to_active(cand_codes, history=history, as_of=as_of)
                keep_mask = [c in allowed for c in cand_codes]
                cand_ids = [j for j, k in zip(cand_ids, keep_mask) if k]
                cand_codes = [c for c, k in zip(cand_codes, keep_mask) if k]
        if not cand_codes:
            predictions.append(Prediction(note_id=note_id, codes=[], scores=[]))
            if dump_scores_npz is not None:
                pool_rankings.append((note_id, [], []))
            continue

        # ----------------------------- ranking ---------------------------
        if rerank_mode == "cross":
            # Cross-encoder rerank.  With --rerank-chunks N the note is split
            # into N pieces, every piece is scored against every candidate,
            # and each candidate keeps its BEST piece.  That is the only way
            # to cover a note longer than the encoder's 512-token ceiling: a
            # code documented in the operative detail is otherwise judged on
            # the note's header, which is what the pair budget leaves visible.
            # N=1 is byte-identical to the historical path.
            if rerank_chunks <= 1:
                pairs = [(text, code_to_desc[c]) for c in cand_codes]
                scores = cross_enc.score(pairs, batch_size=rerank_batch_size)
            else:
                per_chunk = np.stack([
                    cross_enc.score(
                        [(part, code_to_desc[c]) for c in cand_codes],
                        batch_size=rerank_batch_size,
                    )
                    for part in _split_into_chunks(text, n_chunks=rerank_chunks)
                ])
                scores = per_chunk.max(axis=0)
            order = np.argsort(-scores)

            if i < 50 and len(scores):
                sampled_score_quantiles.append(
                    (float(np.min(scores)),
                     float(np.median(scores)),
                     float(np.max(scores)))
                )

            rank_scores = scores
            kept_codes: List[str] = []
            kept_scores: List[float] = []
            for j in order:
                if scores[j] < rerank_threshold:
                    break
                kept_codes.append(cand_codes[int(j)])
                kept_scores.append(float(scores[int(j)]))
                if max_keep is not None and len(kept_codes) >= max_keep:
                    break

            if len(kept_codes) == 0:
                n_zero_after_threshold += 1
                floor = min(rerank_min_keep, len(order))
                for j in order[:floor]:
                    kept_codes.append(cand_codes[int(j)])
                    kept_scores.append(float(scores[int(j)]))

        else:
            # rerank_mode == "fusion" — Reciprocal Rank Fusion of BM25
            # and dense.  RRF score for a candidate is the sum over the
            # rankers it appears in:  Σ 1 / (k + rank).
            # Candidates that appear in only one ranker still get a
            # non-trivial score; candidates absent from both are filtered
            # earlier.
            rrf_scores = np.zeros(len(cand_codes), dtype=np.float32)
            for pos, j in enumerate(cand_ids):
                info = cand_info[int(j)]
                if "bm25_rank" in info:
                    rrf_scores[pos] += 1.0 / (rrf_k + info["bm25_rank"])
                if "dense_rank" in info:
                    rrf_scores[pos] += 1.0 / (rrf_k + info["dense_rank"])
            order = np.argsort(-rrf_scores)

            if i < 50 and len(rrf_scores):
                sampled_score_quantiles.append(
                    (float(np.min(rrf_scores)),
                     float(np.median(rrf_scores)),
                     float(np.max(rrf_scores)))
                )

            rank_scores = rrf_scores
            cap = max_keep if max_keep is not None else max(rerank_min_keep, 10)
            kept_codes = [cand_codes[int(j)] for j in order[:cap]]
            kept_scores = [float(rrf_scores[int(j)]) for j in order[:cap]]

        predictions.append(
            Prediction(note_id=note_id, codes=kept_codes, scores=kept_scores)
        )
        if dump_scores_npz is not None:
            # The whole reranked candidate union, before --rerank-threshold
            # and --max-keep: the pool a reviewer could be shown at any
            # budget B, not this run's operating point.
            pool_rankings.append((
                note_id,
                [cand_codes[int(j)] for j in order],
                [float(rank_scores[int(j)]) for j in order],
            ))

    if sampled_score_quantiles:
        arr = np.array(sampled_score_quantiles)
        median_max = float(np.median(arr[:, 2]))
        LOGGER.info(
            "kb_index score sample (mode=%s, first %d notes): "
            "min p50=%.4f, median p50=%.4f, max p50=%.4f",
            rerank_mode,
            len(sampled_score_quantiles),
            float(np.median(arr[:, 0])),
            float(np.median(arr[:, 1])),
            median_max,
        )
        # Loud warning when the cross-encoder produces near-zero
        # sigmoid scores for every pair — a sure sign it's
        # out-of-distribution on this data and is contributing nothing
        # except a flat ordering on top of the BM25 ∪ dense union.
        if rerank_mode == "cross" and median_max < 0.01:
            LOGGER.warning(
                "kb_index: cross-encoder is producing near-zero sigmoid scores "
                "(median max=%.4f).  This reranker is OOD on medical "
                "coding pairs — its ordering carries little signal.  "
                "Re-run with --rerank-mode fusion to skip the cross-"
                "encoder and rank by Reciprocal Rank Fusion of BM25 + "
                "dense instead.",
                median_max,
            )
    if rerank_mode == "cross" and n_zero_after_threshold:
        LOGGER.info(
            "kb_index: %d/%d notes had no candidate above threshold=%.3f and "
            "fell back to top-%d (consider --rerank-mode fusion or "
            "lowering --rerank-threshold based on the score sample above)",
            n_zero_after_threshold, len(predictions),
            rerank_threshold, rerank_min_keep,
        )

    write_predictions(predictions, out_csv, include_scores=True)
    if dump_scores_npz is not None:
        write_scores_npz(pool_rankings, dump_scores_npz)
    log_prediction_stats(predictions, label="kb_index",
                         out_path=stats_sidecar(out_csv))
    return len(predictions)


# ---------------------------------------------------------------------------
# Diagnose: per-stage candidate recall + gold-code rank distribution
# ---------------------------------------------------------------------------

def _ranks_of(target: Set[str], ordered_ids: Sequence[int],
              id_to_code: Sequence[str]) -> Dict[str, Optional[int]]:
    """For each gold code in ``target``, return its rank (0-indexed) in
    ``ordered_ids`` or None if absent."""
    out: Dict[str, Optional[int]] = {c: None for c in target}
    for r, j in enumerate(ordered_ids):
        c = id_to_code[int(j)]
        if c in out and out[c] is None:
            out[c] = r
    return out


def diagnose_b3(
    notes_csv: Path,
    index_dir: Path,
    out_json: Path,
    audit_csv: Optional[Path] = None,
    gold_code_col: str = "proc_codes",
    bm25_top_k: int = 500,
    dense_top_k: int = 500,
    query_chunks: int = 1,
    biencoder: Optional[str] = None,
    bi_max_length: Optional[int] = None,
    bi_query_max_length: int = 256,
    bi_batch_size: int = 64,
    max_note_tokens: int = 512,
    seed: int = 42,
    limit: Optional[int] = None,
    local_models_dir: Optional[Path] = None,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
) -> Dict:
    """
    Run the candidate-generation step of kb_index against a labeled CSV and
    report retrieval recall + gold-code rank distribution.

    No reranker is loaded — we just want to know whether the BM25 ∪ dense
    union contains the gold codes at all.  If it doesn't, no rerank /
    fusion can recover, and the index needs a redesign rather than a parameter
    tweak.

    Output:
      * a JSON report with aggregate stats,
      * an optional per-note audit CSV (one row per gold code with its
        ranks in BM25, dense, and the union ordering),
      * a console summary.
    """
    index_dir = Path(index_dir)
    with open(index_dir / "manifest.json") as f:
        manifest = json.load(f)
    bi_model = biencoder or manifest["biencoder"]
    bi_kb_max_length = bi_max_length or int(manifest.get("bi_max_length", 64))

    bm25_dat = np.load(index_dir / "bm25_corpus.npz", allow_pickle=True)
    bm25_codes = list(bm25_dat["codes"])
    bm25_corpus = [list(d) for d in bm25_dat["corpus"]]
    bm25 = _build_bm25(bm25_corpus)

    dense_dat = np.load(index_dir / "dense.npz", allow_pickle=True)
    dense_codes = list(dense_dat["codes"])
    dense_vecs = dense_dat["vectors"].astype(np.float32)
    if bm25_codes != dense_codes:
        raise ValueError("BM25 and dense indexes have inconsistent code orderings.")

    bi_enc = BiEncoder(
        model_name=bi_model,
        max_length=bi_kb_max_length,
        local_models_dir=local_models_dir,
    )

    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    if gold_code_col not in df.columns:
        raise ValueError(
            f"Diagnose CSV missing gold column {gold_code_col!r}; "
            f"have {list(df.columns)}"
        )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)
    LOGGER.info("Diagnose: %d labeled notes loaded from %s", len(df), notes_csv)

    truncated = [
        truncate_text_by_tokens(t, max_tokens=max_note_tokens)
        for t in df["note_text"].tolist()
    ]

    # Per-note chunked queries: split each truncated note into N chunks,
    # encode all of them in a single batch, and remember which note each
    # chunk came from.  query_chunks=1 reproduces today's behavior
    # exactly (each note is one query).
    LOGGER.info("Diagnose: query_chunks=%d", query_chunks)
    note_to_chunk_rows: List[List[int]] = []  # per note → list of row idx in q_vecs / chunk_texts
    chunk_texts: List[str] = []
    for t in truncated:
        chunks = _split_into_chunks(t, n_chunks=query_chunks)
        idx_list: List[int] = []
        for c in chunks:
            idx_list.append(len(chunk_texts))
            chunk_texts.append(c)
        note_to_chunk_rows.append(idx_list)
    q_vecs = bi_enc.encode(
        chunk_texts, batch_size=bi_batch_size,
        max_length_override=bi_query_max_length,
    )

    # Recall buckets to compute.
    K_BUCKETS = [10, 50, 100, 200, 500]
    recall_hits = {f"bm25_recall@{k}": 0 for k in K_BUCKETS}
    recall_hits.update({f"dense_recall@{k}": 0 for k in K_BUCKETS})
    recall_hits.update({f"union_recall@{k}": 0 for k in K_BUCKETS})
    n_gold_total = 0
    n_gold_in_kb = 0
    n_gold_missing_anywhere = 0  # gold codes absent from top-500 union

    audit_rows: List[Dict] = []
    worst: List[Tuple[str, Set[str], Dict[str, Optional[int]]]] = []

    bm25_codes_set = set(bm25_codes)

    note_ids = df["note_id"].astype(str).tolist()
    gold_raw = df[gold_code_col].tolist()
    for i, note_id in enumerate(
        tqdm(note_ids, total=len(df), desc="kb_index diagnose")
    ):
        chunk_rows = note_to_chunk_rows[i]
        chunks_for_note = [chunk_texts[r] for r in chunk_rows]

        # Gold parse — stdlib-only NaN check so this path doesn't depend
        # on the pandas import being in scope (a partial sync of this
        # file used to break diagnose with `NameError: pd`).
        raw_cell = gold_raw[i]
        is_missing = (
            raw_cell is None
            or (isinstance(raw_cell, float) and raw_cell != raw_cell)  # NaN
        )
        raw_gold = "" if is_missing else str(raw_cell)
        gold = {c.strip().upper() for c in raw_gold.split("|") if c.strip()}
        if not gold:
            continue
        n_gold_total += len(gold)
        n_gold_in_kb += sum(1 for c in gold if c in bm25_codes_set)

        # BM25 retrieval per chunk → merge by best (lowest) rank seen
        # for each KB row.  Chunks see different vocab-frequency
        # signals, so a code mentioned only in one section gets its
        # natural top-rank in that chunk's BM25 even if it'd be lost
        # in a full-note query.
        bm25_best_rank: Dict[int, int] = {}
        for chunk in chunks_for_note:
            scores = bm25.get_scores(tokenize(chunk))
            this_order = np.argsort(-scores)[:bm25_top_k]
            for r, j in enumerate(this_order):
                jj = int(j)
                if jj not in bm25_best_rank or r < bm25_best_rank[jj]:
                    bm25_best_rank[jj] = r
        bm25_full_order = [j for j, _ in sorted(
            bm25_best_rank.items(), key=lambda kv: kv[1]
        )]

        # Dense retrieval per chunk → merge by best (lowest) rank.
        dense_best_rank: Dict[int, int] = {}
        for r_idx in chunk_rows:
            dense_idx_row, _ = _topk_dense_cosine(
                q_vecs[r_idx : r_idx + 1], dense_vecs, top_k=dense_top_k
            )
            for r, j in enumerate(dense_idx_row[0]):
                jj = int(j)
                if jj not in dense_best_rank or r < dense_best_rank[jj]:
                    dense_best_rank[jj] = r
        dense_full_order = [j for j, _ in sorted(
            dense_best_rank.items(), key=lambda kv: kv[1]
        )]

        # Per-bucket hit checks
        bm25_codes_at_k = {k: {bm25_codes[int(j)] for j in bm25_full_order[:k]}
                           for k in K_BUCKETS}
        dense_codes_at_k = {k: {bm25_codes[int(j)] for j in dense_full_order[:k]}
                            for k in K_BUCKETS}
        union_codes_at_k = {k: bm25_codes_at_k[k] | dense_codes_at_k[k]
                            for k in K_BUCKETS}
        for k in K_BUCKETS:
            recall_hits[f"bm25_recall@{k}"] += sum(1 for c in gold if c in bm25_codes_at_k[k])
            recall_hits[f"dense_recall@{k}"] += sum(1 for c in gold if c in dense_codes_at_k[k])
            recall_hits[f"union_recall@{k}"] += sum(1 for c in gold if c in union_codes_at_k[k])

        # Per-gold-code ranks (None if absent from top-500)
        bm25_ranks = _ranks_of(gold, bm25_full_order, bm25_codes)
        dense_ranks = _ranks_of(gold, dense_full_order, bm25_codes)
        # Union order: simple round-robin merge by descending position
        union_seen: Dict[int, int] = {}
        for r, j in enumerate(bm25_full_order):
            union_seen.setdefault(int(j), r)
        for r, j in enumerate(dense_full_order):
            if int(j) not in union_seen:
                union_seen[int(j)] = r
        union_order = [j for j, _ in sorted(union_seen.items(), key=lambda kv: kv[1])]
        union_ranks = _ranks_of(gold, union_order, bm25_codes)

        n_missing_this_note = sum(1 for c in gold if union_ranks[c] is None)
        n_gold_missing_anywhere += n_missing_this_note

        if audit_csv is not None:
            for c in sorted(gold):
                audit_rows.append({
                    "note_id": note_id,
                    "gold_code": c,
                    "in_kb": c in bm25_codes_set,
                    "bm25_rank": bm25_ranks[c],
                    "dense_rank": dense_ranks[c],
                    "union_rank": union_ranks[c],
                })

        # Track 10 worst-case notes (most gold codes missing from union top-500).
        if n_missing_this_note >= max(1, len(gold) // 2):
            worst.append((note_id, gold, union_ranks))

    # Aggregate
    summary = {k: round(v / max(1, n_gold_total), 4) for k, v in recall_hits.items()}
    summary.update({
        "n_notes_evaluated": int((df[gold_code_col].fillna("").astype(str).str.strip() != "").sum()),
        "n_gold_codes_total": n_gold_total,
        "n_gold_in_kb": n_gold_in_kb,
        "pct_gold_in_kb": round(100.0 * n_gold_in_kb / max(1, n_gold_total), 2),
        "n_gold_missing_top500_union": n_gold_missing_anywhere,
        "pct_gold_missing_top500_union": round(
            100.0 * n_gold_missing_anywhere / max(1, n_gold_total), 2
        ),
        "bm25_top_k_searched": bm25_top_k,
        "dense_top_k_searched": dense_top_k,
        "max_note_tokens": max_note_tokens,
    })

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("kb_index diagnose: wrote %s", out_json)

    if audit_csv is not None:
        audit_csv = Path(audit_csv)
        audit_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)
        LOGGER.info("kb_index diagnose: wrote per-gold-code audit -> %s", audit_csv)

    # Pretty console summary.
    LOGGER.info("=" * 60)
    LOGGER.info("kb_index diagnose summary  (n_gold=%d, %.1f%% are in KB)",
                n_gold_total, summary["pct_gold_in_kb"])
    LOGGER.info("-" * 60)
    LOGGER.info("%-18s %s", "k", "  ".join(f"@{k:>3d}" for k in K_BUCKETS))
    for src in ("bm25", "dense", "union"):
        cells = "  ".join(f"{summary[f'{src}_recall@{k}']*100:5.1f}" for k in K_BUCKETS)
        LOGGER.info("%-18s %s", f"{src} recall (%)", cells)
    LOGGER.info("-" * 60)
    LOGGER.info("%d/%d (%.1f%%) gold codes are absent from the top-500 union",
                n_gold_missing_anywhere, n_gold_total,
                summary["pct_gold_missing_top500_union"])
    if summary["union_recall@100"] < 0.30:
        LOGGER.warning(
            "VERDICT: union_recall@100 = %.1f%% — retrieval is broken. "
            "A section-aware query and a richer KB index are the "
            "first things to try.",
            summary["union_recall@100"] * 100,
        )
    elif summary["union_recall@100"] >= 0.50:
        LOGGER.info(
            "VERDICT: union_recall@100 = %.1f%% — retrieval is healthy. "
            "If F1 is still low, the ranking step is the bottleneck — "
            "tune fusion weights or per-code thresholds.",
            summary["union_recall@100"] * 100,
        )
    else:
        LOGGER.info(
            "VERDICT: union_recall@100 = %.1f%% — borderline.  Try "
            "a section-aware query AND increase --bm25-top-k / "
            "--dense-top-k.",
            summary["union_recall@100"] * 100,
        )

    if worst:
        LOGGER.info("-" * 60)
        LOGGER.info("Worst-case notes (≥half of gold codes outside top-500):")
        for note_id, gold, ranks in worst[:10]:
            ranks_str = ", ".join(
                f"{c}=({'<500' if ranks[c] is None else ranks[c]})"
                for c in sorted(gold)
            )
            LOGGER.info("  %s  gold ranks in union: %s", note_id, ranks_str)
    LOGGER.info("=" * 60)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_local_models_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--local-models-dir", type=Path, default=None,
                   help="Directory holding offline copies of the HF "
                        "models (e.g. ./models/).  Falls back to "
                        "$CPT_REC_LOCAL_MODELS_DIR or ./models/.  Set "
                        "CPT_REC_REQUIRE_LOCAL_MODELS=1 to fail when a "
                        "local copy is missing.")


def _build_index_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("build-index", help="Build BM25+dense KB indexes.")
    p.add_argument("--kb", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--biencoder", default=DEFAULT_BIENCODER)
    p.add_argument("--bi-max-length", type=int, default=64)
    p.add_argument("--bi-batch-size", type=int, default=128)
    p.add_argument("--kb-text-fields", default="code_description",
                   help="Comma-separated KB columns + synthetic fields "
                        "to concatenate into each code's indexed text. "
                        "Empty values per row are skipped, so codes "
                        "with shallower hierarchies still get clean "
                        "text.  Synthetic fields available:\n"
                        "  code_range_deepest — most specific non-empty "
                        "code_range_{1..6}_description for that code;\n"
                        "  code_range_path    — breadcrumb of every "
                        "non-empty range description, joined with ' > '. \n"
                        "Recommended richer setting: "
                        "'code_description,code_lay_term,code_range_deepest'. "
                        "If you change this, you must rebuild the index.")
    _add_local_models_arg(p)


def _build_diagnose_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("diagnose",
                      help="Run candidate retrieval against a labeled "
                           "CSV and report recall + gold-rank distribution. "
                           "No reranker is loaded.  Use this BEFORE "
                           "tuning thresholds — if union_recall@100 is "
                           "already low, no rerank/fusion can help.")
    p.add_argument("--notes", required=True, type=Path,
                   help="Labeled CSV (val split) with proc_codes column.")
    p.add_argument("--index-dir", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path,
                   help="Where to write the aggregate summary.")
    p.add_argument("--audit-csv", type=Path, default=None,
                   help="Optional per-gold-code audit CSV "
                        "(note_id, gold_code, ranks in BM25/dense/union).")
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--bm25-top-k", type=int, default=500)
    p.add_argument("--dense-top-k", type=int, default=500)
    p.add_argument("--query-chunks", type=int, default=1,
                   help="Split each note into N word-equal chunks and "
                        "issue separate BM25+dense queries per chunk, "
                        "then merge by best-rank.  N=1 is the default "
                        "(today's behavior).  N=4 typically catches "
                        "codes that are mentioned only in one section "
                        "of a multi-procedure note — measured directly "
                        "by re-running diagnose with this flag.")
    p.add_argument("--biencoder", default=None)
    p.add_argument("--bi-max-length", type=int, default=None)
    p.add_argument("--bi-query-max-length", type=int, default=256)
    p.add_argument("--bi-batch-size", type=int, default=64)
    p.add_argument("--max-note-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)
    _add_local_models_arg(p)


def _build_predict_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("predict", help="Two-stage rerank predictions.")
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--kb", type=Path, default=None,
                   help="Optional KB CSV; required only for legacy "
                        "(format_version < 2) indexes that don't embed "
                        "their descriptions.")
    p.add_argument("--index-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bm25-top-k", type=int, default=50)
    p.add_argument("--dense-top-k", type=int, default=50)
    p.add_argument("--reranker", default=DEFAULT_RERANKER)
    p.add_argument("--rerank-max-length", type=int, default=384,
                   help="Token budget for the PACKED PAIR "
                        "`[CLS] note [SEP] description [SEP]` -- not per side "
                        "and not a pair count. Hard-capped by the encoder's "
                        "position embeddings (512 for every BERT-family "
                        "reranker), so it cannot be set from the note-length "
                        "distribution. Use --rerank-chunks to cover a long "
                        "note.")
    p.add_argument("--rerank-chunks", type=int, default=1,
                   help="Split the note into N pieces, score every piece "
                        "against every candidate, and keep each candidate's "
                        "BEST piece. N=1 (default) is byte-identical to the "
                        "historical single-window path; N>1 costs N x the "
                        "cross-encoder forward passes and is the only way to "
                        "see past the 512-token ceiling.")
    p.add_argument("--rerank-batch-size", type=int, default=32)
    p.add_argument("--rerank-threshold", type=float, default=0.05,
                   help="Sigmoid-normalized [0,1] reranker score "
                        "threshold (cross-encoder logits are sigmoided "
                        "in CrossEncoder.score).  MS-MARCO-style "
                        "rerankers produce low absolute scores on "
                        "out-of-distribution medical-coding pairs; "
                        "0.05 is a safe permissive default, retune from "
                        "the per-note score-distribution log line.")
    p.add_argument("--rerank-min-keep", type=int, default=5,
                   help="If thresholding emits 0 codes for a note, fall "
                        "back to the top-N highest-scoring candidates "
                        "regardless of threshold.  Prevents the "
                        "all-empty failure mode.  Set 0 to disable.  "
                        "Ignored when --rerank-mode=fusion.")
    p.add_argument("--rerank-mode", default="cross",
                   choices=["cross", "fusion"],
                   help="'cross': BM25 ∪ dense → cross-encoder rerank "
                        "(default; best when the reranker is in-domain). "
                        "'fusion': skip the cross-encoder and rank by "
                        "Reciprocal Rank Fusion of BM25 + dense — "
                        "recommended when the reranker is OOD on medical "
                        "text and produces near-zero sigmoid scores.")
    p.add_argument("--rrf-k", type=int, default=60,
                   help="RRF smoothing constant for --rerank-mode=fusion "
                        "(60 is the canonical TREC default).")
    p.add_argument("--query-chunks", type=int, default=1,
                   help="Split each note into N word-equal chunks and "
                        "issue separate BM25+dense queries per chunk, "
                        "merging by best-rank.  Catches codes mentioned "
                        "only in one section of a multi-procedure note. "
                        "N=1 = today's behavior; N=4 typically lifts "
                        "recall@100 by 10-20pp on heterogeneous notes.")
    p.add_argument("--max-keep", type=int, default=None)
    p.add_argument("--biencoder", default=None,
                   help="Override stored biencoder.")
    p.add_argument("--bi-max-length", type=int, default=None,
                   help="KB-side sequence budget (defaults to the value "
                        "stored in manifest.json — overriding it would "
                        "invalidate the cached dense vectors).")
    p.add_argument("--bi-query-max-length", type=int, default=256,
                   help="Query-side sequence budget — independent of the "
                        "KB-side cap, since op-note excerpts are much "
                        "longer than KB code descriptions.")
    p.add_argument("--bi-batch-size", type=int, default=64)
    p.add_argument("--max-note-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--history-changes", type=Path, default=None,
                   help="Optional code_changes.csv for date-aware "
                        "candidate-pool restriction; pair with --history-"
                        "deleted.")
    p.add_argument("--history-deleted", type=Path, default=None)
    p.add_argument("--as-of-col", default="PROCEDURE_DATE",
                   help="Date column on the test CSV used to restrict "
                        "the candidate pool to codes active on that date.")
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)
    p.add_argument("--dump-scores-npz", type=Path, default=None,
                   help="Also write the FULL reranked candidate union to "
                        "this NPZ (before --rerank-threshold / --max-keep), "
                        "for `cptrec-evaluate --scores-npz`.  Off by default; "
                        "the predictions CSV is unchanged either way.")
    _add_local_models_arg(p)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="kb_index: BM25+semantic candidate gen → cross-encoder rerank."
    )
    sp = parser.add_subparsers(dest="cmd", required=True)
    _build_index_parser(sp)
    _build_predict_parser(sp)
    _build_diagnose_parser(sp)
    args = parser.parse_args()

    if args.cmd == "build-index":
        kb_text_fields = tuple(
            f.strip() for f in args.kb_text_fields.split(",") if f.strip()
        ) or _DEFAULT_KB_TEXT_FIELDS
        build_index(
            kb_csv=args.kb,
            out_dir=args.out_dir,
            biencoder=args.biencoder,
            bi_max_length=args.bi_max_length,
            bi_batch_size=args.bi_batch_size,
            kb_text_fields=kb_text_fields,
            local_models_dir=args.local_models_dir,
        )
    elif args.cmd == "predict":
        predict_b3(
            notes_csv=args.notes,
            kb_csv=args.kb,
            index_dir=args.index_dir,
            out_csv=args.out,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            reranker=args.reranker,
            rerank_max_length=args.rerank_max_length,
            rerank_batch_size=args.rerank_batch_size,
            rerank_threshold=args.rerank_threshold,
            max_keep=args.max_keep,
            biencoder=args.biencoder,
            bi_max_length=args.bi_max_length,
            bi_query_max_length=args.bi_query_max_length,
            bi_batch_size=args.bi_batch_size,
            max_note_tokens=args.max_note_tokens,
            rerank_min_keep=args.rerank_min_keep,
            rerank_chunks=args.rerank_chunks,
            rerank_mode=args.rerank_mode,
            rrf_k=args.rrf_k,
            query_chunks=args.query_chunks,
            seed=args.seed,
            limit=args.limit,
            history_changes=args.history_changes,
            history_deleted=args.history_deleted,
            as_of_col=args.as_of_col,
            local_models_dir=args.local_models_dir,
            note_id_col=args.note_id_col,
            note_text_col=args.note_text_col,
            dump_scores_npz=args.dump_scores_npz,
        )
    elif args.cmd == "diagnose":
        diagnose_b3(
            notes_csv=args.notes,
            index_dir=args.index_dir,
            out_json=args.out_json,
            audit_csv=args.audit_csv,
            gold_code_col=args.gold_code_col,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            query_chunks=args.query_chunks,
            biencoder=args.biencoder,
            bi_max_length=args.bi_max_length,
            bi_query_max_length=args.bi_query_max_length,
            bi_batch_size=args.bi_batch_size,
            max_note_tokens=args.max_note_tokens,
            seed=args.seed,
            limit=args.limit,
            local_models_dir=args.local_models_dir,
            note_id_col=args.note_id_col,
            note_text_col=args.note_text_col,
        )


if __name__ == "__main__":
    main()
