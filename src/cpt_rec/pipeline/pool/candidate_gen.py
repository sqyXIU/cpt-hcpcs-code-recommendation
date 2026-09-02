# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Per-snippet candidate code generation.

For each snippet we want a **compact** pool (~20-40 codes) of plausibly
relevant CPT/HCPCS codes so the scorer only spends compute on candidates
with a realistic chance of being correct.

Sources
-------

1. **BM25** over code descriptions (high lexical recall on things like
   "colonoscopy", "Achilles tendon", specific device names).
2. **Dense bi-encoder** cosine similarity over code descriptions (picks up
   paraphrases and synonyms that BM25 misses).
3. **Union** of the two rankings.  We keep each source's top-k and take
   the set union — simple and strong.

We reuse the KB indexes produced by :mod:`cpt_rec.baselines.kb_index`'s
``build-index`` subcommand, so no new index build is required when
``cptrec-build-kb-index build-index`` has already been run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cpt_rec.baselines.kb_index import BiEncoder
from cpt_rec.baselines.bm25_index import (
    _bm25_doc_weight_matrix,
    _build_bm25,
    tokenize,
)

LOGGER = logging.getLogger(__name__)


class KBCandidateIndex:
    """
    Thin wrapper around a ``cptrec-build-kb-index build-index`` artifact dir.

    Exposes ``bm25_topk`` and ``dense_topk`` over KB codes, plus a
    convenience ``topk_union`` that merges both.
    """

    def __init__(
        self,
        index_dir: Path | str,
        bi_encoder: Optional[BiEncoder] = None,
        bi_model_override: Optional[str] = None,
        bi_max_length: Optional[int] = None,
    ) -> None:
        index_dir = Path(index_dir)
        with open(index_dir / "manifest.json") as f:
            manifest = json.load(f)
        self.manifest = manifest

        bm25_dat = np.load(index_dir / "bm25_corpus.npz", allow_pickle=True)
        self.codes: List[str] = [str(c) for c in bm25_dat["codes"]]
        self._bm25_corpus: List[List[str]] = [list(d) for d in bm25_dat["corpus"]]
        self._bm25 = _build_bm25(self._bm25_corpus)

        dense_dat = np.load(index_dir / "dense.npz", allow_pickle=True)
        dense_codes = [str(c) for c in dense_dat["codes"]]
        if dense_codes != self.codes:
            raise ValueError(
                "BM25 and dense indexes have inconsistent code orderings; "
                "re-run `cptrec-build-kb-index build-index`."
            )
        self.dense_vecs: np.ndarray = dense_dat["vectors"].astype(np.float32)

        # Code -> row index. Lets the candidate-pool assembler map an
        # arbitrary candidate code (from neighbor / LLM sources, not just KB
        # retrieval) back to its dense vector / KB row in O(1).
        self.code_to_row: dict = {c: i for i, c in enumerate(self.codes)}

        if bi_encoder is None:
            self.bi_encoder = BiEncoder(
                model_name=bi_model_override or manifest["biencoder"],
                max_length=bi_max_length or int(manifest.get("bi_max_length", 64)),
            )
        else:
            self.bi_encoder = bi_encoder

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_b3_artifact(
        cls,
        index_dir: Path | str,
        bi_encoder: Optional[BiEncoder] = None,
        bi_model_override: Optional[str] = None,
        bi_max_length: Optional[int] = None,
    ) -> "KBCandidateIndex":
        """
        Construct a :class:`KBCandidateIndex` directly from the artifact
        directory produced by ``cptrec-build-kb-index build-index``.

        This is the documented entry-point, preferred over calling ``__init__`` for callers that want a
        symbolic, intent-revealing constructor.  The two are equivalent
        today, but keeping a separate factory leaves room for future
        artifact-format upgrades (e.g. quantized dense vectors, ANN
        indexes) without breaking call sites.
        """
        return cls(
            index_dir=index_dir,
            bi_encoder=bi_encoder,
            bi_model_override=bi_model_override,
            bi_max_length=bi_max_length,
        )

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------
    def bm25_topk(self, query_text: str, top_k: int) -> List[int]:
        scores = self._bm25.get_scores(tokenize(query_text))
        if top_k < len(scores):
            idx = np.argpartition(-scores, top_k - 1)[:top_k]
            idx = idx[np.argsort(-scores[idx])]
        else:
            idx = np.argsort(-scores)
        return [int(j) for j in idx]

    def bm25_topk_scored(
        self, query_text: str, top_k: int
    ) -> List[Tuple[int, float]]:
        """Like :meth:`bm25_topk` but also returns the raw BM25 score.

        Used by the candidate-pool assembler to record per-source
        provenance (rank + score) in the φ(n, c) feature vector.
        """
        scores = self._bm25.get_scores(tokenize(query_text))
        if top_k < len(scores):
            idx = np.argpartition(-scores, top_k - 1)[:top_k]
            idx = idx[np.argsort(-scores[idx])]
        else:
            idx = np.argsort(-scores)
        return [(int(j), float(scores[j])) for j in idx]

    def _ensure_bm25_matrix(self):
        """Lazily build + cache the sparse KB document-weight matrix.

        ``rank_bm25.get_scores`` is a pure-Python loop over all KB docs *per
        query token*; the candidate-pool assembler calls it once per snippet
        (up to ``max_snippets`` per note), which dominated pool-assembly time
        (~5 s/note → ~30 h over 19 k notes).  Precompute the query-independent
        weight matrix once so each note's snippets reduce to a single sparse
        mat-mul (see :func:`bm25_topk_scored_batch`).  Cached on the instance;
        ``(None, None)`` when the sparse build is unavailable (no SciPy /
        missing BM25 internals) so callers fall back to the per-query path.
        """
        # NB: ``_bm25_Wt`` holds a SciPy CSR matrix once built, so we must test
        # *presence* with ``hasattr`` rather than an equality sentinel — comparing
        # a sparse matrix against a string triggers ``__ne__`` -> ``np.isnan`` and
        # raises ``TypeError``.
        if not hasattr(self, "_bm25_Wt"):
            built = _bm25_doc_weight_matrix(self._bm25)
            if built is None:
                LOGGER.warning(
                    "KBCandidateIndex: vectorized BM25 unavailable; using slow "
                    "per-query loop."
                )
                self._bm25_Wt = None
                self._bm25_term_to_col = None
            else:
                W, term_to_col = built
                self._bm25_Wt = W.transpose().tocsr()  # (vocab × n_codes)
                self._bm25_term_to_col = term_to_col
        return self._bm25_Wt, self._bm25_term_to_col

    def bm25_topk_scored_batch(
        self, query_texts: Sequence[str], top_k: int, query_chunk: int = 512
    ) -> List[List[Tuple[int, float]]]:
        """Vectorized :meth:`bm25_topk_scored` for many queries at once.

        Scores each chunk of queries against the precomputed KB weight matrix
        with a single sparse mat-mul.  Output is aligned to ``query_texts`` and
        each row is identical to calling :meth:`bm25_topk_scored` per query
        (``Counter`` reproduces query-term multiplicity, so ``Q @ Wᵀ`` equals
        ``get_scores`` exactly).  Falls back to the per-query path when the
        sparse build is unavailable.
        """
        n = len(query_texts)
        if n == 0:
            return []
        Wt, term_to_col = self._ensure_bm25_matrix()
        if Wt is None:
            return [self.bm25_topk_scored(q, top_k) for q in query_texts]

        from collections import Counter

        from scipy.sparse import csr_matrix

        vocab = len(term_to_col)
        n_docs = Wt.shape[1]
        eff_k = min(top_k, n_docs)

        out: List[List[Tuple[int, float]]] = []
        for start in range(0, n, query_chunk):
            chunk = query_texts[start : start + query_chunk]
            rows: List[int] = []
            cols: List[int] = []
            data: List[float] = []
            for r, q in enumerate(chunk):
                for term, cnt in Counter(tokenize(str(q))).items():
                    col = term_to_col.get(term)
                    if col is not None:
                        rows.append(r)
                        cols.append(col)
                        data.append(float(cnt))
            Q = csr_matrix(
                (
                    np.asarray(data, dtype=np.float32),
                    (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
                ),
                shape=(len(chunk), vocab),
            )
            scores = (Q @ Wt).toarray()  # (chunk × n_codes), dense
            for r in range(scores.shape[0]):
                sc = scores[r]
                if eff_k < n_docs:
                    idx = np.argpartition(-sc, eff_k - 1)[:eff_k]
                    idx = idx[np.argsort(-sc[idx])]
                else:
                    idx = np.argsort(-sc)
                out.append([(int(j), float(sc[j])) for j in idx[:eff_k]])
        return out

    def encode_queries(
        self, queries: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        """L2-normalized snippet/query embeddings in the dense KB space.

        Thin pass-through to the wrapped bi-encoder so the assembler can
        reuse one snippet encode for both dense retrieval and per-candidate
        best-evidence-snippet selection (``snip_embs @ dense_vecs.T``).
        """
        if not queries:
            return np.zeros((0, self.dense_vecs.shape[1]), dtype=np.float32)
        return self.bi_encoder.encode(
            list(queries), batch_size=batch_size, show_progress=False
        )

    def dense_topk_batch(
        self, queries: Sequence[str], top_k: int, batch_size: int = 64
    ) -> List[List[int]]:
        import time as _time
        _t0 = _time.time()
        LOGGER.info(
            "dense_topk_batch: encoding %d queries (max_len=%s, batch=%d, device=%s)",
            len(queries), self.bi_encoder.max_length, batch_size, self.bi_encoder.device,
        )
        vecs = self.bi_encoder.encode(
            list(queries), batch_size=batch_size, show_progress=False
        )  # (n_q, d)
        LOGGER.info(
            "dense_topk_batch: encode done in %.2fs, vecs.shape=%s",
            _time.time() - _t0, vecs.shape,
        )
        if vecs.size == 0:
            return [[] for _ in queries]
        _t1 = _time.time()
        sims = vecs @ self.dense_vecs.T  # (n_q, n_codes)
        if top_k < sims.shape[1]:
            part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
            rows = np.arange(sims.shape[0])[:, None]
            ps = sims[rows, part]
            order = np.argsort(-ps, axis=1)
            idx = part[rows, order]
        else:
            idx = np.argsort(-sims, axis=1)[:, :top_k]
        LOGGER.info("dense_topk_batch: rank done in %.2fs", _time.time() - _t1)
        return [[int(j) for j in row] for row in idx]

    def topk_union_batch(
        self,
        queries: Sequence[str],
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        dense_batch_size: int = 64,
    ) -> List[List[str]]:
        """
        Return the union of BM25 and dense top-k codes for each query.
        Ordering of returned codes is BM25 first (to keep lexical hits at
        the top), then dense-only codes in dense rank order.
        """
        import time as _time
        LOGGER.info(
            "topk_union_batch: %d queries (bm25_k=%d dense_k=%d)",
            len(queries), bm25_top_k, dense_top_k,
        )
        _t_dense = _time.time()
        dense_idx = self.dense_topk_batch(queries, dense_top_k, dense_batch_size)
        LOGGER.info(
            "topk_union_batch: dense block total %.2fs; starting BM25 loop",
            _time.time() - _t_dense,
        )
        _t_bm = _time.time()
        out: List[List[str]] = []
        for q, d_row in zip(queries, dense_idx):
            bm25_row = self.bm25_topk(q, bm25_top_k)
            seen: set = set()
            merged: List[int] = []
            for j in bm25_row:
                if j not in seen:
                    seen.add(j)
                    merged.append(j)
            for j in d_row:
                if j not in seen:
                    seen.add(j)
                    merged.append(j)
            out.append([self.codes[j] for j in merged])
        LOGGER.info(
            "topk_union_batch: BM25 loop %.2fs over %d queries",
            _time.time() - _t_bm, len(queries),
        )
        return out


def merge_snippet_candidates(
    per_snippet: Sequence[Sequence[str]],
    max_candidates: Optional[int] = None,
) -> List[str]:
    """
    Merge per-snippet candidate lists into a single note-level candidate
    pool.  Preserves order of first appearance and deduplicates.
    """
    seen: set = set()
    out: List[str] = []
    for row in per_snippet:
        for c in row:
            if c not in seen:
                seen.add(c)
                out.append(c)
                if max_candidates is not None and len(out) >= max_candidates:
                    return out
    return out
