# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
BM25 index over training operative notes (used by M1 and M4).

The index stores the BM25-tokenized corpus, the original ``note_id`` for
each row, and the gold ``proc_codes`` set for each row.  At query time,
we tokenize the test note with the same tokenizer and return the top-k
training neighbors with their similarity scores AND code sets.

Implementation
--------------
Uses ``rank_bm25.BM25Okapi`` if installed (default), with a hand-rolled
fallback if not — keeps the dep optional but encourages installing it
for performance.

Tokenization
------------
Lowercase + ``\\b\\w+\\b`` token regex.  Keeps numerics so that codes /
sizes / years (e.g. ``8mm``, ``45378``) survive — for M1 we don't want
to strip these, and they help BM25 distinguish similar procedures.
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase + word-token split."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Pure-python BM25 fallback (used when rank_bm25 is unavailable)
# ---------------------------------------------------------------------------

class _BM25Okapi:
    """Minimal BM25Okapi implementation; vectorizes scoring per-query."""

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)

        # Term frequencies per doc + document frequency
        self.doc_term_freqs: List[Dict[str, int]] = []
        df: Dict[str, int] = defaultdict(int)
        for doc in corpus:
            counts = Counter(doc)
            self.doc_term_freqs.append(counts)
            for term in counts:
                df[term] += 1

        # IDF using BM25 formula with smoothing
        self.idf: Dict[str, float] = {}
        for term, n in df.items():
            self.idf[term] = math.log(
                1.0 + (self.corpus_size - n + 0.5) / (n + 0.5)
            )

    def get_scores(self, query_tokens: Sequence[str]):
        import numpy as np

        scores = np.zeros(self.corpus_size, dtype=np.float64)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


def _build_bm25(corpus: Sequence[Sequence[str]]):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(corpus)
    except ImportError:
        LOGGER.warning(
            "rank_bm25 not installed; using slow pure-python fallback. "
            "`pip install rank_bm25` for ~10x speed-up."
        )
        return _BM25Okapi(corpus)


def _bm25_doc_weight_matrix(bm25):
    """Build a sparse document-weight matrix ``W`` (n_docs × vocab) such that
    the BM25 score of a query equals ``W @ qtf`` exactly, where ``qtf`` is the
    query's per-term count vector.

    BM25Okapi's per-document term weight
    ``idf(t) · tf(d,t)(k1+1) / (tf(d,t) + k1(1−b+b·dl_d/avgdl))`` does not depend
    on the query, so it can be precomputed once and every neighbor query reduces
    to a single sparse mat-vec — replacing ``rank_bm25``'s Python loop over
    ``len(query) × n_docs`` (the ~4 s/query, ~21 h-over-19k-notes bottleneck).

    Returns ``(W_csr, term_to_col)`` or ``None`` when SciPy is unavailable or
    the BM25 object doesn't expose the needed internals (then callers fall back
    to the per-query path).  Works for both ``rank_bm25.BM25Okapi`` and the
    pure-python :class:`_BM25Okapi` fallback (attribute names differ).
    """
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
    except ImportError:
        return None

    doc_freqs = getattr(bm25, "doc_freqs", None)
    if doc_freqs is None:
        doc_freqs = getattr(bm25, "doc_term_freqs", None)
    idf = getattr(bm25, "idf", None)
    doc_len = getattr(bm25, "doc_len", None)
    if doc_len is None:
        doc_len = getattr(bm25, "doc_lens", None)
    avgdl = getattr(bm25, "avgdl", None)
    k1 = float(getattr(bm25, "k1", 1.5))
    b = float(getattr(bm25, "b", 0.75))
    if not doc_freqs or idf is None or doc_len is None or not avgdl:
        return None

    avgdl = float(avgdl)
    n_docs = len(doc_freqs)
    term_to_col: Dict[str, int] = {t: i for i, t in enumerate(idf.keys())}

    nnz = sum(len(f) for f in doc_freqs)
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    p = 0
    for d, freqs in enumerate(doc_freqs):
        norm = k1 * (1.0 - b + b * (doc_len[d] / avgdl))
        for term, tf in freqs.items():
            col = term_to_col.get(term)
            if col is None:
                continue
            w = idf.get(term, 0.0) * (tf * (k1 + 1.0)) / (tf + norm)
            if w != 0.0:
                rows[p] = d
                cols[p] = col
                data[p] = w
                p += 1
    W = csr_matrix(
        (data[:p], (rows[:p], cols[:p])),
        shape=(n_docs, len(term_to_col)),
    )
    return W, term_to_col


# ---------------------------------------------------------------------------
# Train-note index
# ---------------------------------------------------------------------------

class TrainNoteBM25Index:
    """
    Holds tokenized training corpus + per-note metadata and supports
    top-k neighbor lookup.

    Parameters
    ----------
    note_ids : sequence of str
    code_sets : sequence of set[str]   — gold codes per note
    corpus    : sequence of list[str]  — tokenized note texts
    """

    def __init__(
        self,
        note_ids: Sequence[str],
        code_sets: Sequence[Set[str]],
        corpus: Sequence[Sequence[str]],
        raw_texts: Optional[Sequence[str]] = None,
        gold_code_col: str = "proc_codes",
    ) -> None:
        if not (len(note_ids) == len(code_sets) == len(corpus)):
            raise ValueError("note_ids, code_sets, corpus must have same length")
        if raw_texts is not None and len(raw_texts) != len(note_ids):
            raise ValueError("raw_texts must have the same length as note_ids")
        self.note_ids: List[str] = list(note_ids)
        self.code_sets: List[Set[str]] = [set(s) for s in code_sets]
        self.corpus: List[List[str]] = [list(t) for t in corpus]
        self.raw_texts: Optional[List[str]] = (
            list(raw_texts) if raw_texts is not None else None
        )
        self.gold_code_col: str = gold_code_col
        self.bm25 = _build_bm25(self.corpus)
        LOGGER.info(
            "BM25 train-note index built: %d docs, avgdl=%.1f",
            len(self.corpus),
            sum(len(d) for d in self.corpus) / max(1, len(self.corpus)),
        )

    @classmethod
    def from_csv(
        cls,
        train_csv: Path | str,
        note_id_col: Optional[str] = None,
        note_text_col: Optional[str] = None,
        gold_code_col: str = "proc_codes",
        keep_raw_text: bool = True,
    ) -> "TrainNoteBM25Index":
        """
        Build the index from a training CSV.

        Each gold ``proc_codes`` cell is parsed and passed through the
        canonical procedure-code shape filter
        (``preprocess.code_utils.is_valid_proc_code``) — same gate the
        leakage scrubber uses — so a stray modifier or prose token in the
        labels can't pollute M1's vote vocabulary or M4's candidate union.

        When ``keep_raw_text=True`` (default), the original (untokenized)
        ``note_text`` strings are kept on the index so M4 can render them
        back into the LLM prompt without reconstructing from a lossy
        bag-of-words.  Set False to save memory if the index is for M1
        only.
        """
        from cpt_rec.baselines.common import (
            load_notes_for_prediction,
        )
        from cpt_rec.common.preprocess.code_utils import (
            is_valid_proc_code,
        )

        df = load_notes_for_prediction(
            train_csv, note_id_col=note_id_col, note_text_col=note_text_col
        )
        if gold_code_col not in df.columns:
            raise ValueError(
                f"Train CSV missing gold column '{gold_code_col}'."
            )

        note_ids = df["note_id"].astype(str).tolist()
        code_sets: List[Set[str]] = []
        n_dropped_total = 0
        dropped_examples: Set[str] = set()
        for s in df[gold_code_col].fillna(""):
            raw = {c.strip().upper() for c in str(s).split("|") if c.strip()}
            kept = {c for c in raw if is_valid_proc_code(c)}
            dropped = raw - kept
            if dropped:
                n_dropped_total += len(dropped)
                if len(dropped_examples) < 10:
                    dropped_examples.update(list(dropped)[: 10 - len(dropped_examples)])
            code_sets.append(kept)
        if n_dropped_total:
            LOGGER.info(
                "TrainNoteBM25Index: shape filter dropped %d non-procedure-code "
                "token(s) from gold sets (examples: %s)",
                n_dropped_total,
                sorted(dropped_examples),
            )
        raw_texts = df["note_text"].astype(str).tolist()
        corpus = [tokenize(t) for t in raw_texts]
        idx = cls(note_ids, code_sets, corpus)
        if keep_raw_text:
            idx.raw_texts = raw_texts
        idx.gold_code_col = gold_code_col
        return idx

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def topk(
        self,
        query_text: str,
        top_k: int = 25,
    ) -> List[Tuple[int, float]]:
        """Return the top-k ``(corpus_index, bm25_score)`` for the query."""
        import numpy as np

        toks = tokenize(query_text)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        if top_k < len(scores):
            idx = np.argpartition(-scores, top_k - 1)[:top_k]
            idx = idx[np.argsort(-scores[idx])]
        else:
            idx = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in idx[:top_k]]

    def neighbors(
        self,
        query_text: str,
        top_k: int = 25,
    ) -> List[Tuple[str, float, Set[str]]]:
        """Return ``(neighbor_note_id, score, neighbor_code_set)`` triples."""
        return [
            (self.note_ids[i], score, self.code_sets[i])
            for i, score in self.topk(query_text, top_k=top_k)
        ]

    def batch_neighbors(
        self,
        query_texts: Sequence[str],
        top_k: int = 25,
        query_chunk: int = 512,
    ) -> List[List[Tuple[str, float, Set[str]]]]:
        """Vectorized top-k neighbor lookup for many queries at once.

        Builds the BM25 document-weight matrix once (see
        :func:`_bm25_doc_weight_matrix`) and scores each chunk of queries with a
        single sparse mat-mul, instead of ``rank_bm25``'s per-query Python loop
        over the whole corpus.  Output is aligned to ``query_texts`` and is
        identical to calling :meth:`neighbors` per query.  Falls back to the
        per-query path when the sparse build is unavailable (no SciPy / missing
        internals).
        """
        import numpy as np

        built = _bm25_doc_weight_matrix(self.bm25)
        if built is None:
            LOGGER.warning(
                "batch_neighbors: sparse path unavailable; using slow per-query loop."
            )
            return [self.neighbors(str(q), top_k=top_k) for q in query_texts]

        from scipy.sparse import csr_matrix

        W, term_to_col = built
        Wt = W.transpose().tocsr()  # (vocab × n_docs)
        vocab = len(term_to_col)
        n_docs = W.shape[0]
        eff_k = min(top_k, n_docs)

        results: List[List[Tuple[str, float, Set[str]]]] = []
        for start in range(0, len(query_texts), query_chunk):
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
                (np.asarray(data, dtype=np.float32),
                 (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                shape=(len(chunk), vocab),
            )
            scores = (Q @ Wt).toarray()  # (chunk × n_docs), dense
            for r in range(scores.shape[0]):
                sc = scores[r]
                if eff_k < n_docs:
                    idx = np.argpartition(-sc, eff_k - 1)[:eff_k]
                    idx = idx[np.argsort(-sc[idx])]
                else:
                    idx = np.argsort(-sc)
                results.append(
                    [
                        (self.note_ids[i], float(sc[i]), self.code_sets[i])
                        for i in idx[:eff_k]
                    ]
                )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "note_ids": self.note_ids,
                    "code_sets": self.code_sets,
                    "corpus": self.corpus,
                    "raw_texts": self.raw_texts,
                    "gold_code_col": self.gold_code_col,
                    "version": 2,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        LOGGER.info("Saved BM25 train-note index -> %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "TrainNoteBM25Index":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(
            note_ids=obj["note_ids"],
            code_sets=obj["code_sets"],
            corpus=obj["corpus"],
            raw_texts=obj.get("raw_texts"),
            gold_code_col=obj.get("gold_code_col", "proc_codes"),
        )
