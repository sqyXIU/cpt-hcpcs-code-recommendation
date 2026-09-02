# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Candidate-pool assembly with source-tagged φ(n, c) features.

The cross-encoder verifier is the binding
F1 constraint: on the lifted candidate pool its actual micro-F1 (~0.468) is
far below the pool's oracle ceiling (~0.92–0.94).  A bi-encoder that embeds
snippets and code descriptions *independently* (no token-level cross
attention) simply cannot disambiguate fine-grained CPT siblings.

This module builds, per note, the union candidate pool the
**cross-encoder verifier** scores:

    C(note) = KB-BM25@k  ∪  KB-dense@k  ∪  neighbor-gold@k  [∪  LLM-concept]

and attaches to every candidate:

* its **best evidence snippet** — the snippet whose bi-encoder embedding is
  most cosine-similar to the candidate's KB dense vector.  This collapses the
  M×N (snippet × candidate) matrix to one (evidence, code) pair per
  candidate, which is what the cross-encoder consumes.  The single
  ``sims = snippet_embs @ kb_dense_vecs.T`` product per note serves *both*
  dense retrieval *and* best-snippet selection, so no KB re-encode is needed.
* a **source-tagged feature vector** φ(n, c): per-source presence / rank /
  score, neighbor frequency, best-snippet cosine + position, source count,
  HCPCS flag, description length, and (v1b) LLM presence / rank / confidence.

Design notes
------------

* **No gold injection.**  Training labels are simply ``code ∈ gold``; the
  verifier only ever sees in-pool candidates, so the φ distribution at train
  and inference time is identical.  Gold codes the retrieval sources miss are
  unreachable regardless of the scorer, so excluding them from training keeps
  the verifier well-calibrated on the reachable pool (the apples-to-apples
  comparison surface against the bi-encoder's 0.468).
* **v1a vs v1b.**  ``v1a`` is source-agnostic over KB∪BM25∪neighbor and needs
  no train-time LLM generation (LLM-concept candidates are applied only at
  inference).  ``v1b`` adds the ``llm_*`` features and expects an LLM-concept
  source map at train time too.
* **Neighbor self-leakage guard.**  At train time the BM25 neighbor index
  contains the note itself; pass ``exclude_note_id=True`` so a note can never
  be its own neighbor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.pipeline.pool.candidate_gen import KBCandidateIndex
from cpt_rec.pipeline.pool.snippetize import (
    Snippet,
    select_snippet_indices,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature ordering — the canonical column order of φ(n, c).
#
# These names are persisted into the verifier ``config.json`` (under
# ``feature_order``) so train and predict vectorize φ identically.  Adding a
# feature => new ordered name appended here + populated in ``_compute_feats``.
# ---------------------------------------------------------------------------

_V1A_FEATURES: List[str] = [
    "kb_bm25_present",
    "kb_bm25_rank",
    "kb_bm25_score",
    "kb_dense_present",
    "kb_dense_rank",
    "kb_dense_score",
    "nbr_present",
    "nbr_rank",
    "nbr_frac",
    "best_snip_cos",
    "best_snip_pos",
    "n_sources",
    "is_hcpcs",
    "desc_len",
]

_V1B_FEATURES: List[str] = _V1A_FEATURES + [
    "llm_present",
    "llm_rank",
    "llm_confidence",
]

FEATURE_ORDERS: Dict[str, List[str]] = {
    "v1a": _V1A_FEATURES,
    "v1b": _V1B_FEATURES,
}


def feature_dim(version: str) -> int:
    return len(FEATURE_ORDERS[version])


def features_to_vector(feats: Dict[str, float], order: Sequence[str]) -> np.ndarray:
    """Vectorize a named-feature dict into the canonical column order."""
    return np.asarray([float(feats.get(k, 0.0)) for k in order], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    """One (note, candidate-code) row the verifier will score."""

    code: str
    description: str
    best_snippet_idx: int        # index into NotePool.snippet_texts
    best_snippet_cos: float      # cos(best snippet emb, code dense vec)
    sources: Tuple[str, ...]     # subset of {"bm25","dense","neighbor","llm"}
    feats: Dict[str, float]      # named raw features (superset of all orders)


@dataclass
class NotePool:
    """A note's assembled candidate pool plus its evidence snippets."""

    note_id: str
    snippet_texts: List[str]            # tagged_text() per snippet, in order
    records: List[CandidateRecord] = field(default_factory=list)

    def codes(self) -> List[str]:
        return [r.code for r in self.records]


@dataclass
class _CandidateSignals:
    """Raw per-``(note, code)`` signals consumed by ``_compute_feats``.

    Bundles what used to be a 16-argument call into one self-documenting
    record.  Every field maps 1:1 to a φ feature or its normalizer; the
    ``in_*`` flags say which sources surfaced the code and the ``*_rank`` /
    ``*_score`` fields are ``None`` when the corresponding source did not.
    """

    code: str
    desc: str
    n_snippets: int
    best_snippet_idx: int
    best_snippet_cos: float
    in_bm25: bool
    bm25_rank: Optional[int]
    bm25_score: Optional[float]
    bm25_max: float
    in_dense: bool
    dense_rank: Optional[int]
    in_neighbor: bool
    neighbor_rank: Optional[int]
    neighbor_count: int
    n_neighbors_used: int
    in_llm: bool
    llm_rank: Optional[int]
    llm_confidence: Optional[float]
    n_sources: int


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class PoolAssembler:
    """Assemble :class:`NotePool` objects with source-tagged φ features.

    Parameters
    ----------
    kb :
        Knowledge base (code descriptions + ``system`` for the HCPCS flag).
    kb_index :
        ``cptrec-build-kb-index`` artifact wrapper.  Supplies the snippet bi-encoder,
        the KB dense vectors, BM25 ranking, and ``code_to_row``.
    bm25_top_k, dense_top_k :
        Per-snippet retrieval depths whose union (across snippets) forms the
        KB portion of the pool.  Mirrors ``KBCandidateIndex.topk_union_batch``
        so the pool matches the recall-ceiling measurements.
    neighbor_top_k :
        Number of BM25 training-note neighbors whose gold sets union into the
        neighbor source.
    llm_norm_k :
        Normalizer for the LLM-concept rank feature.
    feature_version :
        ``"v1a"`` or ``"v1b"`` — selects which φ columns the verifier reads
        (the raw ``feats`` dict always carries the full superset).
    encode_batch_size :
        Snippet bi-encoder batch size.
    """

    def __init__(
        self,
        kb: CodeKnowledgeBase,
        kb_index: KBCandidateIndex,
        bm25_top_k: int = 25,
        dense_top_k: int = 25,
        neighbor_top_k: int = 25,
        llm_norm_k: int = 50,
        feature_version: str = "v1a",
        encode_batch_size: int = 64,
        snippet_select: str = "length",
    ) -> None:
        if snippet_select not in SNIPPET_SELECTORS:
            raise ValueError(
                f"unknown snippet_select {snippet_select!r}; "
                f"expected one of {sorted(SNIPPET_SELECTORS)}"
            )
        if feature_version not in FEATURE_ORDERS:
            raise ValueError(
                f"unknown feature_version {feature_version!r}; "
                f"expected one of {sorted(FEATURE_ORDERS)}"
            )
        self.kb = kb
        self.kb_index = kb_index
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.neighbor_top_k = neighbor_top_k
        self.llm_norm_k = llm_norm_k
        self.feature_version = feature_version
        self.encode_batch_size = encode_batch_size
        self.snippet_select = snippet_select
        self._code_to_row = kb_index.code_to_row
        self._codes = kb_index.codes
        self._n_codes = len(self._codes)

    # ------------------------------------------------------------------
    def assemble_note(
        self,
        note_id: str,
        snippets: List[Snippet],
        neighbor_ranked: Optional[List[Tuple[str, Set[str]]]] = None,
        llm_map: Optional[Dict[str, Tuple[int, float]]] = None,
        exclude_note_id: bool = False,
        select_k: Optional[int] = None,
    ) -> Optional[NotePool]:
        """Assemble one note's union candidate pool with φ features.

        ``neighbor_ranked`` is the note's ranked neighbor list
        ``[(neighbor_note_id, neighbor_gold_set), ...]`` as produced by
        :func:`evaluation.candidate_union.neighbor_pools` (already truncated
        to ``>= neighbor_top_k``).  ``llm_map`` maps ``code -> (rank, conf)``
        for the LLM-concept source.  Returns ``None`` when the note yields no
        snippets.

        The four retrieval sources (dense, BM25, neighbor-gold, LLM-concept)
        are each built by a private ``_*_source`` helper and unioned; a single
        snippet-vs-KB similarity matrix powers *both* dense retrieval and the
        best-evidence-snippet selection, so the KB is never re-encoded.
        """
        snip_texts = [s.tagged_text() for s in snippets]
        n_snippets = len(snip_texts)
        if n_snippets == 0:
            return None

        snip_embs = self.kb_index.encode_queries(
            snip_texts, batch_size=self.encode_batch_size
        )  # (n_snippets, d), L2-normalized
        sims = snip_embs @ self.kb_index.dense_vecs.T  # (n_snippets, n_codes)

        # ---- S2: evidence-snippet selection -----------------------------
        # Spend the snippet budget on windows that look like coded procedure
        # text.  This runs AFTER the encode because it needs ``sims`` -- the
        # same matrix dense retrieval already requires, so it costs one
        # reduction, not a second forward pass.  Under "length" the budget was
        # already applied upstream by ``snippets_for_note`` and this is a
        # no-op, which is what keeps the historical path byte-identical.
        score_fn = SNIPPET_SELECTORS[self.snippet_select]
        scores = score_fn(sims)
        if scores is not None and select_k and n_snippets > select_k:
            keep = select_snippet_indices(scores, select_k)
            snippets = [snippets[i] for i in keep]
            snip_texts = [snip_texts[i] for i in keep]
            sims = sims[keep]
            n_snippets = len(keep)

        # Best evidence snippet per KB code (also the note-level dense score).
        best_snippet_per_code = sims.argmax(axis=0)   # (n_codes,)
        best_cos_per_code = sims.max(axis=0)          # (n_codes,)

        # ---- the four candidate sources ---------------------------------
        dense_rank = self._dense_source(sims, best_cos_per_code)
        bm25_score, bm25_rank, bm25_max = self._bm25_source(snip_texts)
        nbr_rank, nbr_count, n_nbr_used = self._neighbor_source(
            note_id, neighbor_ranked, exclude_note_id
        )
        llm_valid = self._llm_source(llm_map)

        # ---- union + φ per candidate ------------------------------------
        cand_rows = set(dense_rank) | set(bm25_score)
        cand_codes: Set[str] = {self._codes[row] for row in cand_rows}
        cand_codes.update(nbr_rank.keys())
        cand_codes.update(llm_valid.keys())

        records: List[CandidateRecord] = []
        for code in sorted(cand_codes):
            code_row = self._code_to_row[code]
            in_bm25 = code_row in bm25_score
            in_dense = code_row in dense_rank
            in_neighbor = code in nbr_rank
            in_llm = code in llm_valid
            sources = [
                name
                for name, present in (
                    ("bm25", in_bm25),
                    ("dense", in_dense),
                    ("neighbor", in_neighbor),
                    ("llm", in_llm),
                )
                if present
            ]
            llm_rank, llm_conf = llm_valid.get(code, (None, None))
            signals = _CandidateSignals(
                code=code,
                desc=self.kb.description(code) or "",
                n_snippets=n_snippets,
                best_snippet_idx=int(best_snippet_per_code[code_row]),
                best_snippet_cos=float(best_cos_per_code[code_row]),
                in_bm25=in_bm25,
                bm25_rank=bm25_rank.get(code_row),
                bm25_score=bm25_score.get(code_row),
                bm25_max=bm25_max,
                in_dense=in_dense,
                dense_rank=dense_rank.get(code_row),
                in_neighbor=in_neighbor,
                neighbor_rank=nbr_rank.get(code),
                neighbor_count=nbr_count.get(code, 0),
                n_neighbors_used=n_nbr_used,
                in_llm=in_llm,
                llm_rank=llm_rank,
                llm_confidence=llm_conf,
                n_sources=len(sources),
            )
            records.append(
                CandidateRecord(
                    code=code,
                    description=signals.desc,
                    best_snippet_idx=signals.best_snippet_idx,
                    best_snippet_cos=signals.best_snippet_cos,
                    sources=tuple(sources),
                    feats=self._compute_feats(signals),
                )
            )

        return NotePool(note_id=str(note_id), snippet_texts=snip_texts, records=records)

    # ------------------------------------------------------------------
    # Per-source builders (each returns rank/score lookups keyed by KB row or
    # code).  ``assemble_note`` unions their keys into the candidate pool.
    # ------------------------------------------------------------------
    def _dense_source(
        self, sims: np.ndarray, best_cos_per_code: np.ndarray
    ) -> Dict[int, int]:
        """Per-snippet dense top-``dense_top_k`` union → ``code_row -> rank``.

        Rank is by descending best (across-snippet) cosine, matching how the
        recall-ceiling pool was measured.  ``np.argpartition`` gives the
        unordered top-k per snippet in O(n_codes) without a full sort.

        Scalability
        -----------
        Both this top-k and the ``sims = snip_embs @ dense_vecs.T`` in
        :meth:`assemble_note` are O(n_codes) *per snippet*; over the full
        CPT/HCPCS vocabulary that full scan is the dominant per-note cost.  The
        targeted fix is to back the dense top-k with a prebuilt ANN index
        (FAISS ``IndexFlatIP`` / HNSW or ``hnswlib``) on ``dense_vecs`` and
        compute ``best_cos`` / ``best_snippet`` lazily for only the ~few-hundred
        pooled candidate codes (a column-max over the union) instead of all
        n_codes — turning per-snippet O(n_codes) into O(log n_codes) while
        leaving every φ value identical.
        """
        n_snippets = sims.shape[0]
        dense_rows: Set[int] = set()
        dense_k = min(self.dense_top_k, self._n_codes)
        if dense_k > 0:
            for snippet_idx in range(n_snippets):
                snippet_scores = sims[snippet_idx]
                if dense_k < self._n_codes:
                    top = np.argpartition(-snippet_scores, dense_k - 1)[:dense_k]
                else:
                    top = np.arange(self._n_codes)
                dense_rows.update(int(row) for row in top)
        dense_sorted = sorted(dense_rows, key=lambda row: -best_cos_per_code[row])
        return {code_row: rank for rank, code_row in enumerate(dense_sorted)}

    def _bm25_source(
        self, snip_texts: List[str]
    ) -> Tuple[Dict[int, float], Dict[int, int], float]:
        """Per-snippet BM25 top-``bm25_top_k`` union → (score, rank, note-max).

        One vectorized sparse mat-mul over all snippets instead of a per-snippet
        ``get_scores`` Python loop (the historical pool-assembly bottleneck);
        each code keeps its best score across snippets, ranked descending.
        """
        bm25_score: Dict[int, float] = {}
        for per_snippet in self.kb_index.bm25_topk_scored_batch(
            snip_texts, self.bm25_top_k
        ):
            for code_row, score in per_snippet:
                if code_row not in bm25_score or score > bm25_score[code_row]:
                    bm25_score[code_row] = score
        bm25_sorted = sorted(bm25_score, key=lambda row: -bm25_score[row])
        bm25_rank = {code_row: rank for rank, code_row in enumerate(bm25_sorted)}
        bm25_max = max(bm25_score.values()) if bm25_score else 1.0
        return bm25_score, bm25_rank, bm25_max

    def _neighbor_source(
        self,
        note_id: str,
        neighbor_ranked: Optional[List[Tuple[str, Set[str]]]],
        exclude_note_id: bool,
    ) -> Tuple[Dict[str, int], Dict[str, int], int]:
        """Top-``neighbor_top_k`` BM25 neighbor notes → gold-code (rank, count).

        Records each code's earliest (best) neighbor rank and how many of the
        used neighbors carry it.  ``exclude_note_id`` drops the note from its
        own neighbor list (the train BM25 index contains it — the self-leakage
        guard).  Returns ``(best_rank, count, n_neighbors_used)``.
        """
        nbr_rank: Dict[str, int] = {}
        nbr_count: Dict[str, int] = {}
        if not neighbor_ranked:
            return nbr_rank, nbr_count, 0

        used: List[Tuple[str, Set[str]]] = []
        for nbr_id, codeset in neighbor_ranked:
            if exclude_note_id and str(nbr_id) == str(note_id):
                continue
            used.append((nbr_id, codeset))
            if len(used) >= self.neighbor_top_k:
                break
        for rank, (_nbr_id, codeset) in enumerate(used):
            for code in codeset:
                if code not in self._code_to_row:
                    continue
                if code not in nbr_rank:
                    nbr_rank[code] = rank  # ascending rank -> first is best
                nbr_count[code] = nbr_count.get(code, 0) + 1
        return nbr_rank, nbr_count, len(used)

    def _llm_source(
        self, llm_map: Optional[Dict[str, Tuple[int, float]]]
    ) -> Dict[str, Tuple[int, float]]:
        """LLM-concept ``code -> (rank, confidence)`` restricted to in-KB codes."""
        llm_map = llm_map or {}
        return {code: rc for code, rc in llm_map.items() if code in self._code_to_row}

    # ------------------------------------------------------------------
    def _compute_feats(self, sig: _CandidateSignals) -> Dict[str, float]:
        """Named raw φ features (full superset; vectorized later by feature order).

        Rank features are normalized into ``[0, 1]`` with ``1.0`` as the
        "absent / worst" sentinel, so a code missing from a source reads as a
        uniformly bad rank rather than a spurious ``0``.  Scores are normalized
        within the note (BM25 by the per-note max; the dense score is already a
        cosine).

        =================  ================================================
        φ feature          meaning / normalization
        =================  ================================================
        kb_bm25_present    1 if surfaced by BM25
        kb_bm25_rank       union rank / ``bm25_top_k``    (1.0 = absent)
        kb_bm25_score      score / per-note max
        kb_dense_present   1 if surfaced by dense retrieval
        kb_dense_rank      union rank / ``dense_top_k``   (1.0 = absent)
        kb_dense_score     best-snippet cosine            (0 if absent)
        nbr_present        1 if in a neighbor's gold set
        nbr_rank           best neighbor rank / ``neighbor_top_k``
        nbr_frac           neighbors carrying it / neighbors used
        best_snip_cos      best-snippet cosine
        best_snip_pos      best snippet's position in ``[0, 1]``
        n_sources          #sources / 4
        is_hcpcs           HCPCS-system flag
        desc_len           min(#desc words / 64, 1)
        llm_present        (v1b) 1 if from LLM-concept
        llm_rank           (v1b) rank / ``llm_norm_k``
        llm_confidence     (v1b) model confidence (0.5 if present-but-unscored)
        =================  ================================================
        """
        bm25_k = max(1, self.bm25_top_k)
        dense_k = max(1, self.dense_top_k)
        neighbor_k = max(1, self.neighbor_top_k)
        llm_k = max(1, self.llm_norm_k)
        is_hcpcs = 1.0 if (self.kb.system(sig.code) or "").upper() == "HCPCS" else 0.0
        desc_len = min(len(sig.desc.split()) / 64.0, 1.0)
        return {
            "kb_bm25_present": 1.0 if sig.in_bm25 else 0.0,
            "kb_bm25_rank": (sig.bm25_rank / bm25_k) if (sig.in_bm25 and sig.bm25_rank is not None) else 1.0,
            "kb_bm25_score": (sig.bm25_score / sig.bm25_max) if (sig.in_bm25 and sig.bm25_score is not None and sig.bm25_max > 0) else 0.0,
            "kb_dense_present": 1.0 if sig.in_dense else 0.0,
            "kb_dense_rank": (sig.dense_rank / dense_k) if (sig.in_dense and sig.dense_rank is not None) else 1.0,
            "kb_dense_score": sig.best_snippet_cos if sig.in_dense else 0.0,
            "nbr_present": 1.0 if sig.in_neighbor else 0.0,
            "nbr_rank": (sig.neighbor_rank / neighbor_k) if (sig.in_neighbor and sig.neighbor_rank is not None) else 1.0,
            "nbr_frac": (sig.neighbor_count / sig.n_neighbors_used) if (sig.in_neighbor and sig.n_neighbors_used > 0) else 0.0,
            "best_snip_cos": float(sig.best_snippet_cos),
            "best_snip_pos": (sig.best_snippet_idx / max(1, sig.n_snippets - 1)) if sig.n_snippets > 1 else 0.0,
            "n_sources": sig.n_sources / 4.0,
            "is_hcpcs": is_hcpcs,
            "desc_len": desc_len,
            "llm_present": 1.0 if sig.in_llm else 0.0,
            "llm_rank": (sig.llm_rank / llm_k) if (sig.in_llm and sig.llm_rank is not None) else 1.0,
            "llm_confidence": float(sig.llm_confidence) if (sig.in_llm and sig.llm_confidence is not None) else (0.5 if sig.in_llm else 0.0),
        }


# ---------------------------------------------------------------------------
# Assembler configuration (single source of truth for train ↔ predict parity)
# ---------------------------------------------------------------------------

# S2 selector scores, computed on the snippet x KB-code cosine matrix that
# dense retrieval already builds.  Each maps (n_snippets, n_codes) -> (n_snippets,).
SNIPPET_SELECTORS: Dict[str, Callable] = {
    "length": lambda sims: None,          # sentinel: selection happened upstream
    "cosmax": lambda sims: sims.max(axis=1),
    "cosmargin": lambda sims: sims.max(axis=1) - sims.mean(axis=1),
}


@dataclass(frozen=True)
class AssemblerConfig:
    """Hyper-parameters that fully determine candidate-pool assembly.

    Persisted verbatim into the verifier ``config.json`` under the
    ``"assembler"`` key at train time and re-read at predict time, so the pool a
    model is *scored* on is byte-for-byte the pool it was *trained* on.  This is
    the single source of truth for assembler defaults — it stops ``train_verifier``
    and ``predict_verifier`` from silently drifting apart, which would confound an
    A/B (the pool would differ for reasons other than the variable under test).

    The first four fields configure :class:`PoolAssembler`; the three
    ``snippet_*`` / ``max_snippets`` fields configure
    :func:`cpt_rec.pipeline.crossencoder.verifier_data.build_note_pools`; ``encode_batch_size`` is
    the snippet bi-encoder batch size.
    """

    bm25_top_k: int = 25
    dense_top_k: int = 25
    neighbor_top_k: int = 25
    llm_norm_k: int = 50
    max_snippets: int = 32
    snippet_max_words: int = 180
    snippet_overlap_words: int = 60
    encode_batch_size: int = 64
    # --- S2 evidence-snippet selection ---------------------------------
    # "length"    = historical path: down-selection happens upstream in
    #               ``snippets_for_note`` and ranks by ``n_words`` only.
    # "cosmax"    = keep the snippets with the highest peak cosine to ANY KB
    #               code -- "does this window look like a coded procedure?"
    # "cosmargin" = peak minus mean cosine.  Rejects generic surgical
    #               boilerplate, which sits moderately close to *every* code
    #               and so scores high on cosmax but flat on cosmargin.
    snippet_select: str = "length"
    # Windows allowed through to the encoder before selection (0 = no cap).
    # Bounds the cost of dropping the section filter, since a whole note
    # produces many more windows than six sections do.
    snippet_pool_cap: int = 0

    def to_config_dict(self) -> Dict[str, int]:
        """The ``"assembler"`` block to embed in ``config.json``."""
        return {
            "bm25_top_k": self.bm25_top_k,
            "dense_top_k": self.dense_top_k,
            "neighbor_top_k": self.neighbor_top_k,
            "llm_norm_k": self.llm_norm_k,
            "max_snippets": self.max_snippets,
            "snippet_max_words": self.snippet_max_words,
            "snippet_overlap_words": self.snippet_overlap_words,
            "encode_batch_size": self.encode_batch_size,
            "snippet_select": self.snippet_select,
            "snippet_pool_cap": self.snippet_pool_cap,
        }

    @classmethod
    def from_config_dict(cls, cfg: Dict) -> "AssemblerConfig":
        """Parse from a loaded ``config.json`` (reads its ``"assembler"`` block).

        Missing keys fall back to the dataclass defaults, so a model whose
        ``config.json`` predates a given field still loads with the historical
        default rather than raising.
        """
        asm = dict(cfg.get("assembler", {}))
        d = cls()
        return cls(
            bm25_top_k=int(asm.get("bm25_top_k", d.bm25_top_k)),
            dense_top_k=int(asm.get("dense_top_k", d.dense_top_k)),
            neighbor_top_k=int(asm.get("neighbor_top_k", d.neighbor_top_k)),
            llm_norm_k=int(asm.get("llm_norm_k", d.llm_norm_k)),
            max_snippets=int(asm.get("max_snippets", d.max_snippets)),
            snippet_max_words=int(asm.get("snippet_max_words", d.snippet_max_words)),
            snippet_overlap_words=int(asm.get("snippet_overlap_words", d.snippet_overlap_words)),
            encode_batch_size=int(asm.get("encode_batch_size", d.encode_batch_size)),
            snippet_select=str(asm.get("snippet_select", d.snippet_select)),
            snippet_pool_cap=int(asm.get("snippet_pool_cap", d.snippet_pool_cap)),
        )

    def build_assembler(
        self, kb: CodeKnowledgeBase, kb_index: KBCandidateIndex, feature_version: str = "v1a"
    ) -> "PoolAssembler":
        """Construct the :class:`PoolAssembler` these settings describe."""
        return PoolAssembler(
            kb=kb,
            kb_index=kb_index,
            bm25_top_k=self.bm25_top_k,
            dense_top_k=self.dense_top_k,
            neighbor_top_k=self.neighbor_top_k,
            snippet_select=self.snippet_select,
            llm_norm_k=self.llm_norm_k,
            feature_version=feature_version,
            encode_batch_size=self.encode_batch_size,
        )
