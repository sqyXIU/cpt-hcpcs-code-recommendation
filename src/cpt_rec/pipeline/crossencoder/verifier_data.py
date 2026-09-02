# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Data plumbing for the candidate verifier.

Responsibilities
----------------

* :func:`load_llm_concept_sources` — parse an LLM candidate-prior ``.sources.csv``
  into ``{note_id: {code: (llm_rank, llm_confidence)}}`` for the LLM-concept
  source (v1b / inference-time application).
* :func:`build_note_pools` — drive :class:`candidate_pool.PoolAssembler` over a
  notes CSV (snippetize → assemble pool + φ), returning the per-note
  :class:`NotePool` list and the gold map.  Supports a pickle cache so the
  (GPU-bound) snippet encode is paid once and reused across epochs.
* :class:`PoolExampleDataset` — flattens pools into pair-level
  ``(evidence, code_desc, φ, label)`` training examples with optional
  per-note negative subsampling.  The subsampler can be made
  **sibling-aware**: given a KB and ``neg_sibling_frac`` it fills
  part of each note's negative budget from ``family_codes(gold) ∩ pool`` —
  the same-deepest-range CPT confounders the verifier most needs to learn to
  rank *below* the gold — before topping up at random.  Each example also
  carries a ``group_id`` (its note index) so a grouped ranking loss can be
  computed per note.
  batch (so every note's kept positives and negatives stay together) for the
  per-note grouped ranking loss; shuffles group order each epoch.
* :class:`PairCollator` — tokenizes ``(evidence, code_desc)`` pairs and stacks
  φ + labels (+ ``group_ids``) into a batch for either verifier architecture.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from cpt_rec.baselines.common import (
    _ID_CANDIDATES,
    _TEXT_CANDIDATES,
    _pick_col,
)
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.pipeline.pool.candidate_pool import (
    FEATURE_ORDERS,
    NotePool,
    PoolAssembler,
    features_to_vector,
)
from cpt_rec.pipeline.pool.snippetize import snippets_for_note
from cpt_rec.common.preprocess.code_utils import pipe_split_codes

LOGGER = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 3  # bump when NotePool / φ layout OR the assembly key changes


# ---------------------------------------------------------------------------
# LLM-concept source loader
# ---------------------------------------------------------------------------

def load_llm_concept_sources(
    sources_csv: Path | str,
    keep_sources: Sequence[str] = ("llm_concept",),
) -> Dict[str, Dict[str, Tuple[int, Optional[float]]]]:
    """Parse an LLM candidate-prior ``.sources.csv``.

    Columns (an optional user-supplied prior; see docs/OWN_CORPUS.md):
    ``note_id, code, source, llm_rank, llm_confidence, supporting_phrase``.

    Returns ``{note_id: {code: (llm_rank, llm_confidence_or_None)}}`` keeping
    only rows whose ``source`` is in ``keep_sources`` (default the locked
    medgemma concept source).  Codes are uppercased; on duplicate codes the
    best (smallest) rank wins.
    """
    df = pd.read_csv(sources_csv, dtype=str, keep_default_na=False)
    need = {"note_id", "code", "source", "llm_rank", "llm_confidence"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{sources_csv} missing columns: {sorted(missing)}")
    keep = set(keep_sources)
    out: Dict[str, Dict[str, Tuple[int, Optional[float]]]] = {}
    for nid, code, source, rank, conf in zip(
        df["note_id"], df["code"], df["source"], df["llm_rank"], df["llm_confidence"]
    ):
        if source not in keep:
            continue
        nid = str(nid).strip()
        code = str(code).strip().upper()
        if not code:
            continue
        try:
            r = int(float(rank))
        except (TypeError, ValueError):
            r = 0
        c: Optional[float]
        try:
            c = float(conf) if str(conf).strip() != "" else None
        except (TypeError, ValueError):
            c = None
        bucket = out.setdefault(nid, {})
        prev = bucket.get(code)
        if prev is None or r < prev[0]:
            bucket[code] = (r, c)
    LOGGER.info(
        "load_llm_concept_sources: %s notes, sources=%s", len(out), sorted(keep)
    )
    return out


# ---------------------------------------------------------------------------
# Note-pool building (with cache)
# ---------------------------------------------------------------------------

def _read_notes_frame(
    notes_csv: Path | str,
    note_id_col: Optional[str],
    note_text_col: Optional[str],
    gold_code_col: str,
    section_cols: Optional[Sequence[str]],
    include_gold: bool,
) -> Tuple[pd.DataFrame, str, str, List[str], bool]:
    """Load the notes CSV and resolve its column layout.

    The note-id and note-text columns are taken from the explicit args when
    given, else inferred by matching known aliases (``_ID_CANDIDATES`` /
    ``_TEXT_CANDIDATES``).  ``read_csv`` uses ``keep_default_na=False`` so blank
    section cells stay ``""`` rather than ``NaN`` (the snippetizer expects str).

    Returns ``(df, id_col, text_col, valid_section_cols, has_gold)`` where
    ``valid_section_cols`` is the requested section columns actually present and
    ``has_gold`` is True only when gold labels were requested *and* the column
    exists (inference-time CSVs have no gold).
    """
    df = pd.read_csv(notes_csv, dtype=str, keep_default_na=False)
    id_col = note_id_col or _pick_col(df, _ID_CANDIDATES, "note-id")
    text_col = note_text_col or _pick_col(df, _TEXT_CANDIDATES, "note-text")
    valid_sections = [c for c in (section_cols or []) if c in df.columns]
    has_gold = include_gold and gold_code_col in df.columns
    return df, id_col, text_col, valid_sections, has_gold


def build_note_pools(
    notes_csv: Path | str,
    assembler: PoolAssembler,
    bm25_index_path: Optional[Path] = None,
    llm_sources_csv: Optional[Path] = None,
    keep_llm_sources: Sequence[str] = ("llm_concept",),
    is_train: bool = False,
    gold_code_col: str = "proc_codes",
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    section_cols: Optional[Sequence[str]] = None,
    include_gold: bool = True,
    max_snippets: int = 32,
    snippet_max_words: int = 180,
    snippet_overlap_words: int = 60,
    snippet_select: str = "length",
    snippet_pool_cap: int = 0,
    limit: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[List[NotePool], Dict[str, Set[str]]]:
    """Assemble per-note pools (with φ) for every row of ``notes_csv``.

    ``is_train`` enables the neighbor self-leakage guard (the train BM25 index
    contains the note itself).  When ``cache_path`` exists and matches the
    assembler config it is loaded instead of recomputing the snippet encode.
    """
    # Everything that determines pool membership / φ must be in this key: a
    # cache hit under a *different* assembly config silently overrides the
    # requested config (v2 caches keyed on feature_version alone had exactly
    # that hazard — e.g. a 192-token-retrieval cache reused by a 320 run).
    assembly_key = {
        "feature_version": assembler.feature_version,
        "bm25_top_k": assembler.bm25_top_k,
        "dense_top_k": assembler.dense_top_k,
        "neighbor_top_k": assembler.neighbor_top_k,
        "llm_norm_k": assembler.llm_norm_k,
        "bi_max_length": int(getattr(assembler.kb_index.bi_encoder, "max_length", 0)),
        "max_snippets": int(max_snippets),
        "snippet_max_words": int(snippet_max_words),
        "snippet_overlap_words": int(snippet_overlap_words),
        # S2 selection is pool-determining: a different selector keeps
        # different evidence, so it must key the cache or a "cosmargin" run
        # would silently reuse a "length" pool.
        "snippet_select": str(snippet_select),
        "snippet_pool_cap": int(snippet_pool_cap),
        "notes_csv_name": Path(notes_csv).name,
        "is_train": bool(is_train),
        "has_neighbor": bm25_index_path is not None,
        "llm_sources_name": Path(llm_sources_csv).name if llm_sources_csv else None,
        "keep_llm_sources": sorted(keep_llm_sources),
        "section_cols": list(section_cols or []),
        "limit": limit,
    }
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        cached = _load_pool_cache(cache_path, assembly_key)
        if cached is not None:
            # ``include_gold`` is deliberately NOT part of ``assembly_key``:
            # pool MEMBERSHIP never depends on it (``assemble_note`` is not
            # given the gold set), only the returned ``gold`` dict does.  That
            # makes one cache legitimately shareable between a training run's
            # val pool (include_gold=True) and a predict pass (False) -- the
            # expensive half is identical and predict discards ``gold``.
            #
            # The one direction that is NOT safe is the reverse: a cache
            # written by predict carries ``gold == {}``, and a later train run
            # loading it would evaluate every epoch against no labels and pick
            # ``best_threshold``/``best_epoch`` from garbage, silently.  Reject
            # that case and rebuild.  A gold-less CSV that asked for gold is
            # the only false positive, and it costs one recompute.
            if include_gold and cached[0] and not cached[1]:
                LOGGER.warning(
                    "pool cache %s has NO gold labels but include_gold=True "
                    "(it was almost certainly written by a predict run, which "
                    "never stores gold); rebuilding rather than training "
                    "against empty labels.",
                    cache_path,
                )
            else:
                LOGGER.info("build_note_pools: loaded cache %s", cache_path)
                return cached

    # Read the notes frame FIRST so ``limit`` is applied before the (expensive)
    # neighbor query.  Slicing to the first ``limit`` rows here is what makes a
    # ``--max-*-notes`` smoke run actually cheap: previously neighbor_pools ran
    # BM25 over the *entire* CSV regardless of ``limit`` (~21 h on 19k notes).
    df, id_col, text_col, valid_sections, has_gold = _read_notes_frame(
        notes_csv, note_id_col, note_text_col, gold_code_col, section_cols, include_gold
    )
    if limit is not None and limit < len(df):
        df = df.head(limit)
    wanted_ids = {str(v).strip() for v in df[id_col]}

    # ---- neighbor + LLM source maps -------------------------------------
    neighbor_map: Dict[str, List[Tuple[str, Set[str]]]] = {}
    if bm25_index_path is not None:
        from cpt_rec.common.evaluation.candidate_union import neighbor_pools

        neighbor_map = neighbor_pools(
            notes_csv=Path(notes_csv),
            bm25_index_path=Path(bm25_index_path),
            max_top_k=assembler.neighbor_top_k + (1 if is_train else 0),
            note_id_col=note_id_col,
            note_text_col=note_text_col,
            note_ids=wanted_ids,
        )

    llm_map_all: Dict[str, Dict[str, Tuple[int, Optional[float]]]] = {}
    if llm_sources_csv is not None:
        llm_map_all = load_llm_concept_sources(llm_sources_csv, keep_llm_sources)

    LOGGER.info(
        "build_note_pools: %d rows, id=%s text=%s gold=%s sections=%d neighbor=%s llm=%s",
        len(df), id_col, text_col, has_gold, len(valid_sections),
        bm25_index_path is not None, llm_sources_csv is not None,
    )

    from tqdm import tqdm

    pools: List[NotePool] = []
    gold: Dict[str, Set[str]] = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="assemble pools"):
        nid = str(row[id_col]).strip()
        text = str(row[text_col])
        sections = (
            {c: str(row[c]) for c in valid_sections} if valid_sections else None
        )
        # Under S2 selection the budget is spent AFTER encoding, so the
        # snippetizer must not pre-truncate by length -- it only applies the
        # optional cost cap.  Under "length" this is the historical call.
        snippets = snippets_for_note(
            note_id=nid,
            note_text=text,
            sections=sections,
            max_words=snippet_max_words,
            overlap_words=snippet_overlap_words,
            max_snippets=(
                max_snippets if snippet_select == "length"
                else (snippet_pool_cap or None)
            ),
        )
        if not snippets:
            continue
        pool = assembler.assemble_note(
            note_id=nid,
            snippets=snippets,
            select_k=None if snippet_select == "length" else max_snippets,
            neighbor_ranked=neighbor_map.get(nid),
            llm_map=llm_map_all.get(nid),
            exclude_note_id=is_train,
        )
        if pool is None or not pool.records:
            continue
        pools.append(pool)
        if has_gold:
            gold[nid] = set(pipe_split_codes(row[gold_code_col]))

    LOGGER.info("build_note_pools: assembled %d pools", len(pools))
    if cache_path is not None:
        _save_pool_cache(cache_path, pools, gold, assembly_key)
        LOGGER.info("build_note_pools: wrote cache %s", cache_path)
    return pools, gold


def _save_pool_cache(
    path: Path, pools: List[NotePool], gold: Dict[str, Set[str]], assembly_key: Dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write via a pid-unique temp file + atomic rename.  Several runs that share
    # an assembly config legitimately share a cache path and are launched
    # concurrently (one per GPU); writing the final path directly leaves a
    # window in which a reader sees a partially-written pickle.  This is
    # HARDENING, not a fix for an observed failure: the interleaving was not
    # reproducible, and because ``_load_pool_cache`` catches the unpickling
    # error the realistic worst case was a rejected cache and a rebuild, never
    # a wrong pool.  The rename removes the window anyway -- the file is either
    # the previous good cache or a complete new one, never a mixture.
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(
                {
                    "format": CACHE_FORMAT_VERSION,
                    "assembly_key": assembly_key,
                    "pools": pools,
                    "gold": gold,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _load_pool_cache(
    path: Path, assembly_key: Dict
) -> Optional[Tuple[List[NotePool], Dict[str, Set[str]]]]:
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
    except Exception as exc:  # pragma: no cover - corrupt/old cache
        LOGGER.warning("pool cache %s unreadable (%s); rebuilding", path, exc)
        return None
    if blob.get("format") != CACHE_FORMAT_VERSION:
        LOGGER.warning("pool cache %s stale format; rebuilding", path)
        return None
    if blob.get("assembly_key") != assembly_key:
        LOGGER.warning(
            "pool cache %s assembly key mismatch; rebuilding.\n  cached : %s\n  wanted : %s",
            path, blob.get("assembly_key"), assembly_key,
        )
        return None
    return blob["pools"], blob["gold"]


# ---------------------------------------------------------------------------
# Pair-level dataset
# ---------------------------------------------------------------------------

class PoolExampleDataset:
    """Flatten pools into ``(evidence, code_desc, φ, label)`` examples.

    ``neg_per_pos`` (train only): keep all positive (gold) candidates and at
    most ``neg_per_pos × n_pos`` negatives per note (or ``max_neg_per_note``
    when the note has no in-pool gold).  ``None`` keeps the full pool, which
    matches the inference-time distribution exactly.

    Sibling-aware negatives: when ``kb`` is given and
    ``neg_sibling_frac > 0``, the per-note negative budget is filled first from
    the pool negatives that share a *deepest code range* with some gold code
    (``CodeKnowledgeBase.family_codes``) — the near-miss CPT confounders the
    cross-encoder most needs to rank below the gold — and only then topped up at
    random.  With the defaults (``kb=None`` / ``neg_sibling_frac=0.0``) the
    subsampler is the original uniform ``rng.choice``, so the champion run is
    reproduced byte-for-byte.

    Every emitted example carries a ``group_id`` (the index of its source pool
    in ``pools``); :attr:`group_examples` lists, per group, the flat example
    indices belonging to it
    and the per-note grouped ranking loss.
    """

    def __init__(
        self,
        pools: Sequence[NotePool],
        gold: Dict[str, Set[str]],
        feature_version: str,
        neg_per_pos: Optional[int] = None,
        max_neg_per_note: Optional[int] = None,
        seed: int = 0,
        kb: Optional[CodeKnowledgeBase] = None,
        neg_sibling_frac: float = 0.0,
    ) -> None:
        self.order = FEATURE_ORDERS[feature_version]
        self.kb = kb
        self.neg_sibling_frac = float(neg_sibling_frac)
        self._sibling_cache: Dict[str, Set[str]] = {}
        rng = np.random.default_rng(seed)

        evidences: List[str] = []
        descs: List[str] = []
        feats: List[np.ndarray] = []
        labels: List[float] = []
        groups: List[int] = []
        group_examples: List[List[int]] = []
        n_sibling_negs = 0

        for gi, pool in enumerate(pools):
            g = gold.get(pool.note_id, set())
            pos_idx = [i for i, r in enumerate(pool.records) if r.code in g]
            neg_idx = [i for i, r in enumerate(pool.records) if r.code not in g]

            if neg_per_pos is not None:
                if pos_idx:
                    budget = neg_per_pos * len(pos_idx)
                else:
                    budget = max_neg_per_note if max_neg_per_note is not None else len(neg_idx)
                if budget < len(neg_idx):
                    neg_idx, n_sib = self._subsample_negatives(
                        pool, neg_idx, budget, g, rng
                    )
                    n_sibling_negs += n_sib
            keep = sorted(pos_idx + neg_idx)

            ex_ids: List[int] = []
            for i in keep:
                rec = pool.records[i]
                evidences.append(pool.snippet_texts[rec.best_snippet_idx])
                descs.append(rec.description)
                feats.append(features_to_vector(rec.feats, self.order))
                labels.append(1.0 if rec.code in g else 0.0)
                groups.append(gi)
                ex_ids.append(len(labels) - 1)
            if ex_ids:
                group_examples.append(ex_ids)

        self.evidences = evidences
        self.descs = descs
        self.feats = (
            np.stack(feats).astype(np.float32)
            if feats
            else np.zeros((0, len(self.order)), dtype=np.float32)
        )
        self.labels = np.asarray(labels, dtype=np.float32)
        self.groups = np.asarray(groups, dtype=np.int64)
        self.group_examples = group_examples
        n_pos = int(self.labels.sum())
        LOGGER.info(
            "PoolExampleDataset: %d examples (%d pos / %d neg) feat_dim=%d "
            "groups=%d sibling_negs=%d (sibling_frac=%.2f)",
            len(self.labels), n_pos, len(self.labels) - n_pos, self.feats.shape[1],
            len(self.group_examples), n_sibling_negs, self.neg_sibling_frac,
        )

    # ------------------------------------------------------------------
    def _siblings(self, code: str) -> Set[str]:
        """Memoized ``family_codes(code)`` (same deepest range, incl. ``code``)."""
        s = self._sibling_cache.get(code)
        if s is None:
            s = self.kb.family_codes(code) if self.kb is not None else set()
            self._sibling_cache[code] = s
        return s

    def _subsample_negatives(
        self,
        pool: NotePool,
        neg_idx: List[int],
        budget: int,
        gold_codes: Set[str],
        rng: np.random.Generator,
    ) -> Tuple[List[int], int]:
        """Pick ``budget`` negatives for one note, returning ``(chosen, n_sib)``.

        With ``kb=None`` or ``neg_sibling_frac<=0`` this is the original uniform
        sample (``n_sib=0``).  Otherwise it reserves ``round(frac × budget)`` of
        the budget for *sibling* negatives — pool codes sharing a deepest range
        with any gold code — then fills the remainder uniformly from whatever is
        left (leftover siblings + non-siblings), so the total is always
        ``budget`` even when siblings are scarce or plentiful.
        """
        if self.kb is None or self.neg_sibling_frac <= 0.0:
            return [int(i) for i in rng.choice(neg_idx, size=budget, replace=False)], 0

        sib_target = set()
        for gc in gold_codes:
            sib_target |= self._siblings(gc)
        sib_neg = [i for i in neg_idx if pool.records[i].code in sib_target]
        other_neg = [i for i in neg_idx if pool.records[i].code not in sib_target]

        sib_quota = min(len(sib_neg), int(round(self.neg_sibling_frac * budget)))
        chosen: List[int] = []
        if sib_quota > 0:
            chosen = [int(i) for i in rng.choice(sib_neg, size=sib_quota, replace=False)]
        n_sib = len(chosen)

        remaining = budget - n_sib
        if remaining > 0:
            chosen_set = set(chosen)
            leftover = [i for i in (sib_neg + other_neg) if i not in chosen_set]
            if remaining < len(leftover):
                chosen.extend(
                    int(i) for i in rng.choice(leftover, size=remaining, replace=False)
                )
            else:
                chosen.extend(leftover)
        return chosen, n_sib

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Dict[str, object]:
        return {
            "evidence": self.evidences[i],
            "desc": self.descs[i],
            "features": self.feats[i],
            "label": float(self.labels[i]),
            "group_id": int(self.groups[i]),
        }

    def pos_weight(self) -> float:
        """``n_neg / n_pos`` for an optional BCE ``pos_weight``."""
        n_pos = float(self.labels.sum())
        n_neg = float(len(self.labels) - n_pos)
        return (n_neg / n_pos) if n_pos > 0 else 1.0


# Minimum code-side tokens kept by the robust ``only_second`` encoder so the
# candidate's identity (its description, which sits at the FRONT of the code
# string, before any appended card) survives even when the evidence is long.
_ONLY_SECOND_CODE_FLOOR = 64


def encode_text_pairs(
    tokenizer,
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    max_length: int,
    truncation: str = "longest_first",
):
    """Tokenize ``(evidence, code_desc)`` text pairs → ``(input_ids, attention_mask)``.

    Shared by :class:`PairCollator` (train) and :func:`score_pool` (eval/predict)
    so the two paths always encode identically.

    For every strategy except ``only_second`` this is a thin wrapper over the HF
    tokenizer, byte-identical to the champion for the default ``longest_first``.

    ``only_second`` is made ROBUST here. HF's native ``only_second`` refuses to
    trim the first sequence, so it raises ``Sequence to truncate too short`` when
    the evidence (``text_a``) alone exceeds the window. We instead split the
    window by an evidence-priority budget: evidence keeps up to the whole window,
    the code side (``text_b`` = description + optional card) takes the remainder
    and is trimmed FROM THE END (dropping the appended card first, then the
    description). We reserve ``_ONLY_SECOND_CODE_FLOOR`` tokens for the code side
    when it has content so the candidate's identity is never lost, and fall back
    to trimming the evidence only when it alone fills the window. Special tokens
    and attention mask match the standard pair encoding ([CLS] a [SEP] b [SEP]);
    token_type_ids are dropped, exactly as the champion's collator/scorer do.
    """
    texts_a = [str(t) for t in texts_a]
    texts_b = [str(t) for t in texts_b]

    if truncation != "only_second":
        enc = tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    # Robust evidence-priority split for only_second.
    n_special = tokenizer.num_special_tokens_to_add(pair=True)
    budget = max(0, max_length - n_special)
    a_ids = tokenizer(texts_a, add_special_tokens=False, truncation=False)["input_ids"]
    b_ids = tokenizer(texts_b, add_special_tokens=False, truncation=False)["input_ids"]

    rows = []
    for a, b in zip(a_ids, b_ids):
        floor = min(len(b), _ONLY_SECOND_CODE_FLOOR)
        a = a[: max(0, budget - floor)]
        b = b[: max(0, budget - len(a))]
        rows.append(tokenizer.build_inputs_with_special_tokens(a, b))

    enc = tokenizer.pad(
        {"input_ids": rows},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def score_pool(
    model,
    pool: NotePool,
    order: Sequence[str],
    device,
    arch: str,
    tokenizer=None,
    max_length: int = 192,
    batch_size: int = 128,
    truncation: str = "longest_first",
) -> np.ndarray:
    """Score every candidate in a :class:`NotePool`, returning probabilities
    aligned to ``pool.records``.  Shared by the val loop and the predictor.

    ``truncation`` is the HuggingFace pair-truncation strategy. The default
    ``longest_first`` is the champion's behavior; ``only_second`` preserves the
    evidence (first) sequence and trims only the code/card text (second), which
    requires ``max_length`` to be wide enough to hold the evidence.
    """
    import torch

    recs = pool.records
    if not recs:
        return np.zeros(0, dtype=np.float32)
    feats = np.stack([features_to_vector(r.feats, order) for r in recs]).astype(np.float32)
    out: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(recs), batch_size):
            chunk = recs[start : start + batch_size]
            f = torch.tensor(feats[start : start + batch_size], device=device)
            if arch == "cross_encoder":
                input_ids, attention_mask = encode_text_pairs(
                    tokenizer,
                    [pool.snippet_texts[r.best_snippet_idx] for r in chunk],
                    [r.description for r in chunk],
                    max_length=max_length,
                    truncation=truncation,
                )
                logits = model(
                    input_ids.to(device), attention_mask.to(device), f
                )
            else:
                logits = model(features=f)
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


class PairCollator:
    """Tokenize ``(evidence, code_desc)`` pairs and stack φ + labels.

    Always emits ``group_ids`` (the per-example note index) so the training
    loop can compute the per-note grouped ranking loss; it is harmless for the
    pointwise-BCE path, which simply ignores it.
    """

    def __init__(self, tokenizer, max_length: int = 192, need_text: bool = True,
                 pair_truncation: str = "longest_first"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.need_text = need_text
        self.pair_truncation = pair_truncation

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        import torch

        features = torch.tensor(
            np.stack([b["features"] for b in batch]), dtype=torch.float32
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        group_ids = torch.tensor(
            [int(b.get("group_id", 0)) for b in batch], dtype=torch.long
        )
        out: Dict[str, object] = {
            "features": features, "labels": labels, "group_ids": group_ids,
        }
        if self.need_text:
            input_ids, attention_mask = encode_text_pairs(
                self.tokenizer,
                [str(b["evidence"]) for b in batch],
                [str(b["desc"]) for b in batch],
                max_length=self.max_length,
                truncation=self.pair_truncation,
            )
            out["input_ids"] = input_ids
            out["attention_mask"] = attention_mask
        return out
