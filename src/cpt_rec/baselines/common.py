# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Shared utilities for the baselines.

Every baseline produces a predictions CSV with the schema consumed by the
Evaluation harness:

    note_id       — string, matches gold NOTE_ID
    pred_codes    — pipe-separated predicted codes
    pred_scores   — pipe-separated float scores, aligned with pred_codes
                    (optional but strongly encouraged)

This module centralizes:

* loading an input notes CSV (normalized gold or raw op-note CSV)
* writing the predictions CSV (atomic on POSIX)
* a tiny ``Prediction`` dataclass used internally
* code post-filters (keep only in-KB codes, drop empty strings, dedupe)
* deterministic note ordering + ``--limit`` slicing
* end-of-run prediction-stats logging
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Per-note prediction, ready to be serialized."""

    note_id: str
    codes: List[str] = field(default_factory=list)
    scores: Optional[List[float]] = None

    def __post_init__(self) -> None:
        self.codes = [c.strip().upper() for c in self.codes if c and str(c).strip()]
        if self.scores is not None:
            if len(self.scores) != len(self.codes):
                raise ValueError(
                    f"note {self.note_id}: n_codes={len(self.codes)} "
                    f"!= n_scores={len(self.scores)}"
                )


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

_ID_CANDIDATES = ("NOTE_ID", "note_id", "note_index")
_TEXT_CANDIDATES = ("NOTE_TEXT", "note_text", "text")


def _pick_col(df: pd.DataFrame, candidates: Sequence[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find {what} column; looked for {candidates}; "
        f"have {list(df.columns)}"
    )


def load_notes_for_prediction(
    csv_path: Path | str,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a notes CSV and return a DataFrame with two guaranteed columns:
    ``note_id`` (str) and ``note_text`` (str).  Other columns are preserved
    untouched so downstream code can grab them if needed.
    """
    df = pd.read_csv(csv_path, dtype=str)
    id_col = note_id_col or _pick_col(df, _ID_CANDIDATES, "note-id")
    text_col = note_text_col or _pick_col(df, _TEXT_CANDIDATES, "note-text")

    df = df.rename(columns={id_col: "note_id", text_col: "note_text"})
    df["note_id"] = df["note_id"].astype(str).str.strip()
    df["note_text"] = df["note_text"].fillna("").astype(str)
    return df


# ---------------------------------------------------------------------------
# Unified input policy
# ---------------------------------------------------------------------------

#: The one note budget every system in the benchmark honours, in cl100k
#: tokens.  4096 is not a round number picked for looks — it is the largest
#: budget *every* component can actually hold:
#:
#:   * M2's Longformer encodes 4096 positions natively (its architectural max);
#:   * M5's SFT trainer is sized for one 80 GB H100 near this length;
#:   * the verifier's snippetizer covers 32 x 180 words, comfortably past it;
#:   * kb_index's cross-encoder holds only 512, so it covers 4096 as 8 chunks
#:     rather than truncating (--query-chunks / --rerank-chunks).
#:
#: Above 4096 the Longformer would start truncating while the API models did
#: not, which is exactly the incompatibility this policy exists to remove.
#: On the operative-note corpus it truncates 1.5-3.0 % of notes.
UNIFIED_NOTE_BUDGET = 4096


def log_note_budget(
    system: str,
    max_note_tokens: Optional[int],
    input_kind: str,
    covered_by: Optional[str] = None,
) -> Dict[str, object]:
    """Emit one uniform, greppable line describing this run's input policy.

    ``paper-ready`` means a reader can *verify* that a comparison table is
    input-matched instead of trusting it.  Every baseline prints this line, so
    ``grep 'NOTE BUDGET' `` across a run's logs is the audit.

    ``input_kind`` is ``whole-note`` or ``sections:<n>``.  ``covered_by``
    names the mechanism when the encoder cannot hold the budget in one pass
    (e.g. ``"8 chunks x 512"``).
    """
    ok = (max_note_tokens is not None and
          int(max_note_tokens) == UNIFIED_NOTE_BUDGET)
    verdict = "MATCHES-POLICY" if ok else "DEVIATES-FROM-POLICY"
    LOGGER.info(
        "NOTE BUDGET | system=%s | input=%s | max_note_tokens=%s | "
        "unified=%d | %s%s",
        system, input_kind, max_note_tokens, UNIFIED_NOTE_BUDGET, verdict,
        f" | covered_by={covered_by}" if covered_by else "",
    )
    if not ok:
        LOGGER.warning(
            "%s is NOT on the unified input policy (%s tokens vs %d). Any "
            "cross-system table including this run differs in INPUT as well "
            "as in method -- say so, or re-run at %d.",
            system, max_note_tokens, UNIFIED_NOTE_BUDGET, UNIFIED_NOTE_BUDGET,
        )
    if input_kind != "whole-note":
        LOGGER.warning(
            "%s reads %s, not the whole note. The unified policy is the WHOLE "
            "NOTE -- it is the only input both corpora can share (MIMIC "
            "discharge summaries have no sections).", system, input_kind,
        )
    return {
        "system": system, "input": input_kind,
        "max_note_tokens": max_note_tokens,
        "unified_note_budget": UNIFIED_NOTE_BUDGET,
        "matches_policy": bool(ok and input_kind == "whole-note"),
    }


# ---------------------------------------------------------------------------
# Code list post-filtering
# ---------------------------------------------------------------------------

def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def filter_to_known_codes(
    codes: Sequence[str],
    known: Set[str],
    scores: Optional[Sequence[float]] = None,
) -> Tuple[List[str], Optional[List[float]]]:
    """Drop any code not present in the KB."""
    out_codes: List[str] = []
    out_scores: List[float] = [] if scores is not None else None  # type: ignore
    for i, c in enumerate(codes):
        c_up = c.strip().upper()
        if c_up in known:
            out_codes.append(c_up)
            if out_scores is not None:
                out_scores.append(float(scores[i]))
    return out_codes, out_scores


# ---------------------------------------------------------------------------
# Budget-matched shortlists for generative baselines
# ---------------------------------------------------------------------------

def rank_by_self_consistency(
    samples: Sequence[Sequence[str]],
) -> Tuple[List[str], List[float]]:
    """Rank codes by how often independent samples agree on them.

    A generative baseline emits a *list*, not a distribution, so its ranking
    has previously been generation order carrying synthetic scores
    (``len(codes) - i``).  Sampling the same prompt ``n`` times and scoring
    each code by the fraction of samples containing it yields a genuine
    marginal-probability estimate from the generator itself -- no retriever
    and no external pool -- which is what makes R@B commensurable with a
    scorer's ``sigmoid`` output.

    Ties (common at small ``n``) break on mean within-sample rank, so a code
    the model consistently names first outranks one it consistently names
    last at the same frequency.

    Returns ``(codes, scores)`` best-first with ``scores`` in ``(0, 1]``.
    """
    n = len([s for s in samples])
    if n == 0:
        return [], []
    hits: Dict[str, int] = {}
    rank_sum: Dict[str, float] = {}
    for sample in samples:
        for pos, code in enumerate(dedupe_keep_order(
                [c.strip().upper() for c in sample if c and str(c).strip()])):
            hits[code] = hits.get(code, 0) + 1
            rank_sum[code] = rank_sum.get(code, 0.0) + float(pos)
    ordered = sorted(
        hits,
        key=lambda c: (-hits[c] / n, rank_sum[c] / hits[c], c),
    )
    return ordered, [hits[c] / n for c in ordered]


def complete_shortlist(
    codes: Sequence[str],
    scores: Sequence[float],
    k: int,
    pad_pool: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[float]]:
    """Trim to ``k``, and -- only when ``pad_pool`` is given -- pad up to ``k``.

    Padding is **off by default and never implicit**.  A model that returns
    fewer than ``k`` codes is otherwise reported at the size it actually
    emitted, because filling the tail from a retrieved pool measures the
    retriever rather than the generator (the long-standing M4 rule).  When a
    caller does opt in, every padded code is scored strictly below every
    generated one, so padding can only ever occupy the tail of the ranking
    and can never displace or re-order what the model itself produced.

    The returned list may still be shorter than ``k`` if the pool is
    exhausted; callers should report the achieved cardinality rather than
    assume the budget was filled.
    """
    k = int(k)
    out_c = [c.strip().upper() for c in codes][:k]
    out_s = [float(x) for x in scores][:k]
    if pad_pool is None or len(out_c) >= k:
        return out_c, out_s
    floor = min(out_s) if out_s else 1.0
    seen = set(out_c)
    for j, cand in enumerate(pad_pool):
        if len(out_c) >= k:
            break
        cu = cand.strip().upper()
        if not cu or cu in seen:
            continue
        seen.add(cu)
        out_c.append(cu)
        # Strictly below the generated tier, and monotonically decreasing in
        # pool order so the retriever's own ranking survives inside the pad.
        out_s.append(floor * 1e-3 / (j + 1))
    return out_c, out_s


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_predictions(
    predictions: Iterable[Prediction],
    out_csv: Path | str,
    include_scores: bool = True,
) -> int:
    """
    Write predictions in the harness-compatible format.

    Atomic on POSIX: we write to ``<out_csv>.tmp`` in the same directory and
    rename on success, so a checkpointed run that gets interrupted mid-flush
    cannot leave behind a half-written CSV.

    Read-side convention: consumers should always read these CSVs with
    ``pd.read_csv(path, dtype=str).fillna("")``.  Without that, pandas can
    infer ``pred_codes`` as ``float64`` whenever every non-empty value
    happens to look numeric (e.g. all 5-digit CPT codes plus an empty
    cell), which silently turns ``"43235"`` into ``"43235.0"``.

    Returns the number of rows written.
    """
    rows: List[Dict[str, str]] = []
    for pred in predictions:
        codes_str = "|".join(pred.codes)
        row: Dict[str, str] = {"note_id": pred.note_id, "pred_codes": codes_str}
        if include_scores and pred.scores is not None:
            row["pred_scores"] = "|".join(f"{s:.6f}" for s in pred.scores)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=out_csv.name + ".",
        suffix=".tmp",
        dir=str(out_csv.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_path)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_csv)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    LOGGER.info("Wrote %d predictions -> %s", len(df), out_csv)
    return len(df)


# ---------------------------------------------------------------------------
# Full-pool score dump (ranking metrics)
# ---------------------------------------------------------------------------

def write_scores_npz(
    rankings: Iterable[Tuple[str, Sequence[str], Sequence[float]]],
    out_npz: Path | str,
) -> int:
    """
    Write a ragged ``note_id -> (candidate_codes, probs)`` NPZ.

    This is the *pre-threshold* candidate pool with its scores, i.e. exactly
    what ``cptrec-verifier-predict --save-npz`` writes for the verifier, so that
    ``cptrec-evaluate --scores-npz`` computes the identical shortlist-review
    suite (R@B / Coverage@B / FamilyMRR / ...) for a baseline.  Emitting only
    the thresholded prediction set would cap R@B at the baseline's own
    operating point, which is the comparison the review-budget metrics exist
    to avoid.

    Returns the number of notes written.
    """
    import numpy as np

    note_ids: List[str] = []
    cand: List[object] = []
    probs: List[object] = []
    for note_id, codes, scores in rankings:
        note_ids.append(str(note_id))
        cand.append(np.array([str(c) for c in codes], dtype=object))
        probs.append(np.array([float(s) for s in scores], dtype=np.float32))

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        note_ids=np.array(note_ids, dtype=object),
        candidate_codes=np.array(cand, dtype=object),
        probs=np.array(probs, dtype=object),
    )
    pool = [len(c) for c in cand]
    LOGGER.info(
        "Wrote score NPZ: %d notes, pool mean %.1f / min %d / max %d -> %s",
        len(note_ids),
        (sum(pool) / len(pool)) if pool else 0.0,
        min(pool) if pool else 0,
        max(pool) if pool else 0,
        out_npz,
    )
    return len(note_ids)


# ---------------------------------------------------------------------------
# Deterministic ordering, --limit slicing, and prediction-stats logging
# ---------------------------------------------------------------------------

def apply_seed_and_limit(
    df: pd.DataFrame,
    seed: int = 42,
    limit: Optional[int] = None,
    sort_col: str = "note_id",
) -> pd.DataFrame:
    """
    Sort the input DataFrame by *sort_col* and optionally slice the first
    ``limit`` rows.

    Sorting (rather than shuffling) is deliberate: a checkpointed long run
    that is interrupted at row N must resume from the same row N on
    re-launch, regardless of pandas/CSV iteration order.  Pass ``--limit``
    on the CLI for smoke tests; pass ``--seed`` for parity with the rest
    of the pipeline (currently unused but reserved for future shuffles).
    """
    if sort_col not in df.columns:
        raise ValueError(f"sort column {sort_col!r} not present in DataFrame")
    out = df.sort_values(sort_col, kind="mergesort").reset_index(drop=True)
    if limit is not None and limit > 0 and limit < len(out):
        out = out.iloc[:limit].copy()
    LOGGER.info(
        "apply_seed_and_limit: %d rows -> %d rows (sort=%s, seed=%d, limit=%s)",
        len(df), len(out), sort_col, seed, limit,
    )
    return out


# ---------------------------------------------------------------------------
# Local-model resolver  (offline / disconnected-server support)
# ---------------------------------------------------------------------------
#
# The training server has no Hugging Face Hub connectivity, so every
# ``from_pretrained(<repo_id>)`` call must be redirected to a copy of the
# model that lives under a local cache directory.  The standard layout we
# support mirrors the Hub:
#
#     models/yikuan8/Clinical-Longformer/                 (full repo path)
#     models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/
#     models/cross-encoder/ms-marco-MiniLM-L-6-v2/
#
# We also accept the flat ``git clone``-style layout where only the
# basename survives:
#
#     models/Clinical-Longformer/
#     models/SapBERT-from-PubMedBERT-fulltext/
#     models/ms-marco-MiniLM-L-6-v2/
#
# A model is considered "present" when its directory contains either
# ``config.json`` (the standard HF marker file) or ``model.safetensors`` /
# ``pytorch_model.bin``.  When neither layout matches we fall back to the
# original name, so a developer with internet still gets the Hub path.
#
# The directory is configurable via:
#   * the function arg ``models_dir=`` (highest precedence),
#   * the ``CPT_REC_LOCAL_MODELS_DIR`` env var,
#   * the project default ``models/``.
#
# Set ``CPT_REC_REQUIRE_LOCAL_MODELS=1`` (or pass ``strict=True``) to make the
# resolver raise ``FileNotFoundError`` when no local copy is found —
# useful as a guard on a known-offline machine so HF doesn't mask the
# real cause with a vague network error.

_DEFAULT_MODELS_DIR_ENV = "CPT_REC_LOCAL_MODELS_DIR"
_REQUIRE_LOCAL_ENV = "CPT_REC_REQUIRE_LOCAL_MODELS"


def _looks_like_model_dir(p: Path) -> bool:
    """
    Permissive check: any directory is acceptable.

    We deliberately do NOT require ``config.json`` / ``model.safetensors``
    / ``pytorch_model.bin`` at the top of the dir, because:

    * tokenizer-only dirs (used by some pipelines) lack ``config.json``;
    * sentencepiece-based snapshots may only ship ``tokenizer.model``;
    * cross-encoders sometimes nest weights under ``0_Transformer/``.

    If the dir is missing real model files, HF will raise a clean error
    with the actual missing-file path — which is better than us silently
    falling through to the Hub and producing an SSL/timeout cascade.
    """
    return p.is_dir()


def force_offline_hf(why: str = "") -> None:
    """
    Make Hugging Face refuse to touch the network from this process on.

    Idempotent — safe to call multiple times.  Setting these env vars
    *after* huggingface_hub has been imported is still effective for
    subsequent network calls because the lib reads them on each request.
    """
    set_now = []
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(k) not in ("1", "true", "True"):
            os.environ[k] = "1"
            set_now.append(k)
    if set_now:
        LOGGER.info(
            "Forcing HF offline mode%s: set %s",
            f" ({why})" if why else "",
            ", ".join(set_now),
        )


def _list_models_dir_inventory(cache_root: Path, limit: int = 50) -> str:
    """One-line-per-dir snapshot of *cache_root* for error messages."""
    if not cache_root.exists():
        return f"  (cache dir does not exist: {cache_root})"
    if not cache_root.is_dir():
        return f"  (not a directory: {cache_root})"
    entries: List[str] = []
    for child in sorted(cache_root.iterdir()):
        if not child.is_dir():
            continue
        # Show one level of nested children too, since HF repo-id layout
        # ("org/model/") puts the real snapshot one level deeper.
        nested = [c.name for c in sorted(child.iterdir()) if c.is_dir()]
        if nested:
            entries.append(f"  {child.name}/  -> [{', '.join(nested[:5])}{'...' if len(nested) > 5 else ''}]")
        else:
            entries.append(f"  {child.name}/")
        if len(entries) >= limit:
            entries.append(f"  ... ({len(list(cache_root.iterdir())) - limit} more)")
            break
    if not entries:
        return f"  (empty: {cache_root})"
    return "\n".join(entries)


def resolve_local_model(
    name_or_path: str,
    models_dir: Optional[Path | str] = None,
    *,
    strict: Optional[bool] = None,
) -> str:
    """
    Map a model identifier to a local directory path when one is
    available; otherwise return ``name_or_path`` unchanged.

    Resolution order (first hit wins):

    1. ``name_or_path`` is already an existing directory.
    2. ``<models_dir>/<name_or_path>`` (HF repo-id layout, e.g.
       ``models/cambridgeltl/SapBERT-...``).
    3. ``<models_dir>/<basename>`` (flat layout, e.g.
       ``models/SapBERT-...``).

    When ``models_dir`` is supplied explicitly (caller passed
    ``--local-models-dir``), we treat that as a clear "offline intent"
    signal and:

    * default ``strict`` to True so a name mismatch fails fast, and
    * call :func:`force_offline_hf` so HF won't fall through to the Hub
      even if a downstream loader sidesteps the resolver.

    Strict mode raises ``FileNotFoundError`` listing every path tried
    *and* an inventory of ``models_dir`` so the typo is obvious.
    """
    # Detect whether the caller explicitly provided a cache dir.
    explicit_dir = models_dir is not None

    # 1) Already a local directory?  Honor it.
    p = Path(name_or_path)
    if p.exists() and _looks_like_model_dir(p):
        return str(p)

    # 2) Resolve the cache root.
    if models_dir is None:
        models_dir = os.environ.get(_DEFAULT_MODELS_DIR_ENV, "models")
    cache_root = Path(models_dir)

    # If the caller pinned a local cache, force HF offline immediately —
    # before HF gets a chance to try the network during from_pretrained.
    if explicit_dir:
        force_offline_hf(why=f"--local-models-dir={cache_root}")

    # 3) Try repo-id layout, then basename layout.
    candidates = [cache_root / name_or_path]
    base = name_or_path.rsplit("/", 1)[-1]
    if base != name_or_path:
        candidates.append(cache_root / base)
    for cand in candidates:
        if _looks_like_model_dir(cand):
            LOGGER.info(
                "resolve_local_model: %r -> %s (local cache hit)",
                name_or_path, cand,
            )
            return str(cand)

    # 4) Strict mode → fail fast with a helpful message.
    require_local = strict
    if require_local is None:
        env_strict = os.environ.get(_REQUIRE_LOCAL_ENV, "").lower() in {"1", "true", "yes"}
        require_local = env_strict or explicit_dir
    if require_local:
        tried = "\n  ".join(str(c) for c in candidates)
        inventory = _list_models_dir_inventory(cache_root)
        raise FileNotFoundError(
            f"resolve_local_model: no local copy of {name_or_path!r} found.\n"
            f"  Tried (none of these is a directory):\n  {tried}\n"
            f"  Available under {cache_root}:\n{inventory}\n"
            f"  Pass the correct basename via --biencoder / --reranker / "
            f"--base-model / --encoder, or rename the directory to match."
        )

    # 5) Pass-through for the developer-laptop case (Hub still reachable).
    LOGGER.info(
        "resolve_local_model: %r not in local cache (%s); falling back to "
        "Hugging Face Hub", name_or_path, cache_root,
    )
    return name_or_path


# ---------------------------------------------------------------------------
# Temporal alignment helpers (CodeHistory crosswalk + active-codes restriction)
# ---------------------------------------------------------------------------

def maybe_load_code_history(
    history_changes: Optional[Path | str],
    history_deleted: Optional[Path | str],
    kb_csv: Optional[Path | str] = None,
    active_kb_codes: Optional[Iterable[str]] = None,
):
    """
    Load a ``CodeHistory`` if both history CSVs are provided; else None.

    Returns a ``(history, active_kb_codes)`` tuple.  ``active_kb_codes``
    is the frozen set of codes in the 2026 KB — needed by
    ``CodeHistory`` to distinguish "has no events, always active in
    2026" from "never heard of".

    Either source of the active code set is accepted:

    * ``active_kb_codes`` — when the caller already has them in memory
      (e.g. kb_index has them baked into its retrieval-index artifact, so
      reloading the KB CSV is wasteful and the CLI doesn't even need a
      ``--kb`` flag);
    * ``kb_csv`` — when only the path is available (M1, M4).

    When both are absent, the function raises with a message that names
    *both* possible inputs so the caller can pick the one that matches
    their data flow.

    When ``history_changes`` and ``history_deleted`` are both None,
    returns ``(None, None)`` so callers can opt out cheaply.
    """
    if not (history_changes and history_deleted):
        return None, None
    from cpt_rec.common.knowledge.code_history import CodeHistory

    if active_kb_codes is not None:
        active = frozenset(active_kb_codes)
    elif kb_csv is not None:
        from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
        kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
        active = frozenset(kb.codes)
    else:
        raise ValueError(
            "maybe_load_code_history needs either active_kb_codes "
            "(in-memory) or kb_csv (path) when history is enabled"
        )
    history = CodeHistory.from_csvs(
        changes_csv=Path(history_changes),
        deleted_csv=Path(history_deleted),
        active_kb_codes=active,
    )
    LOGGER.info(
        "Loaded CodeHistory: %d total codes (%d outside active 2026 KB)",
        len(history.all_codes()),
        len(history.all_codes()) - len(active),
    )
    return history, active


def parse_as_of(value) -> Optional["date"]:
    """Coerce a date-column cell to ``datetime.date`` or None."""
    from datetime import date  # local import to keep top-level cheap
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.date()


def crosswalk_codes_for_date(
    codes: Iterable[str],
    history,
    as_of,
    keep_unresolved: bool = True,
) -> Set[str]:
    """
    Map a code set forward through ``history`` to its date-aligned form.

    For every code:
      * if active on ``as_of`` → kept as-is,
      * elif a substitute is resolvable on ``as_of`` → substituted,
      * else → dropped (or kept verbatim if ``keep_unresolved=True``).

    Used by M1/M4 to substitute retired training-neighbor labels into the
    target note's date frame before voting / candidate construction.
    """
    if history is None or as_of is None:
        return set(codes)
    out: Set[str] = set()
    for c in codes:
        if not c:
            continue
        if history.is_active(c, as_of):
            out.add(c)
            continue
        sub = history.crosswalk(c, as_of)
        if sub is not None:
            out.add(sub)
        elif keep_unresolved:
            out.add(c)
    return out


def restrict_to_active(
    codes: Iterable[str],
    history,
    as_of,
) -> Set[str]:
    """
    Filter ``codes`` down to those active on ``as_of`` (no substitution).

    Used by kb_index to restrict the BM25-∪-dense candidate pool per-query so
    a 2018 note doesn't get scored against codes that didn't exist yet.
    Returns ``set(codes)`` unchanged when history or as_of are absent.
    """
    if history is None or as_of is None:
        return set(codes)
    return {c for c in codes if c and history.is_active(c, as_of)}


#: Below this achieved/asked ratio a --shortlist-k run is reported as a
#: failed budget rather than a matched-budget result.
_FILL_WARN = 0.80


def stats_sidecar(out_csv: Path) -> Path:
    """
    Where a run's aggregate stats go: ``<out>_stats.json`` beside ``<out>.csv``.

    The ``_stats.json`` name is load-bearing.  ``.gitignore`` rule 3a
    re-includes ``outputs/**/*stats.json`` by NAME wherever it sits, so the
    sidecar syncs from a VUMC run automatically, and on MIMIC it is the file
    ``cptrec-bench-export --aux`` lifts out of the restricted tree.  Renaming it
    silently stops both.
    """
    return Path(out_csv).with_name(Path(out_csv).stem + "_stats.json")


def log_prediction_stats(
    predictions: Sequence["Prediction"],
    label: str = "predictions",
    budget_k: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Log summary stats over a list of Prediction objects: total notes,
    total codes emitted, mean/median codes per note, and number of empty
    predictions.  Returns the same dict for caller consumption.

    ``budget_k`` is the cardinality the run *asked* for (``--shortlist-k``).
    Passing it adds a **budget-fill audit**: the mean achieved cardinality as
    a fraction of ``k``, and the share of notes that actually reached ``k``.
    A matched-budget claim is only meaningful when the fill rate is high, and
    a model that quietly returns 4 codes under ``k=10`` otherwise looks
    healthy in every downstream metric -- its R@8, R@10 and R@20 are simply
    the same saturated number.  Below ``_FILL_WARN`` the audit escalates to a
    WARNING that names the consequence, so the shortfall cannot pass
    unnoticed a second time.  Omitting ``budget_k`` leaves behaviour and the
    returned dict byte-identical to before.

    ``out_path`` writes the same dict as JSON.  Every caller used to discard
    the return value, so cardinality and budget fill existed only in a
    terminal -- which on a restricted corpus means they existed only on the
    server, and reached the paper by being retyped.  The sidecar is a pure
    aggregate (counts and means over the PREDICTIONS, never gold), so it
    crosses the DUA boundary through ``cptrec-bench-export --aux``.  Omitting it
    leaves behaviour byte-identical.
    """
    n_notes = len(predictions)
    counts = [len(p.codes) for p in predictions]
    total_codes = sum(counts)
    n_empty = sum(1 for c in counts if c == 0)
    mean_codes = (total_codes / n_notes) if n_notes else 0.0
    sorted_counts = sorted(counts)
    median_codes = (
        float(sorted_counts[n_notes // 2])
        if n_notes
        else 0.0
    )
    stats = {
        "n_notes": float(n_notes),
        "total_codes": float(total_codes),
        "mean_codes_per_note": round(mean_codes, 3),
        "median_codes_per_note": median_codes,
        "n_empty_predictions": float(n_empty),
        "pct_empty_predictions": round(100.0 * n_empty / max(1, n_notes), 2),
    }
    LOGGER.info(
        "[%s] n_notes=%d, total_codes=%d, mean=%.2f, median=%g, "
        "empty=%d (%.1f%%)",
        label,
        n_notes,
        total_codes,
        mean_codes,
        median_codes,
        n_empty,
        stats["pct_empty_predictions"],
    )
    if budget_k:
        k = int(budget_k)
        n_at_k = sum(1 for c in counts if c >= k)
        fill = mean_codes / k if k else 0.0
        stats.update({
            "budget_k": float(k),
            "budget_fill_rate": round(fill, 4),
            "n_notes_at_budget": float(n_at_k),
            "pct_notes_at_budget": round(100.0 * n_at_k / max(1, n_notes), 2),
        })
        msg = ("[%s] budget audit: asked k=%d, achieved mean=%.2f "
               "(fill %.0f%%), %d/%d notes (%.1f%%) reached k")
        args = (label, k, mean_codes, 100.0 * fill, n_at_k, n_notes,
                stats["pct_notes_at_budget"])
        if fill < _FILL_WARN:
            LOGGER.warning(msg, *args)
            LOGGER.warning(
                "[%s] the budget was NOT met. R@B above the achieved "
                "cardinality is saturated, not measured -- any matched-budget "
                "comparison against a ranker that fills k is invalid. Quote "
                "shown_at beside every R@B, and see --pad-to-k (retrieval "
                "baselines) or --self-consistency N (generative baselines).",
                label,
            )
        else:
            LOGGER.info(msg, *args)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"label": label, **stats}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("[%s] prediction stats -> %s", label, out)
    return stats


# ---------------------------------------------------------------------------
# Pre-trimming for long notes (token-budget helper)
# ---------------------------------------------------------------------------

def truncate_text_by_tokens(
    text: str,
    max_tokens: int = 2048,
    encoding_name: str = "cl100k_base",
) -> str:
    """
    Truncate *text* to roughly ``max_tokens`` tokens using tiktoken if
    available, else fall back to whitespace word count.  Returns the prefix
    that fits.
    """
    if not text:
        return ""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        toks = enc.encode(text, disallowed_special=())
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        # Word fallback — approximate tokens as 0.75 * words
        words = text.split()
        approx = int(max_tokens / 0.75)
        if len(words) <= approx:
            return text
        return " ".join(words[:approx])
