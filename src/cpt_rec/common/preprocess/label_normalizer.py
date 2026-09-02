#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Normalize CPT_CODES labels and produce code-frequency / drift statistics.

Three CLI modes
---------------
``normalize``
    Read a split CSV (train / val / test / drift), strip modifiers,
    deduplicate per note, filter short notes, write a cleaned CSV.

``stats``
    Read a *normalized* training CSV, compute per-code frequency statistics,
    write ``code_frequency_stats.csv``.

``drift``
    Read normalized val / test / drift CSVs plus the training stats, annotate
    every code as ``seen_in_train`` or ``unseen_in_train``, and flag codes
    absent from the 2025 knowledge base.

All three modes can be run sequentially or independently.
"""

from __future__ import annotations

import argparse
import functools
import logging
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from cpt_rec.common.knowledge.code_history import CodeHistory
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.common.preprocess.code_utils import (
    normalize_code_list,
    strip_modifier,
)

LOGGER = logging.getLogger(__name__)

# Default tiktoken encoding.  ``cl100k_base`` is used by GPT-3.5 / GPT-4
# and matches the tokenizer most LLM length limits are quoted against.
# Override via the ``CPT_REC_TIKTOKEN_ENCODING`` env var if needed.
_DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _get_tiktoken_encoder(encoding_name: str = _DEFAULT_TIKTOKEN_ENCODING):
    """
    Lazily load (and cache) a tiktoken encoder.

    Returns ``None`` if tiktoken is not installed, so callers can gracefully
    fall back to whitespace counting.
    """
    try:
        import tiktoken
    except ImportError:
        LOGGER.warning(
            "tiktoken not installed; falling back to whitespace word count. "
            "Install with `pip install tiktoken` for precise token counts."
        )
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # pragma: no cover - network / unknown encoding
        LOGGER.warning("tiktoken encoder '%s' unavailable (%s); "
                       "falling back to whitespace word count.",
                       encoding_name, exc)
        return None


def _word_count(text: str, encoding_name: str = _DEFAULT_TIKTOKEN_ENCODING) -> int:
    """
    Precise token count via tiktoken, with whitespace fallback.

    Uses the ``cl100k_base`` encoding by default (GPT-3.5 / GPT-4 tokenizer),
    which aligns with typical LLM context-window accounting.  If tiktoken is
    not installed or the encoder fails to load, falls back to whitespace
    word count so the pipeline stays functional.
    """
    if not text:
        return 0
    enc = _get_tiktoken_encoder(encoding_name)
    if enc is None:
        return len(text.split())
    # ``disallowed_special=()`` prevents tiktoken from raising on strings
    # that happen to contain the special-token sentinels like ``<|endoftext|>``.
    return len(enc.encode(text, disallowed_special=()))


def _load_kb_codes(kb_csv_path: Path) -> Set[str]:
    """Load the set of valid codes from the knowledge-base CSV."""
    df = pd.read_csv(kb_csv_path, usecols=["code"])
    return set(df["code"].astype(str).str.strip().str.upper())


# ---------------------------------------------------------------------------
# Temporal enrichment (CodeHistory-aware)
# ---------------------------------------------------------------------------

def _parse_procedure_date(value: object) -> Optional[date]:
    """Best-effort ``PROCEDURE_DATE`` parser.

    Accepts pandas ``Timestamp``, ``datetime`` / ``date``, and strings
    in any format pandas' ``to_datetime`` can infer (ISO, US-style,
    ``D-MMM-YY``, …).  Returns ``None`` for NaN / NaT / pd.NA / empty
    / unparseable values, and callers treat that as a passthrough
    (no temporal gating applied).
    """
    if value is None:
        return None
    # Catch NaN, NaT, pd.NA in one shot.  ``pd.NaT`` is a subclass of
    # :class:`datetime.datetime` in modern pandas, so a plain
    # ``isinstance(value, datetime)`` branch below would otherwise
    # return ``pd.NaT.date() == NaT`` and leak a non-``None`` sentinel.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # ``pd.isna`` can raise on exotic / unhashable inputs; fall
        # through to the type-specific branches so we still make a
        # best effort at parsing them.
        pass
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    return None


def _load_code_history(
    changes_csv: Path,
    deleted_csv: Path,
    kb_csv: Path,
) -> CodeHistory:
    """Convenience loader used by both the inline and standalone paths."""
    kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
    return CodeHistory.from_csvs(
        changes_csv=changes_csv,
        deleted_csv=deleted_csv,
        active_kb_codes=kb.codes,
    )


def enrich_rows_with_history(
    df: pd.DataFrame,
    history: CodeHistory,
    proc_codes_col: str = "proc_codes",
    date_col: str = "PROCEDURE_DATE",
) -> pd.DataFrame:
    """Add four temporal columns derived from ``history``.

    For every row we read ``proc_codes`` (pipe-separated) and
    ``PROCEDURE_DATE``, then produce:

    - ``proc_codes_valid_on_date``  – subset of ``proc_codes`` that were
      still active on ``PROCEDURE_DATE``, pipe-joined.
    - ``proc_codes_crosswalked``    – one substitute per original code:
      active codes pass through unchanged; deleted codes are replaced
      by the substitute that's active on ``PROCEDURE_DATE`` (chased
      transitively through chains); codes whose crosswalk is
      unresolvable are dropped from this column.
    - ``n_codes_retired_on_date``   – count of original codes that were
      inactive on ``PROCEDURE_DATE``.
    - ``n_codes_crosswalked``       – count of retired codes that had a
      resolvable substitute (``n_codes_retired_on_date`` ≥ this).

    If ``PROCEDURE_DATE`` is missing / unparseable for a row we can't
    temporally gate: we passthrough (``proc_codes_valid_on_date =
    proc_codes_crosswalked = proc_codes``) and set both counts to 0.
    The caller can detect these rows by comparing to
    ``history_rows_with_unparseable_date`` in the summary counters (or
    by checking whether the raw ``PROCEDURE_DATE`` column is null).

    The original ``proc_codes`` column is preserved; this is additive.
    """
    valid_list: List[str] = []
    xwalked_list: List[str] = []
    retired_counts: List[int] = []
    xwalked_counts: List[int] = []

    # Pull the columns once to avoid per-row .get() overhead in iterrows
    codes_series = df[proc_codes_col] if proc_codes_col in df.columns else pd.Series([""] * len(df))
    dates_series = df[date_col] if date_col in df.columns else pd.Series([None] * len(df))

    for raw_codes, raw_date in zip(codes_series.fillna("").astype(str), dates_series):
        codes = [c for c in raw_codes.split("|") if c]
        proc_date = _parse_procedure_date(raw_date)

        if proc_date is None:
            # Passthrough: we can't temporally gate, don't pretend we can.
            valid_list.append("|".join(codes))
            xwalked_list.append("|".join(codes))
            retired_counts.append(0)
            xwalked_counts.append(0)
            continue

        valid: List[str] = []
        xwalked: List[str] = []
        n_retired = 0
        n_xwalked = 0
        for code in codes:
            if history.is_active(code, proc_date):
                valid.append(code)
                xwalked.append(code)
            else:
                n_retired += 1
                sub = history.crosswalk(code, proc_date)
                if sub is not None:
                    xwalked.append(sub)
                    n_xwalked += 1
                # If sub is None, the code is dropped from the xwalked list.

        valid_list.append("|".join(valid))
        xwalked_list.append("|".join(xwalked))
        retired_counts.append(n_retired)
        xwalked_counts.append(n_xwalked)

    out = df.copy()
    out["proc_codes_valid_on_date"] = valid_list
    out["proc_codes_crosswalked"] = xwalked_list
    out["n_codes_retired_on_date"] = retired_counts
    out["n_codes_crosswalked"] = xwalked_counts
    return out


def enrich_csv_with_history(
    input_csv: Path,
    output_csv: Path,
    history: CodeHistory,
    proc_codes_col: str = "proc_codes",
    date_col: str = "PROCEDURE_DATE",
    chunksize: int = 50_000,
) -> Dict[str, int]:
    """Standalone: read a *normalized* CSV, run :func:`enrich_rows_with_history`
    on every chunk, write to ``output_csv`` (which may equal ``input_csv``).

    Returns the same shape of counter dict as :func:`normalize_split`
    but scoped to the enrichment pass (no tokenization, no code
    validation).  Keys:

        rows_in, rows_with_unparseable_date,
        total_codes_retired_on_date, total_codes_crosswalked,
        total_codes_dropped_no_crosswalk.
    """
    counters = {
        "rows_in": 0,
        "rows_with_unparseable_date": 0,
        "total_codes_retired_on_date": 0,
        "total_codes_crosswalked": 0,
        "total_codes_dropped_no_crosswalk": 0,
    }

    first_chunk = True
    reader = pd.read_csv(input_csv, chunksize=chunksize, dtype=str)
    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]
        counters["rows_in"] += len(chunk)

        # Track unparseable-date rows BEFORE enrichment overwrites the
        # signal.  A row is unparseable iff pd.to_datetime returns NaT.
        if date_col in chunk.columns:
            parsed = pd.to_datetime(chunk[date_col], errors="coerce")
            counters["rows_with_unparseable_date"] += int(parsed.isna().sum())
        else:
            counters["rows_with_unparseable_date"] += len(chunk)

        chunk = enrich_rows_with_history(
            chunk, history,
            proc_codes_col=proc_codes_col,
            date_col=date_col,
        )
        counters["total_codes_retired_on_date"] += int(
            chunk["n_codes_retired_on_date"].sum()
        )
        counters["total_codes_crosswalked"] += int(
            chunk["n_codes_crosswalked"].sum()
        )
        counters["total_codes_dropped_no_crosswalk"] += int(
            (chunk["n_codes_retired_on_date"] - chunk["n_codes_crosswalked"]).sum()
        )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        chunk.to_csv(
            output_csv,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

    return counters


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def normalize_split(
    input_csv: Path,
    output_csv: Path,
    kb_codes: Optional[Set[str]] = None,
    min_tokens: int = 100,
    token_col: Optional[str] = "NOTE_TEXT_TOKENS",
    note_text_col: str = "NOTE_TEXT",
    code_col: str = "CPT_CODES",
    chunksize: int = 50_000,
    history: Optional[CodeHistory] = None,
    date_col: str = "PROCEDURE_DATE",
) -> Dict[str, int]:
    """
    Normalize one split CSV.

    For each row:
    1. Parse ``CPT_CODES``, strip modifiers, deduplicate.
    2. Optionally validate against *kb_codes*.
    3. Filter notes with fewer than *min_tokens* tokens.
    4. Optionally enrich with temporal info via *history* (if provided).

    Writes a new CSV with added columns:

    - ``CPT_CODES_RAW``: original string
    - ``proc_codes``: pipe-separated accepted base codes
    - ``n_codes``: number of accepted codes
    - ``codes_unresolved``: pipe-separated codes valid in format but not in KB
    - ``codes_dropped``: pipe-separated tokens that couldn't be parsed

    When ``history`` is provided, four more columns are added
    (see :func:`enrich_rows_with_history`):

    - ``proc_codes_valid_on_date``
    - ``proc_codes_crosswalked``
    - ``n_codes_retired_on_date``
    - ``n_codes_crosswalked``

    Returns a dict of summary counters.
    """
    counters = {
        "rows_in": 0,
        "rows_out": 0,
        "rows_filtered_short": 0,
        "rows_no_codes": 0,
        "total_accepted_codes": 0,
        "total_unresolved_codes": 0,
        "total_dropped_tokens": 0,
    }
    if history is not None:
        counters.update({
            "rows_with_unparseable_date": 0,
            "total_codes_retired_on_date": 0,
            "total_codes_crosswalked": 0,
            "total_codes_dropped_no_crosswalk": 0,
        })

    first_chunk = True
    reader = pd.read_csv(input_csv, chunksize=chunksize, dtype=str)

    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]
        counters["rows_in"] += len(chunk)

        # --- token filtering ---
        if token_col and token_col in chunk.columns:
            tok_counts = pd.to_numeric(chunk[token_col], errors="coerce").fillna(0)
        elif note_text_col in chunk.columns:
            texts = chunk[note_text_col].fillna("").astype(str).tolist()
            enc = _get_tiktoken_encoder()
            if enc is not None:
                # Batched tiktoken is an order of magnitude faster than row-wise.
                encoded = enc.encode_batch(texts, disallowed_special=())
                tok_counts = pd.Series([len(t) for t in encoded], index=chunk.index)
            else:
                tok_counts = pd.Series(
                    [len(t.split()) for t in texts], index=chunk.index
                )
        else:
            tok_counts = pd.Series([min_tokens] * len(chunk))  # no filtering

        keep_mask = tok_counts >= min_tokens
        n_short = (~keep_mask).sum()
        counters["rows_filtered_short"] += int(n_short)
        chunk = chunk.loc[keep_mask].copy()

        # --- code normalization ---
        raw_codes = chunk[code_col].fillna("")
        accepted_list: List[str] = []
        unresolved_list: List[str] = []
        dropped_list: List[str] = []
        n_codes_list: List[int] = []

        for raw in raw_codes:
            acc, unr, drp = normalize_code_list(raw, valid_codes=kb_codes)
            accepted_list.append("|".join(acc))
            unresolved_list.append("|".join(unr))
            dropped_list.append("|".join(drp))
            n_codes_list.append(len(acc))
            counters["total_accepted_codes"] += len(acc)
            counters["total_unresolved_codes"] += len(unr)
            counters["total_dropped_tokens"] += len(drp)

        chunk["CPT_CODES_RAW"] = chunk[code_col].values
        chunk["proc_codes"] = accepted_list
        chunk["n_codes"] = n_codes_list
        chunk["codes_unresolved"] = unresolved_list
        chunk["codes_dropped"] = dropped_list

        # Filter notes with zero accepted codes
        has_codes = chunk["n_codes"] > 0
        counters["rows_no_codes"] += int((~has_codes).sum())
        chunk = chunk.loc[has_codes]
        counters["rows_out"] += len(chunk)

        # --- temporal enrichment (CodeHistory-aware) ---
        if history is not None and len(chunk) > 0:
            # Track unparseable-date rows before enrichment so we can report
            # how many rows fell back to passthrough semantics.
            if date_col in chunk.columns:
                parsed = pd.to_datetime(chunk[date_col], errors="coerce")
                counters["rows_with_unparseable_date"] += int(parsed.isna().sum())
            else:
                counters["rows_with_unparseable_date"] += len(chunk)

            chunk = enrich_rows_with_history(
                chunk, history,
                proc_codes_col="proc_codes",
                date_col=date_col,
            )
            counters["total_codes_retired_on_date"] += int(
                chunk["n_codes_retired_on_date"].sum()
            )
            counters["total_codes_crosswalked"] += int(
                chunk["n_codes_crosswalked"].sum()
            )
            counters["total_codes_dropped_no_crosswalk"] += int(
                (chunk["n_codes_retired_on_date"] - chunk["n_codes_crosswalked"]).sum()
            )

        # Write
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        chunk.to_csv(
            output_csv,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

    return counters


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def compute_code_frequency_stats(
    train_csv: Path,
    output_csv: Path,
    code_col: str = "proc_codes",
) -> pd.DataFrame:
    """
    Compute per-code frequency statistics from the normalized training CSV.

    Reads ``proc_codes`` (pipe-separated) and counts each code.  Produces a
    DataFrame with columns: ``code, frequency, cumulative_freq, rank, bin``
    where *bin* is one of ``head`` (top 80% cumulative), ``torso``
    (80–95%), or ``tail`` (bottom 5%).
    """
    counter: Counter = Counter()
    for chunk in pd.read_csv(train_csv, usecols=[code_col], dtype=str, chunksize=50_000):
        for raw in chunk[code_col].dropna():
            codes = [c for c in str(raw).split("|") if c]
            counter.update(codes)

    if not counter:
        raise ValueError(f"No codes found in {train_csv} column '{code_col}'")

    total = sum(counter.values())
    rows = []
    cumsum = 0
    for rank, (code, freq) in enumerate(counter.most_common(), start=1):
        cumsum += freq
        cum_frac = cumsum / total
        if cum_frac <= 0.80:
            bin_label = "head"
        elif cum_frac <= 0.95:
            bin_label = "torso"
        else:
            bin_label = "tail"
        rows.append({
            "code": code,
            "frequency": freq,
            "cumulative_freq": round(cum_frac, 6),
            "rank": rank,
            "bin": bin_label,
        })

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    LOGGER.info(
        "Code stats: %d unique codes, total occurrences %d, head/torso/tail = %d/%d/%d",
        len(df),
        total,
        (df["bin"] == "head").sum(),
        (df["bin"] == "torso").sum(),
        (df["bin"] == "tail").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------

def annotate_drift(
    split_csv: Path,
    train_stats_csv: Path,
    output_csv: Path,
    kb_codes: Optional[Set[str]] = None,
    code_col: str = "proc_codes",
) -> pd.DataFrame:
    """
    Annotate each (note, code) pair as seen/unseen in training and in KB.

    Reads a normalized split CSV and the training code-frequency stats.
    Outputs a long-format CSV with one row per (note_id, code):

    - ``note_id``
    - ``code``
    - ``seen_in_train``: bool
    - ``in_knowledge_base``: bool
    - ``train_frequency``: int (0 if unseen)
    - ``train_bin``: head / torso / tail / unseen
    """
    # Load training code set
    stats = pd.read_csv(train_stats_csv, dtype={"code": str})
    train_code_freq: Dict[str, int] = dict(
        zip(stats["code"].str.strip().str.upper(), stats["frequency"])
    )
    train_code_bin: Dict[str, str] = dict(
        zip(stats["code"].str.strip().str.upper(), stats["bin"])
    )

    records = []
    note_id_col = "NOTE_ID"

    for chunk in pd.read_csv(split_csv, dtype=str, chunksize=50_000):
        chunk.columns = [c.strip() for c in chunk.columns]
        if note_id_col not in chunk.columns:
            # Fall back to first available ID column
            for candidate in ("note_index", "NOTE_ID", "ENCOUNTER_CSN_ID"):
                if candidate in chunk.columns:
                    note_id_col = candidate
                    break

        # Column-wise zip avoids per-row dict lookups + Series construction.
        # Preserve ``row.get(col, "")`` semantics if a column is missing.
        n_rows = len(chunk)
        ids = chunk[note_id_col] if note_id_col in chunk.columns else [""] * n_rows
        raws = chunk[code_col].fillna("") if code_col in chunk.columns else [""] * n_rows
        for nid, raw in zip(ids, raws):
            for code in str(raw).split("|"):
                if not code:
                    continue
                code_up = code.upper()
                freq = train_code_freq.get(code_up, 0)
                records.append({
                    "note_id": nid,
                    "code": code_up,
                    "seen_in_train": freq > 0,
                    "in_knowledge_base": (code_up in kb_codes) if kb_codes else True,
                    "train_frequency": freq,
                    "train_bin": train_code_bin.get(code_up, "unseen"),
                })

    df = pd.DataFrame.from_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    LOGGER.info(
        "Drift annotation: %d (note, code) pairs, %d unseen codes",
        len(df),
        (df["train_bin"] == "unseen").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label normalization, frequency stats, and drift annotation."
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- normalize ---
    n = sub.add_parser("normalize", help="Normalize one split CSV.")
    n.add_argument("--input", required=True, help="Input split CSV path.")
    n.add_argument("--output", required=True, help="Output normalized CSV path.")
    n.add_argument(
        "--kb-csv",
        default=None,
        help="Path to code knowledge-base CSV for validation (optional).",
    )
    n.add_argument("--min-tokens", type=int, default=100)
    n.add_argument("--code-col", default="CPT_CODES")
    n.add_argument("--note-text-col", default="NOTE_TEXT")
    n.add_argument(
        "--token-col",
        default="NOTE_TEXT_TOKENS",
        help="Column with precomputed token counts (set to '' to disable).",
    )
    n.add_argument(
        "--history-changes",
        default=None,
        help="Path to code_changes.csv. If provided (together with "
             "--history-deleted and --kb-csv), normalize adds four temporal "
             "columns gated on --date-col.",
    )
    n.add_argument(
        "--history-deleted",
        default=None,
        help="Path to deleted_codes.csv. Required for temporal enrichment.",
    )
    n.add_argument(
        "--date-col",
        default="PROCEDURE_DATE",
        help="Name of the row-level date column used for temporal gating "
             "(default: PROCEDURE_DATE).",
    )

    # --- history-enrich ---
    h = sub.add_parser(
        "history-enrich",
        help="Add temporal (valid-on-date + crosswalk) columns to an "
             "already-normalized CSV, without re-running normalization.",
    )
    h.add_argument("--input", required=True,
                   help="Input *normalized* CSV (must contain --proc-codes-col "
                        "and --date-col).")
    h.add_argument("--output", required=True, help="Output enriched CSV path.")
    h.add_argument(
        "--kb-csv",
        required=True,
        help="Path to codes_with_ranges.csv (used to seed active-codes set).",
    )
    h.add_argument("--history-changes", required=True,
                   help="Path to code_changes.csv.")
    h.add_argument("--history-deleted", required=True,
                   help="Path to deleted_codes.csv.")
    h.add_argument("--proc-codes-col", default="proc_codes")
    h.add_argument("--date-col", default="PROCEDURE_DATE")

    # --- stats ---
    s = sub.add_parser("stats", help="Compute code-frequency stats from training split.")
    s.add_argument("--train-csv", required=True, help="Normalized training CSV.")
    s.add_argument("--output", required=True, help="Output stats CSV path.")
    s.add_argument("--code-col", default="proc_codes")

    # --- drift ---
    d = sub.add_parser("drift", help="Annotate codes as seen/unseen relative to training.")
    d.add_argument("--split-csv", required=True, help="Normalized split CSV (val/test/drift).")
    d.add_argument("--train-stats", required=True, help="Training code-frequency stats CSV.")
    d.add_argument("--output", required=True, help="Output drift annotation CSV.")
    d.add_argument(
        "--kb-csv",
        default=None,
        help="Path to code knowledge-base CSV (optional).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "normalize":
        kb_codes = _load_kb_codes(Path(args.kb_csv)) if args.kb_csv else None
        token_col = args.token_col if args.token_col else None

        history = None
        if args.history_changes or args.history_deleted:
            if not (args.history_changes and args.history_deleted and args.kb_csv):
                raise SystemExit(
                    "--history-changes, --history-deleted, and --kb-csv must "
                    "all be provided together for temporal enrichment."
                )
            history = _load_code_history(
                changes_csv=Path(args.history_changes),
                deleted_csv=Path(args.history_deleted),
                kb_csv=Path(args.kb_csv),
            )
            stats = history.summary_stats()
            LOGGER.info(
                "Loaded CodeHistory: %d codes, %d events",
                stats["n_all_codes"],
                stats["n_events_total"],
            )

        counters = normalize_split(
            input_csv=Path(args.input),
            output_csv=Path(args.output),
            kb_codes=kb_codes,
            min_tokens=args.min_tokens,
            token_col=token_col,
            note_text_col=args.note_text_col,
            code_col=args.code_col,
            history=history,
            date_col=args.date_col,
        )
        print("Normalization summary:")
        for k, v in counters.items():
            print(f"  {k}: {v:,}")

    elif args.command == "history-enrich":
        history = _load_code_history(
            changes_csv=Path(args.history_changes),
            deleted_csv=Path(args.history_deleted),
            kb_csv=Path(args.kb_csv),
        )
        stats = history.summary_stats()
        LOGGER.info(
            "Loaded CodeHistory: %d codes, %d events",
            stats["n_all_codes"],
            stats["n_events_total"],
        )
        counters = enrich_csv_with_history(
            input_csv=Path(args.input),
            output_csv=Path(args.output),
            history=history,
            proc_codes_col=args.proc_codes_col,
            date_col=args.date_col,
        )
        print("History-enrichment summary:")
        for k, v in counters.items():
            print(f"  {k}: {v:,}")

    elif args.command == "stats":
        compute_code_frequency_stats(
            train_csv=Path(args.train_csv),
            output_csv=Path(args.output),
            code_col=args.code_col,
        )

    elif args.command == "drift":
        kb_codes = _load_kb_codes(Path(args.kb_csv)) if args.kb_csv else None
        annotate_drift(
            split_csv=Path(args.split_csv),
            train_stats_csv=Path(args.train_stats),
            output_csv=Path(args.output),
            kb_codes=kb_codes,
        )


if __name__ == "__main__":
    main()
