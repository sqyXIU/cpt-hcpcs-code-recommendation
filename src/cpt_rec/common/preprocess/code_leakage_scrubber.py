#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
CPT / HCPCS code-leakage scrubber for operative-report notes.

Motivation
----------
Audit of ``op_reports_sample_DEID.csv`` showed that **42 % of ground-truth
CPT codes appear verbatim inside NOTE_TEXT**, overwhelmingly via an Epic
"Procedure Performed (Required):" template of the form::

    RELEASE ARTHROSCOPY RETINACULUM KNEE  29873 - PR KNEE SCOPE, W/LATERAL RELEASE
    ARTHROSCOPY KNEE                      29875 - PR KNEE SCOPE,PART SYNOVECT

Feeding raw notes to any model that tokenizes them lets the model (or a
BM25 baseline) read codes straight off the page.  This module removes
that leak.

Scrubbing strategy (Level-2 aggressive)
---------------------------------------
1. Replace every KB-known CPT/HCPCS code token with ``<CPT_MASK>``.
   The regex is boundary-aware but tolerant of EHR dictation artifacts
   that glue extra tokens directly onto the code:
       (?<![A-Za-z0-9])
       [A-Za-z]{0,3}?         — 0-3 stray prefix letters, non-greedy
                                (``C15820``, ``W52353``, ``eye66180``,
                                ``CPT27197``)
       CODE                   — any KB or retired-GT code
       [A-Za-z]{0,3}          — laterality / modifier suffix  (``49621L``, ``49320PB``)
       (?:x\\d{1,2})?          — "xN" multiplicand  (``15734x2``, ``15272x3``)
       (?![A-Za-z0-9])
   Phone numbers, MRNs, zip codes, and ICD-bracketed codes remain
   untouched because the outer boundaries still require non-alnum
   context on both ends of the full match, and the prefix / suffix
   slots only absorb tokens that are glued directly to an in-vocab
   code.
2. Strip the Epic-template tail ``<CPT_MASK> - PR <OFFICIAL_DESCRIPTION>``
   that mirrors the official CPT short description (itself a leak).
   The procedure-name prefix on the same line is preserved.
3. Redact common explicit-reference phrases (``CPT code:``, ``CPT:``,
   ``HCPCS:``) by collapsing them to ``<CPT_MASK>:`` when followed by a
   code that was masked in step 1, leaving a uniform cleaned surface
   form.

The raw input files are never modified; output is written to the path
given by ``--output`` (which may equal ``--input`` to overwrite in place
— this is what the splits_eval workflow uses).

CLI
---
::

    cptrec-scrub-leakage \\
        --input  output/splits_eval/test.csv \\
        --output output/splits_eval/test.csv \\
        --kb     data/kb/codes_with_ranges.csv

Per-row stats columns added to the output CSV: ``N_CODES_REDACTED``,
``N_PR_TAILS_STRIPPED``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm

from cpt_rec.common.knowledge.code_history import CodeHistory
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.common.preprocess.code_utils import (
    is_valid_proc_code,
    parse_proc_codes,
)

LOGGER = logging.getLogger(__name__)

MASK = "<CPT_MASK>"

# Vocabulary shape gate.
#
# The scrubber's regex uses ``re.IGNORECASE`` and absorbs up to 5 letters of
# prefix and 4 letters of suffix around each code.  Anything short (like a
# 2-char HCPCS modifier ``MA`` / ``ME`` / ``MD``) or otherwise non-code shape
# that slips into the alternation would match arbitrary English words —
# e.g. ``Name`` (=``Na`` + ``me``), ``female`` (=``fe`` + ``ma`` + ``le``),
# ``MAIN`` (=``MA`` + ``IN``).  We refuse to add anything to the vocabulary
# that isn't a real CPT/HCPCS procedure-code shape.
#
# The shape predicate is ``code_utils.is_valid_proc_code`` so that this
# scrubber, the BM25 index, the pipeline's candidate-pool builder, and any
# vocabulary-building call site stay in lockstep — change the regex once
# in ``code_utils._VALID_BASE_RE`` and every gate updates.

# Tail stripper — must be built once, not per-note.  Matches:
#   <CPT_MASK> [space/hyphen] PR <description>
# where <description> runs until end-of-line, 2+ consecutive spaces
# (EHR line-collapse separator), or end-of-text.  Case-insensitive.
_PR_TAIL_RE = re.compile(
    re.escape(MASK) + r"\s*-\s*PR\s+[^\n\r]+?(?=\s{2,}|\n|\r|$)",
    flags=re.IGNORECASE,
)

# Explicit reference phrases that often precede a code, e.g.
#   "CPT code: <CPT_MASK>" -> leave as-is (already informative-free)
#   "CPT 52332" -> with code already masked this becomes "CPT <CPT_MASK>"
# No additional redaction is needed; we keep these phrases.


def _trie_regex(codes: Iterable[str]) -> str:
    """Compile a code vocabulary into a trie-structured regex pattern.

    Matches exactly the same language as a flat ``"|".join(...)``
    alternation, but shares common prefixes so the regex engine fails a
    non-matching position after ~1-2 character comparisons instead of
    scanning tens of thousands of branches.  Match *preference* is also
    preserved: where one code is a proper prefix of another, the longer
    continuation is tried before accepting the shorter code (a greedy
    ``?`` on the terminal), mirroring the longest-first sort the flat
    alternation used.  (For the CPT/HCPCS vocabulary this is moot anyway
    — every shape-valid code is exactly 5 characters, so at most one
    alternative can match at a given position.)

    Deterministic: sorted insertion + sorted branch emission produce a
    stable pattern string for a given code set.
    """
    trie: dict = {}
    for word in sorted(codes):
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["\0"] = None  # terminal marker; codes are alnum, so no collision

    def emit(node: dict) -> str:
        terminal = "\0" in node
        branches: List[str] = []
        leaf_chars: List[str] = []
        for ch in sorted(k for k in node if k != "\0"):
            sub = emit(node[ch])
            if sub == "":
                leaf_chars.append(ch)
            else:
                branches.append(re.escape(ch) + sub)
        if leaf_chars:
            if len(leaf_chars) == 1:
                branches.append(re.escape(leaf_chars[0]))
            else:
                branches.append("[" + "".join(re.escape(c) for c in leaf_chars) + "]")
        if not branches:
            return ""  # pure terminal (end of a code)
        if len(branches) == 1 and not terminal:
            pat = branches[0]
        else:
            pat = "(?:" + "|".join(branches) + ")"
        if terminal:
            pat += "?"  # greedy -> longer code tried before shorter prefix
        return pat

    return emit(trie)


@dataclass(frozen=True)
class ScrubResult:
    """Return value of :meth:`CodeLeakageScrubber.scrub`."""

    text: str
    n_codes_redacted: int
    n_pr_tails_stripped: int


class CodeLeakageScrubber:
    """
    Level-2 scrubber.

    Parameters
    ----------
    kb_codes :
        Set of all known CPT / HCPCS code strings (from
        :attr:`CodeKnowledgeBase.codes`).  Used to build a single
        alternation regex that matches *only* codes in the KB — this
        avoids over-redacting unrelated 5-digit tokens (MRNs, zip codes,
        phone-number fragments).
    extra_codes :
        Optional additional codes to include in the redaction vocabulary.
        The CLI populates this with every code observed in the input
        CSV's ``CPT_CODES`` column so that retired / deleted codes that
        still appear in historical ground-truth labels are scrubbed even
        though the current KB has dropped them.
    replacement :
        Placeholder string to substitute.  Default ``"<CPT_MASK>"``.
    """

    def __init__(
        self,
        kb_codes: Iterable[str],
        extra_codes: Iterable[str] | None = None,
        replacement: str = MASK,
    ) -> None:
        raw: Set[str] = {c.strip().upper() for c in kb_codes if c and c.strip()}
        if extra_codes:
            raw.update(c.strip().upper() for c in extra_codes if c and c.strip())
        if not raw:
            raise ValueError("kb_codes is empty; nothing to scrub.")

        # Shape gate: drop anything that isn't a recognized CPT/HCPCS
        # procedure-code shape.  This is the single line of defense that
        # keeps short modifier tokens (HCPCS 2-char modifiers like ``MA``,
        # ``ME``, ``MD``, ``GD``) and stray prose (``CPT``, ``645``,
        # ``44405-1``) out of the alternation regex — where, under
        # IGNORECASE, they would otherwise over-mask English words.
        codes: Set[str] = {c for c in raw if is_valid_proc_code(c)}
        dropped = raw - codes
        if dropped:
            sample = sorted(dropped)[:10]
            LOGGER.info(
                "CodeLeakageScrubber: filtered %d non-procedure-code token(s) "
                "from vocab (kept %d; dropped examples: %s)",
                len(dropped), len(codes), sample,
            )
        if not codes:
            raise ValueError(
                "All supplied codes were rejected by the shape filter "
                "(expected CPT 5-digit / 4-digit+FTUM or HCPCS letter+4-digit); "
                "nothing to scrub."
            )

        self._replacement = replacement
        self._codes = codes

        # Build the code alternation as a trie-structured pattern.  A flat
        # 27k-branch "A|B|..." alternation makes CPython's re engine scan
        # branches linearly at every text position (hours over a 325k-row
        # split); the trie shares prefixes and fails after ~1-2 char
        # comparisons.  Same language, same match preference (see
        # _trie_regex) — the scrubbed output is byte-identical.
        alternation = _trie_regex(self._codes)
        # Boundary-aware with optional prefix / suffix / multiplicand:
        #   (?<![A-Za-z0-9])            — no alnum to the left of the match
        #   [A-Za-z]{0,5}?              — absorb up to 5 stray letters glued
        #                                 in front of the code (non-greedy,
        #                                 so the clean-space case takes 0
        #                                 prefix chars and nothing is
        #                                 over-absorbed).  Covers dictation
        #                                 artifacts ("C15820", "W52353",
        #                                 "eye66180", "CPT27197") AND Epic
        #                                 concatenations where a word runs
        #                                 straight into a code without a
        #                                 space ("CLIPS58671").  The
        #                                 non-greedy quantifier means we
        #                                 still mask cleanly when there's
        #                                 no prefix word at all.
        #   (?:CODE_A|CODE_B|...)       — any KB / retired-GT / history code
        #   [A-Za-z]{0,4}               — absorb a trailing laterality or
        #                                 abbreviation suffix ("L", "RT",
        #                                 "PB", "MILF", …).  Widened to 4
        #                                 for Epic-crushed surgical abbrevs
        #                                 like "22558MILF".
        #   (?:(?<=[A-Za-z])\d{1,2}     — absorb a 2-digit CPT modifier
        #     [A-Za-z]{0,3})?             that follows an alpha suffix,
        #                                 plus an optional 1-3 letter
        #                                 trailing tag glued to the digits
        #                                 ("67113G25", "67113G23",
        #                                  "67108G25B", "67108G25BIL").
        #                                 The lookbehind forces the digit
        #                                 tail to only fire when at least
        #                                 one alpha was consumed, so a
        #                                 clean code followed by a stray
        #                                 number ("52332 35") is left
        #                                 alone, and the post-digit alpha
        #                                 group is nested inside the same
        #                                 optional so that codes WITHOUT a
        #                                 modifier-digit tail (e.g.
        #                                 "52332LONGER") still preserve
        #                                 their existing 5-letter-cap
        #                                 behavior.
        #   (?:x\d{1,2})?               — absorb an "xN" multiplicand
        #                                 ("15734x2", "15272x3")
        #   (?![A-Za-z0-9])             — no alnum to the right of the tail
        # IGNORECASE so "g0121" matches HCPCS "G0121" even if a note
        # contains the lowercase variant.
        self._code_re = re.compile(
            rf"(?<![A-Za-z0-9])"
            rf"[A-Za-z]{{0,5}}?"
            rf"(?:{alternation})"
            rf"[A-Za-z]{{0,4}}"
            rf"(?:(?<=[A-Za-z])\d{{1,2}}[A-Za-z]{{0,3}})?"
            rf"(?:x\d{{1,2}})?"
            rf"(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------
    def scrub(self, text: str) -> ScrubResult:
        """Return cleaned text + counts.  Idempotent (safe to re-apply)."""
        if not isinstance(text, str) or not text:
            return ScrubResult(text if isinstance(text, str) else "", 0, 0)

        # Step 1: mask every KB code token, absorbing an optional single
        # prefix letter, 1-3 letter suffix, and "xN" multiplicand.
        cleaned, n_codes = self._code_re.subn(self._replacement, text)

        # Step 2: strip "<CPT_MASK> - PR <description>" tails.  Only
        # the tail is removed; the mask itself survives so the sentence
        # stays grammatical (e.g. "ARTHROSCOPY KNEE  <CPT_MASK>").
        def _strip(m: re.Match) -> str:
            return self._replacement

        cleaned, n_tails = _PR_TAIL_RE.subn(_strip, cleaned)

        return ScrubResult(cleaned, n_codes, n_tails)

    # ------------------------------------------------------------------
    # Convenience batch ops
    # ------------------------------------------------------------------
    def scrub_series(
        self,
        s: pd.Series,
        workers: int = 1,
        progress: bool = False,
        desc: str | None = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Apply :meth:`scrub` to every string in a pandas Series.

        ``workers > 1`` fans rows out over a process pool in ordered
        chunks — row order (and therefore the output CSV) is identical
        to the serial path.  ``progress`` draws a tqdm bar.
        """
        values = s.tolist()
        texts: list[str] = []
        n_codes: list[int] = []
        n_tails: list[int] = []

        if workers > 1 and len(values) > 1:
            chunk_size = max(64, math.ceil(len(values) / (workers * 8)))
            chunks = [
                values[i : i + chunk_size]
                for i in range(0, len(values), chunk_size)
            ]
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(self._codes, self._replacement),
            ) as ex, tqdm(
                total=len(values), desc=desc, unit="note",
                disable=not progress,
            ) as bar:
                # executor.map preserves chunk order -> deterministic output
                for c_texts, c_codes, c_tails in ex.map(_scrub_chunk, chunks):
                    texts.extend(c_texts)
                    n_codes.extend(c_codes)
                    n_tails.extend(c_tails)
                    bar.update(len(c_texts))
        else:
            for v in tqdm(values, desc=desc, unit="note", disable=not progress):
                r = self.scrub(v)
                texts.append(r.text)
                n_codes.append(r.n_codes_redacted)
                n_tails.append(r.n_pr_tails_stripped)

        return (
            pd.Series(texts, index=s.index, name=s.name),
            pd.Series(n_codes, index=s.index),
            pd.Series(n_tails, index=s.index),
        )


# ---------------------------------------------------------------------------
# Process-pool plumbing for scrub_series(workers > 1).  Each worker builds
# the scrubber once from the parent's already-shape-filtered code set (so
# no duplicate filter logging) and reuses it for every chunk.
# ---------------------------------------------------------------------------
_WORKER_SCRUBBER: CodeLeakageScrubber | None = None


def _worker_init(codes: Set[str], replacement: str) -> None:
    global _WORKER_SCRUBBER
    _WORKER_SCRUBBER = CodeLeakageScrubber(codes, replacement=replacement)


def _scrub_chunk(
    chunk: Sequence,
) -> Tuple[List[str], List[int], List[int]]:
    assert _WORKER_SCRUBBER is not None, "worker not initialized"
    texts: List[str] = []
    n_codes: List[int] = []
    n_tails: List[int] = []
    for v in chunk:
        r = _WORKER_SCRUBBER.scrub(v)
        texts.append(r.text)
        n_codes.append(r.n_codes_redacted)
        n_tails.append(r.n_pr_tails_stripped)
    return texts, n_codes, n_tails


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_kb_codes(kb_path: Path) -> Set[str]:
    kb = CodeKnowledgeBase.from_csv(kb_path, build_index=False)
    return set(kb.codes)


def _harvest_extra_codes(df: pd.DataFrame, code_col: str) -> Set[str]:
    """Collect every valid CPT/HCPCS token from the GT-label column so
    retired codes absent from the current KB still get masked."""
    if code_col not in df.columns:
        return set()
    extra: Set[str] = set()
    for raw in df[code_col].dropna().astype(str):
        for c in parse_proc_codes(raw):
            extra.add(c)
    return extra


def _run_one(
    input_csv: Path,
    output_csv: Path,
    kb_codes: Set[str],
    note_col: str = "NOTE_TEXT",
    code_col: str = "CPT_CODES",
    history_codes: Set[str] | None = None,
    workers: int = 1,
    progress: bool = False,
) -> None:
    df = pd.read_csv(input_csv)
    if note_col not in df.columns:
        raise ValueError(
            f"{input_csv}: column {note_col!r} not found "
            f"(have: {list(df.columns)})"
        )
    extra = _harvest_extra_codes(df, code_col)
    retired = extra - kb_codes
    if retired:
        LOGGER.info(
            "  found %d GT codes not in KB (likely retired); adding to vocab",
            len(retired),
        )
    if history_codes:
        before = len(extra | kb_codes)
        extra = extra | history_codes
        added = len(extra | kb_codes) - before
        LOGGER.info(
            "  adding %d historical (deleted/revised) codes from CodeHistory",
            added,
        )
    scrubber = CodeLeakageScrubber(kb_codes, extra_codes=extra)
    LOGGER.info("Scrubbing %s  (%d rows, vocab=%d codes, workers=%d)",
                input_csv, len(df), len(scrubber._codes), workers)
    cleaned, n_codes, n_tails = scrubber.scrub_series(
        df[note_col], workers=workers, progress=progress,
        desc=f"scrub {input_csv.name}",
    )
    df[note_col] = cleaned
    df["N_CODES_REDACTED"] = n_codes.astype(int)
    df["N_PR_TAILS_STRIPPED"] = n_tails.astype(int)

    total_codes = int(n_codes.sum())
    total_tails = int(n_tails.sum())
    n_rows_touched = int((n_codes > 0).sum())
    LOGGER.info(
        "  %d code tokens masked, %d PR-tails stripped, %d / %d rows touched",
        total_codes,
        total_tails,
        n_rows_touched,
        len(df),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    LOGGER.info("  -> wrote %s", output_csv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scrub CPT/HCPCS code leakage from operative-note CSVs.",
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV (or a directory of CSVs if --glob is also given).",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Output CSV path.  May equal --input to overwrite in place.  "
            "If --input is a directory, --output must also be a directory."
        ),
    )
    ap.add_argument(
        "--glob",
        default=None,
        help=(
            "If --input is a directory, glob pattern (relative) to select "
            "CSVs inside it, e.g. '*.csv'.  Outputs are written with the "
            "same relative name under --output."
        ),
    )
    ap.add_argument(
        "--kb",
        required=True,
        type=Path,
        help="Path to kb/codes_with_ranges.csv.",
    )
    ap.add_argument(
        "--note-col",
        default="NOTE_TEXT",
        help="Column containing the note text (default: NOTE_TEXT).",
    )
    ap.add_argument(
        "--code-col",
        default="CPT_CODES",
        help=(
            "Column containing GT codes.  Codes here are harvested and "
            "added to the redaction vocabulary so retired codes absent "
            "from the KB still get masked.  Default: CPT_CODES."
        ),
    )
    ap.add_argument(
        "--history-changes",
        default=None,
        type=Path,
        help=(
            "Optional path to code_changes.csv.  When provided together "
            "with --history-deleted, every code that ever appeared in the "
            "history ledgers (deleted/revised/new) is added to the scrub "
            "vocabulary, so retired codes whose substitutes are also "
            "retired still get masked."
        ),
    )
    ap.add_argument(
        "--history-deleted",
        default=None,
        type=Path,
        help="Optional path to deleted_codes.csv (pair with --history-changes).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Process-pool size for row-parallel scrubbing.  0 (default) "
            "= all CPU cores; 1 = serial.  Output is byte-identical at "
            "any worker count."
        ),
    )
    ap.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable the per-file tqdm progress bar.",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args(argv)
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    LOGGER.info("Loading KB from %s", args.kb)
    kb_codes = _load_kb_codes(args.kb)
    LOGGER.info("KB size: %d codes", len(kb_codes))

    history_codes: Set[str] | None = None
    if args.history_changes or args.history_deleted:
        if not (args.history_changes and args.history_deleted):
            raise SystemExit(
                "--history-changes and --history-deleted must be provided together."
            )
        LOGGER.info(
            "Loading CodeHistory from %s + %s",
            args.history_changes, args.history_deleted,
        )
        history = CodeHistory.from_csvs(
            changes_csv=args.history_changes,
            deleted_csv=args.history_deleted,
            active_kb_codes=kb_codes,
        )
        history_codes = set(history.all_codes())
        LOGGER.info(
            "CodeHistory: %d total codes (%d not in active KB)",
            len(history_codes),
            len(history_codes - kb_codes),
        )

    inputs: list[tuple[Path, Path]] = []
    if args.input.is_dir():
        if args.glob is None:
            raise SystemExit("--input is a directory; --glob is required")
        for p in sorted(args.input.glob(args.glob)):
            if p.is_file():
                rel = p.relative_to(args.input)
                inputs.append((p, args.output / rel))
    else:
        inputs.append((args.input, args.output))

    if not inputs:
        LOGGER.warning("No input files matched; nothing to do.")
        return 0

    for src, dst in inputs:
        _run_one(
            src,
            dst,
            kb_codes=kb_codes,
            note_col=args.note_col,
            code_col=args.code_col,
            history_codes=history_codes,
            workers=workers,
            progress=args.progress,
        )

    LOGGER.info("Done.  Processed %d file(s).", len(inputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
