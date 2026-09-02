# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Shared CPT/HCPCS code parsing and normalization utilities.

This module is the single source of truth for code parsing logic. All other
modules (label normalizer, snippet constructor, knowledge base, baselines,
etc.) should import from here rather than maintaining their own regex
patterns.

Recognized code formats
-----------------------
- CPT Category I:    5 digits              (e.g. ``43235``)
- CPT Category II:   4 digits + ``F``      (e.g. ``2029F``)
- CPT Category III:  4 digits + ``T``      (e.g. ``0479T``)
- CPT PLA:           4 digits + ``U``      (e.g. ``0001U``)
- CPT MAAA:          4 digits + ``M``      (e.g. ``0002M``)
- CPT vaccine admin: 4 digits + ``A``      (e.g. ``0001A``, ``0141A`` — COVID-19
                                            vaccine administration codes added
                                            in 2020)
- HCPCS Level II:    1 letter + 4 digits   (e.g. ``A0428``, ``J1885``, ``S2068``)

The suffix-letter class ``[AFMTU]`` was verified against the 2026 KB +
history snapshot: those are the only 4-digit+letter suffixes actually in
use (47× ``A``, 607× ``F``, 20× ``M``, 1,004× ``T``, 630× ``U``).

Modifier handling
-----------------
Raw billing strings may contain trailing modifiers appended directly to the
base code (e.g. ``52235BL``, ``27130A``, ``67036G25``, ``33405TR``,
``45395R``, ``66999.11``).  The :func:`strip_modifier` function removes these
suffixes to recover the 5-character base code.

Non-code tokens like ``PBHANDEXT``, ``ROBOT39`` are rejected by
:func:`is_valid_code_format`.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# Core regex — matches any standalone CPT/HCPCS code
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"\d{5}"            # CPT Cat I
    r"|\d{4}[AFMTU]"    # CPT vaccine admin (A), Cat II (F), MAAA (M),
                        # Cat III (T), PLA (U).
    r"|[A-Z]\d{4}"      # HCPCS Level II
    r")(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)

# Valid base-code pattern (after normalization to uppercase).
# Single source of truth for the "is this a procedure-code shape?" question;
# imported by the leakage scrubber's vocab gate, the BM25 index builder, the
# pipeline's candidate-pool filter, and other call sites that need the answer.
_VALID_BASE_RE = re.compile(
    r"^(?:\d{5}|\d{4}[AFMTU]|[A-Z]\d{4})$"
)


def pipe_split_codes(value) -> List[str]:
    """
    Split a pipe-separated procedure-code field into an uppercase list.

    Used everywhere a CSV row carries a ``"43235|45378|..."``-style column
    (gold labels, predictions, pseudo-evidence audits, label matrices).
    Tolerates ``None``, ``NaN`` (float), and empty / whitespace-only strings
    by returning ``[]``.  Does NOT strip modifiers — pipe-separated fields
    are by convention already in their normalized base-code form.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [c.strip().upper() for c in s.split("|") if c.strip()]


def parse_proc_codes(code_str: str) -> List[str]:
    """
    Extract CPT + HCPCS codes from a raw string.

    Returns codes in order of appearance, uppercased, de-duplicated.
    This extracts *only* tokens that are already in valid code format;
    it does NOT strip modifiers. For modifier stripping, pipe each raw
    token through :func:`strip_modifier` first.
    """
    if code_str is None or (isinstance(code_str, float) and math.isnan(code_str)):
        return []

    s = str(code_str)
    found = [m.group(0).upper() for m in _CODE_RE.finditer(s)]
    seen: Set[str] = set()
    out: List[str] = []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def is_valid_code_format(code: str) -> bool:
    """
    Return True if *code* matches a recognized CPT/HCPCS format.

    This is a syntactic check only — it does not verify against a knowledge base.
    """
    return bool(_VALID_BASE_RE.match(code.strip().upper()))


def strip_modifier(raw_token: str) -> Optional[str]:
    """
    Attempt to recover a valid base code from a raw billing token that may
    contain appended modifiers.

    Returns the uppercase base code if one can be recovered, or ``None`` if
    the token is not recognizable.

    Examples::

        strip_modifier("52235BL")    -> "52235"
        strip_modifier("27130A")     -> "27130"
        strip_modifier("67036G25")   -> "67036"
        strip_modifier("33405TR")    -> "33405"
        strip_modifier("66999.11")   -> "66999"
        strip_modifier("45395R")     -> "45395"
        strip_modifier("61154T")     -> "61154"  *only* if 61154 is valid Cat-I
                                        "61154" is 5 digits -> valid
                                        but if "0479T" -> "0479T" (valid Cat III)
        strip_modifier("PBHANDEXT")  -> None
        strip_modifier("ROBOT39")    -> None
        strip_modifier("A44320")     -> "A4432" (HCPCS + trailing 0)
        strip_modifier("J0585")      -> "J0585"  (already valid)
    """
    raw = raw_token.strip().upper()
    if not raw:
        return None

    # Already a valid code? Return as-is.
    if _VALID_BASE_RE.match(raw):
        return raw

    # Strategy 1: If it starts with a letter + 4 digits (HCPCS) and has extra
    # trailing chars, try the first 5 characters.
    if len(raw) >= 5 and raw[0].isalpha() and raw[1:5].isdigit():
        candidate = raw[:5]
        if _VALID_BASE_RE.match(candidate):
            return candidate

    # Strategy 2: Extract leading 5-digit CPT stem from tokens like
    # "52235BL", "27130A", "67036G25", "66999.11"
    m = re.match(r"^(\d{5})", raw)
    if m:
        return m.group(1)

    # Strategy 3: Try 4-digit + suffix patterns (vaccine-admin "A", Cat II
    # "F", MAAA "M", Cat III "T", PLA "U").  Only if the FULL token matches;
    # "0479T" is already caught above; "35820T" has 5-digit stem "35820".
    m = re.match(r"^(\d{4}[AFMTU])", raw)
    if m:
        return m.group(1)

    return None


def is_valid_proc_code(code: str) -> bool:
    """
    Return True iff *code* (after upper-casing + strip) matches a recognized
    CPT/HCPCS procedure-code shape.

    This is the canonical gate for any vocabulary-building step that wants to
    keep modifier tokens (``MA``/``ME``/``GD``), versioned annotations
    (``44405-1``), code-system tags (``CPT``), and other non-code strings out
    of its alternation regex.  The ``CodeLeakageScrubber.__init__`` shape
    filter, ``TrainNoteBM25Index.from_csv``, and the pipeline's candidate-pool
    builder all flow through this function, so a single regex change here
    propagates everywhere.

    Alias of :func:`is_valid_code_format`; named separately so callers that
    are filtering *vocabularies* (rather than checking *individual tokens*)
    can use a verb-noun name that reads naturally as a predicate.
    """
    return is_valid_code_format(code)


def normalize_code_list(
    raw_code_string: str,
    *,
    valid_codes: Optional[Set[str]] = None,
) -> tuple[List[str], List[str], List[str]]:
    """
    Parse, strip modifiers, deduplicate, and optionally validate a raw
    comma-separated code string from the ``CPT_CODES`` column.

    Parameters
    ----------
    raw_code_string :
        The raw string, e.g. ``"52332, 52353, 74420"``.
    valid_codes :
        If provided, a set of known-valid code strings (from the knowledge
        base).  Codes not in this set after modifier stripping are placed in
        *unresolved*.

    Returns
    -------
    (accepted, unresolved, dropped) :
        - *accepted*: valid base codes, de-duplicated, in original order.
        - *unresolved*: tokens that produced a syntactically valid code but
          are not in *valid_codes*.
        - *dropped*: tokens that could not be parsed at all.
    """
    if raw_code_string is None or (
        isinstance(raw_code_string, float) and math.isnan(raw_code_string)
    ):
        return [], [], []

    raw = str(raw_code_string)
    # Split on comma, semicolon, pipe, or whitespace
    tokens = re.split(r"[,;|\s]+", raw)

    accepted: List[str] = []
    unresolved: List[str] = []
    dropped: List[str] = []
    seen: Set[str] = set()

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        base = strip_modifier(tok)
        if base is None:
            dropped.append(tok)
            continue

        if base in seen:
            continue
        seen.add(base)

        if valid_codes is not None and base not in valid_codes:
            unresolved.append(base)
        else:
            accepted.append(base)

    return accepted, unresolved, dropped
