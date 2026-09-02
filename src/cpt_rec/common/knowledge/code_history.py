# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
CPT / HCPCS code-history layer.

Lifts the static 2026 :class:`CodeKnowledgeBase` snapshot into a *versioned*
resource that can answer, for any ``(code, as_of)`` pair:

* **Is this code valid on this date?** (:meth:`CodeHistory.is_active`)
* **If not, what code replaces it?** (:meth:`CodeHistory.crosswalk`)
* **What AMA / CMS guidance applies around this date?**
  (:meth:`CodeHistory.advice_for`)
* **What was the full set of valid codes on this date?**
  (:meth:`CodeHistory.active_codes`)

Inputs
------
``data/kb/code_changes.csv``
    New / Revised / Deleted events 2018 → 2026 with AMA / CMS
    ``advice`` text and ``crosswalk_code`` substitutions.  Effective
    dates are annualized to ``date(year, 1, 1)`` because the file is
    only year-resolved.

``data/kb/deleted_codes.csv``
    Full retirement ledger 1990 → 2026 with precise ``deleted_date``
    (format ``D-MMM-YY``) and ``substitute_code``.  Two-digit year
    disambiguated via a fixed cutoff (YY ∈ [00..40] → 20YY, else
    19YY); see :meth:`CodeHistory.parse_deleted_date`.

``codes_with_ranges.csv`` (via ``CodeKnowledgeBase.codes``)
    The active 2026 set.  Needed to disambiguate "has no events, never
    heard of it" from "has no events, always active".

Reconciliation
--------------
For codes that appear as ``Deleted`` in both files (~1.2k overlap):

* effective date comes from ``deleted_codes`` (precise day) when
  present, otherwise from ``code_changes`` (Jan 1 of year);
* crosswalk comes from ``code_changes.crosswalk_code`` first, then
  falls back to ``deleted_codes.substitute_code``;
* ``advice`` comes from ``code_changes`` (``deleted_codes`` has no
  advice column).

This module is pure pandas + stdlib — no LLM, no network, no GPU —
so it can run on a laptop and in unit tests.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Two-digit-year cutoff for deleted_codes.csv's "D-MMM-YY" format.
# YY in [00..CUTOFF] => 20YY, else 19YY.  AMA started CPT in 1966 and
# we have no records before 1990, so 40 is safely forward-looking.
YY_CUTOFF = 40

_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Regex for extracting CPT or HCPCS-Level-II codes from free-form text
# in ``crosswalk_code`` / ``substitute_code`` (some rows contain prose
# like "To report, use 97127" instead of a bare code).
# CPT: 5 digits (e.g. 97127, 00100) or 4 digits + trailing letter
# (e.g. 0479T, 2029F, 0001U, 0002M).
# HCPCS Level II: one uppercase letter + 4 digits (e.g. J1885, G0008).
_CODE_TOKEN_RX = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{4}[A-Z]|\d{5}|[A-Z]\d{4})(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class CodeEvent:
    """A single dated change to a code's status or descriptor."""

    code: str
    code_system: str              # 'CPT' | 'HCPCS'
    change_type: str              # 'new' | 'revised' | 'deleted'
    effective_date: date
    year: int                     # year of the effective_date (redundant, convenient)
    description: Optional[str] = None
    advice: Optional[str] = None
    crosswalk_code: Optional[str] = None
    new_descriptor: Optional[str] = None
    source: str = "code_changes"  # 'code_changes' | 'deleted_codes'


class CodeHistory:
    """Versioned CPT / HCPCS history.  See module docstring."""

    # Expected columns
    _CHANGES_COLS = {
        "code_system", "year", "category", "change_type", "code",
        "description", "advice", "crosswalk_code", "new_descriptor",
    }
    _DELETED_COLS = {
        "code", "description", "deleted_date", "substitute_code",
        "archived_info", "code_system",
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        events: Dict[str, List[CodeEvent]],
        active_kb_codes: Iterable[str],
    ) -> None:
        # Defensive copy + per-code sort by (effective_date, change_type)
        # so the state machine walks events in a stable order.
        self._events: Dict[str, List[CodeEvent]] = {
            c: sorted(evs, key=lambda e: (e.effective_date, e.change_type))
            for c, evs in events.items()
        }
        self._active_kb_codes: FrozenSet[str] = frozenset(
            c.strip().upper() for c in active_kb_codes if isinstance(c, str)
        )

    @classmethod
    def from_csvs(
        cls,
        changes_csv: Path | str,
        deleted_csv: Path | str,
        active_kb_codes: Iterable[str],
    ) -> "CodeHistory":
        """Load both CSVs, reconcile overlaps, return a :class:`CodeHistory`."""
        changes_csv = Path(changes_csv)
        deleted_csv = Path(deleted_csv)

        changes_df = pd.read_csv(changes_csv, dtype=str)
        deleted_df = pd.read_csv(deleted_csv, dtype=str)
        changes_df.columns = [c.strip() for c in changes_df.columns]
        deleted_df.columns = [c.strip() for c in deleted_df.columns]

        missing_c = cls._CHANGES_COLS - set(changes_df.columns)
        missing_d = cls._DELETED_COLS - set(deleted_df.columns)
        if missing_c:
            raise ValueError(f"code_changes.csv missing columns: {missing_c}")
        if missing_d:
            raise ValueError(f"deleted_codes.csv missing columns: {missing_d}")

        events: Dict[str, List[CodeEvent]] = {}

        # ---- pass 1: code_changes.csv (annual resolution) -----------
        for row in changes_df.itertuples(index=False):
            code = _norm_code(row.code)
            if not code:
                continue
            try:
                year = int(row.year)
            except (TypeError, ValueError):
                continue
            ct = _norm_change_type(row.change_type)
            if ct is None:
                continue
            ev = CodeEvent(
                code=code,
                code_system=_norm_system(row.code_system),
                change_type=ct,
                effective_date=date(year, 1, 1),
                year=year,
                description=_s(row.description),
                advice=_s(row.advice),
                crosswalk_code=_s(row.crosswalk_code),
                new_descriptor=_s(row.new_descriptor),
                source="code_changes",
            )
            events.setdefault(code, []).append(ev)

        # ---- pass 2: deleted_codes.csv (precise dates) --------------
        # For codes with a same-year 'deleted' event from code_changes,
        # we *upgrade* that event's effective_date to the precise date
        # and merge in substitute_code when crosswalk_code is missing.
        for row in deleted_df.itertuples(index=False):
            code = _norm_code(row.code)
            if not code:
                continue
            parsed = cls.parse_deleted_date(row.deleted_date)
            if parsed is None:
                LOGGER.debug("Unparseable deleted_date %r for %s", row.deleted_date, code)
                continue
            substitute = _s(row.substitute_code)
            desc = _s(row.description)
            system = _norm_system(row.code_system)

            # See if we already have a Deleted event in the same year
            # (then this is the precise-date upgrade).
            existing = events.get(code, [])
            upgraded = False
            for i, ev in enumerate(existing):
                if ev.change_type == "deleted" and ev.year == parsed.year:
                    merged_xwalk = ev.crosswalk_code or substitute
                    existing[i] = CodeEvent(
                        code=ev.code,
                        code_system=ev.code_system or system,
                        change_type="deleted",
                        effective_date=parsed,
                        year=parsed.year,
                        description=ev.description or desc,
                        advice=ev.advice,
                        crosswalk_code=merged_xwalk,
                        new_descriptor=ev.new_descriptor,
                        source=f"{ev.source}+deleted_codes",
                    )
                    upgraded = True
                    break

            if not upgraded:
                events.setdefault(code, []).append(
                    CodeEvent(
                        code=code,
                        code_system=system,
                        change_type="deleted",
                        effective_date=parsed,
                        year=parsed.year,
                        description=desc,
                        advice=None,
                        crosswalk_code=substitute,
                        new_descriptor=None,
                        source="deleted_codes",
                    )
                )

        # Dedupe exact duplicates (same code/date/type/source)
        for code, evs in events.items():
            seen = set()
            unique = []
            for e in evs:
                key = (e.change_type, e.effective_date, e.source)
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            events[code] = unique

        n_events = sum(len(v) for v in events.values())
        LOGGER.info(
            "CodeHistory: loaded %d events across %d codes (changes=%d, deleted=%d)",
            n_events, len(events), len(changes_df), len(deleted_df),
        )

        return cls(events=events, active_kb_codes=active_kb_codes)

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------
    def all_codes(self) -> Set[str]:
        """Every code we've ever heard of: active KB ∪ historical events."""
        return set(self._events.keys()) | set(self._active_kb_codes)

    def events_for(self, code: str) -> List[CodeEvent]:
        """Return a copy of the event list for *code* (empty if unknown)."""
        c = _norm_code(code)
        return list(self._events.get(c, []))

    def is_active(self, code: str, as_of: date) -> bool:
        """Is *code* a valid billing code on *as_of*?

        State-machine rules:
          * ``new`` → active; ``deleted`` → inactive; ``revised`` → no change.
          * Pre-first-event state:
              - If first state event is ``new``, code not born yet → False.
              - If first state event is ``deleted`` (so the code existed
                before records began), assume active → True.
              - If only ``revised`` events exist, defer to the 2026 KB.
          * No events at all → defer to the 2026 KB.
        """
        c = _norm_code(code)
        ev = self._events.get(c)
        if not ev:
            return c in self._active_kb_codes

        state_events = [e for e in ev if e.change_type in ("new", "deleted")]
        if not state_events:
            return c in self._active_kb_codes

        applicable = [e for e in state_events if e.effective_date <= as_of]
        if not applicable:
            # as_of predates any state change
            first = state_events[0]
            return first.change_type != "new"  # True iff earliest record is a deletion

        # Latest applicable state event wins
        return applicable[-1].change_type == "new"

    def crosswalk(
        self,
        code: str,
        as_of: date,
        max_depth: int = 5,
    ) -> Optional[str]:
        """Return the code that replaces *code* on *as_of*, or None.

        * If *code* is active on *as_of*, returns None (no substitution).
        * Otherwise locates the deletion event ≤ as_of, extracts substitute
          codes from ``crosswalk_code`` (prose-safe), and chases the chain
          until it finds an active substitute.
        * Bounded by *max_depth* to prevent infinite loops.
        """
        c = _norm_code(code)
        if self.is_active(c, as_of):
            return None
        return self._crosswalk(c, as_of, visited=set(), depth=max_depth)

    def _crosswalk(
        self, code: str, as_of: date, visited: Set[str], depth: int
    ) -> Optional[str]:
        if depth <= 0 or code in visited:
            return None
        visited.add(code)

        ev = self._events.get(code, [])
        deletions = [
            e for e in ev
            if e.change_type == "deleted" and e.effective_date <= as_of
        ]
        if not deletions:
            return None
        latest = deletions[-1]  # events are sorted ascending

        candidates = _extract_codes(latest.crosswalk_code)
        # Fall back to new_descriptor only if it looks like a code; for
        # deletions it's usually empty, but some code_changes rows use it.
        if not candidates and latest.new_descriptor:
            candidates = _extract_codes(latest.new_descriptor)

        for sub in candidates:
            if self.is_active(sub, as_of):
                return sub
            chained = self._crosswalk(sub, as_of, visited, depth - 1)
            if chained is not None:
                return chained
        return None

    def advice_for(
        self,
        code: str,
        as_of: date,
        year_window: Tuple[int, int] = (-1, 1),
    ) -> List[str]:
        """Return ``advice`` strings for *code* whose event-year is within
        ``[as_of.year + year_window[0], as_of.year + year_window[1]]``.

        Empty list if the code has no advice in that window.
        """
        c = _norm_code(code)
        lo = as_of.year + year_window[0]
        hi = as_of.year + year_window[1]
        out: List[str] = []
        for e in self._events.get(c, []):
            if e.advice and lo <= e.year <= hi:
                out.append(e.advice)
        return out

    def active_codes(self, as_of: date) -> FrozenSet[str]:
        """The set of codes valid on *as_of*.

        Conservative: iterates over ``all_codes()`` and filters by
        :meth:`is_active`.  O(|all_codes| × k) where k ≤ ~3.  Cache the
        result at the caller if you need repeated queries on the same
        date.
        """
        codes = self.all_codes()
        return frozenset(c for c in codes if self.is_active(c, as_of))

    def crosswalk_map(self, as_of: date) -> Dict[str, str]:
        """Return ``{deleted_code -> substitute}`` for every code deleted
        on or before *as_of* that has a resolvable active substitute.
        """
        out: Dict[str, str] = {}
        for code in self._events:
            if self.is_active(code, as_of):
                continue
            sub = self.crosswalk(code, as_of)
            if sub:
                out[code] = sub
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_events_df(self) -> pd.DataFrame:
        """Flatten all events into a tidy DataFrame for export / auditing."""
        rows = []
        for code, evs in self._events.items():
            for e in evs:
                rows.append({
                    "code": e.code,
                    "code_system": e.code_system,
                    "change_type": e.change_type,
                    "effective_date": e.effective_date.isoformat(),
                    "year": e.year,
                    "description": e.description,
                    "advice": e.advice,
                    "crosswalk_code": e.crosswalk_code,
                    "new_descriptor": e.new_descriptor,
                    "source": e.source,
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["code", "effective_date"]).reset_index(drop=True)
        return df

    def summary_stats(self) -> Dict[str, object]:
        """Small dict of counts, useful for CLI output."""
        by_ct: Dict[str, int] = {}
        for evs in self._events.values():
            for e in evs:
                by_ct[e.change_type] = by_ct.get(e.change_type, 0) + 1
        n_events = sum(len(v) for v in self._events.values())
        return {
            "n_codes_with_events": len(self._events),
            "n_events_total": n_events,
            "n_events_by_type": by_ct,
            "n_active_kb_codes": len(self._active_kb_codes),
            "n_all_codes": len(self.all_codes()),
        }

    # ------------------------------------------------------------------
    # Date parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_deleted_date(
        value: object,
        yy_cutoff: int = YY_CUTOFF,
    ) -> Optional[date]:
        """Parse ``D-MMM-YY`` (e.g. ``"1-Apr-01"``) with a fixed 2-digit-year
        cutoff.

        Returns ``None`` if the value is NaN, empty, or malformed.
        """
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None
        m = re.match(r"^(\d{1,2})-([A-Za-z]{3})-(\d{2})$", s)
        if not m:
            return None
        day_s, mon_s, yy_s = m.groups()
        mon = _MONTHS.get(mon_s.title())
        if mon is None:
            return None
        try:
            yy = int(yy_s)
            day = int(day_s)
        except ValueError:
            return None
        year = 2000 + yy if yy <= yy_cutoff else 1900 + yy
        try:
            return date(year, mon, day)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

_CHANGE_TYPE_MAP = {
    "new": "new",
    "revised": "revised",
    "deleted": "deleted",
}


def _norm_code(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def _norm_system(value: object) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip().upper()
    if v in {"CPT", "HCPCS"}:
        return v
    return v  # pass through; unknown system values are tolerated


def _norm_change_type(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return _CHANGE_TYPE_MAP.get(value.strip().lower())


def _s(value: object) -> Optional[str]:
    """Coerce to a stripped string, mapping empty / NaN to None."""
    if value is None:
        return None
    if isinstance(value, float):
        # Pandas gives NaN as float
        if pd.isna(value):
            return None
        return str(value)
    if not isinstance(value, str):
        return str(value)
    s = value.strip()
    return s if s else None


def _extract_codes(text: Optional[str]) -> List[str]:
    """Pull all CPT / HCPCS-looking tokens out of a free-form string.

    Preserves first-occurrence order.  Returns [] if ``text`` is None.
    """
    if not text:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for m in _CODE_TOKEN_RX.finditer(text):
        tok = m.group(0).upper()
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out
