# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Shared candidate ledger — the central typed object of the next version.

Each ``(note_id, code)`` candidate is one :class:`CandidateLedgerEntry`. Agents
The pipeline stages fill in their fields in order; the final
selector and auditor read the completed record. The schema is the contract that
lets agents stay decoupled — an agent depends on *fields*, not on the previous
agent's internals.

Stdlib-only by design: the ledger is a serialization boundary, not a model.
JSONL is the on-disk form (one entry per line), reconcilable 1:1 with the
``predictions.npz`` companion
(``note_ids`` / ``candidate_codes`` / ``probs``).

One row per (note, candidate) so a prediction can be traced back to the
snippet that produced it without re-running the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class CandidateLedgerEntry:
    """One candidate code for one note, enriched across the agent chain."""

    # --- identity (A3 candidate generation) ---
    note_id: str
    code: str

    # --- provenance (A3): which sources proposed this candidate, and where ---
    candidate_sources: Dict[str, bool] = field(default_factory=dict)
    # Normalized per-source rank feats (verifier's ``*_rank`` features; 1.0 = absent),
    # so float rather than the raw integer rank.
    source_ranks: Dict[str, float] = field(default_factory=dict)
    source_confidences: Dict[str, float] = field(default_factory=dict)

    # --- evidence (A5 candidate-conditioned retrieval) ---
    best_snippet: Optional[str] = None
    section: Optional[str] = None

    # --- code knowledge (A4 code cards / KB) ---
    code_description: Optional[str] = None
    code_family: Optional[str] = None
    sibling_codes: List[str] = field(default_factory=list)

    # --- scoring (A6 cross-encoder verifier) ---
    verifier_score: Optional[float] = None

    # --- decisions (A7 sibling arbiter, A8 cardinality selector) ---
    selected_pre_rule: Optional[bool] = None
    arbiter_winner: Optional[bool] = None
    rejected_reason: Optional[str] = None

    # --- rules (A9 NCCI check + repair trace) ---
    ncci_flags: List[str] = field(default_factory=list)
    selected_post_rule: Optional[bool] = None

    # --- audit (A10) ---
    explanation: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateLedgerEntry":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def write_ledger(entries: Iterable[CandidateLedgerEntry], path: Path) -> int:
    """Write entries as JSONL; returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_ledger(path: Path) -> Iterator[CandidateLedgerEntry]:
    """Stream entries from a JSONL ledger."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield CandidateLedgerEntry.from_dict(json.loads(line))
