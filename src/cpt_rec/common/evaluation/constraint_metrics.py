# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
NCCI-constraint violation metrics for predicted code sets.

Wraps :class:`cpt_rec.common.ncci.rule_checker.NCCIRuleChecker` and
reports, across a corpus of predictions:

* Rate of notes with ≥1 **hard PTP** violation (CCMI = 0 or 9)
* Rate of notes with ≥1 **modifier-contingent PTP** violation (CCMI = 1)
* Rate of notes with ≥1 **AOC** violation (Type 1 hard / Type 3 contractor)
* Rate of notes with ≥1 **MUE** violation (optional; requires unit counts)

Also reports mean violations per note, which is useful when a single set
produces many PTP pair violations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from cpt_rec.common.ncci.rule_checker import CheckResult, NCCIRuleChecker

LOGGER = logging.getLogger(__name__)


@dataclass
class ConstraintMetrics:
    """Violation rates across a set of predicted code sets."""

    n_notes: int

    # Counts of notes with ≥1 violation of each type
    n_notes_hard_ptp: int
    n_notes_modifier_ptp: int
    n_notes_aoc_hard: int  # Type 1 (specific primary required)
    n_notes_aoc_contractor: int  # Type 3 (contractor-defined, softer flag)
    n_notes_mue: int

    # Total per-pair / per-code violation counts
    n_total_hard_ptp: int
    n_total_modifier_ptp: int
    n_total_aoc_hard: int
    n_total_aoc_contractor: int
    n_total_mue: int

    # Optional breakdown of which pairs/codes violated, capped per violation type
    example_hard_ptp: List[Tuple[str, str]] = field(default_factory=list)
    example_aoc_orphans: List[str] = field(default_factory=list)

    # Optional: rate of notes that pass ALL hard checks (hard PTP + AOC hard)
    n_notes_hard_valid: int = 0

    @property
    def rate_hard_ptp(self) -> float:
        return self.n_notes_hard_ptp / self.n_notes if self.n_notes else 0.0

    @property
    def rate_modifier_ptp(self) -> float:
        return self.n_notes_modifier_ptp / self.n_notes if self.n_notes else 0.0

    @property
    def rate_aoc_hard(self) -> float:
        return self.n_notes_aoc_hard / self.n_notes if self.n_notes else 0.0

    @property
    def rate_aoc_contractor(self) -> float:
        return self.n_notes_aoc_contractor / self.n_notes if self.n_notes else 0.0

    @property
    def rate_mue(self) -> float:
        return self.n_notes_mue / self.n_notes if self.n_notes else 0.0

    @property
    def rate_hard_valid(self) -> float:
        return self.n_notes_hard_valid / self.n_notes if self.n_notes else 0.0

    def to_dict(self, include_examples: bool = True) -> dict:
        d = {
            "n_notes": self.n_notes,
            # rates (primary paper numbers)
            "rate_hard_ptp_violations": round(self.rate_hard_ptp, 6),
            "rate_modifier_ptp_violations": round(self.rate_modifier_ptp, 6),
            "rate_aoc_hard_violations": round(self.rate_aoc_hard, 6),
            "rate_aoc_contractor_violations": round(self.rate_aoc_contractor, 6),
            "rate_mue_violations": round(self.rate_mue, 6),
            "rate_hard_valid": round(self.rate_hard_valid, 6),
            # per-note violation counts
            "n_notes_hard_ptp": self.n_notes_hard_ptp,
            "n_notes_modifier_ptp": self.n_notes_modifier_ptp,
            "n_notes_aoc_hard": self.n_notes_aoc_hard,
            "n_notes_aoc_contractor": self.n_notes_aoc_contractor,
            "n_notes_mue": self.n_notes_mue,
            "n_notes_hard_valid": self.n_notes_hard_valid,
            # total violation counts (many pairs per note possible)
            "n_total_hard_ptp_pairs": self.n_total_hard_ptp,
            "n_total_modifier_ptp_pairs": self.n_total_modifier_ptp,
            "n_total_aoc_hard": self.n_total_aoc_hard,
            "n_total_aoc_contractor": self.n_total_aoc_contractor,
            "n_total_mue": self.n_total_mue,
        }
        if include_examples:
            d["example_hard_ptp_pairs"] = [list(p) for p in self.example_hard_ptp]
            d["example_aoc_orphans"] = self.example_aoc_orphans
        return d


def compute_constraint_metrics(
    pred_sets: Iterable[Set[str]],
    checker: NCCIRuleChecker,
    code_units: Optional[Iterable[Optional[Dict[str, int]]]] = None,
    max_examples: int = 20,
) -> ConstraintMetrics:
    """
    Run the NCCI checker across an iterable of predicted code sets.

    Parameters
    ----------
    pred_sets
        Iterable of predicted code-set per note.
    checker
        Initialized NCCIRuleChecker.
    code_units
        Optional parallel iterable of ``{code: units}`` dicts for MUE checking.
        If None, MUE is skipped.
    max_examples
        Cap for the example lists reported back.
    """
    n_notes = 0
    n_notes_hard_ptp = 0
    n_notes_modifier_ptp = 0
    n_notes_aoc_hard = 0
    n_notes_aoc_contractor = 0
    n_notes_mue = 0
    n_notes_hard_valid = 0

    n_total_hard_ptp = 0
    n_total_modifier_ptp = 0
    n_total_aoc_hard = 0
    n_total_aoc_contractor = 0
    n_total_mue = 0

    example_hard_ptp: List[Tuple[str, str]] = []
    example_aoc_orphans: List[str] = []

    if code_units is None:
        units_iter: Iterable[Optional[Dict[str, int]]] = iter(lambda: None, 1)  # infinite Nones
    else:
        units_iter = iter(code_units)

    for pred in pred_sets:
        n_notes += 1
        units = next(units_iter, None) if code_units is not None else None
        result: CheckResult = checker.check(set(pred), code_units=units)

        # Per-note flags
        if result.hard_ptp_violations:
            n_notes_hard_ptp += 1
        if result.modifier_contingent_ptp:
            n_notes_modifier_ptp += 1
        hard_aoc = [v for v in result.aoc_violations if not v.is_contractor_defined]
        contractor_aoc = [v for v in result.aoc_violations if v.is_contractor_defined]
        if hard_aoc:
            n_notes_aoc_hard += 1
        if contractor_aoc:
            n_notes_aoc_contractor += 1
        if result.mue_violations:
            n_notes_mue += 1
        if result.is_hard_valid:
            n_notes_hard_valid += 1

        # Per-violation totals
        n_total_hard_ptp += len(result.hard_ptp_violations)
        n_total_modifier_ptp += len(result.modifier_contingent_ptp)
        n_total_aoc_hard += len(hard_aoc)
        n_total_aoc_contractor += len(contractor_aoc)
        n_total_mue += len(result.mue_violations)

        # Examples (capped)
        if len(example_hard_ptp) < max_examples:
            for v in result.hard_ptp_violations:
                example_hard_ptp.append((v.col1, v.col2))
                if len(example_hard_ptp) >= max_examples:
                    break
        if len(example_aoc_orphans) < max_examples:
            for v in hard_aoc:
                example_aoc_orphans.append(v.addon_code)
                if len(example_aoc_orphans) >= max_examples:
                    break

    return ConstraintMetrics(
        n_notes=n_notes,
        n_notes_hard_ptp=n_notes_hard_ptp,
        n_notes_modifier_ptp=n_notes_modifier_ptp,
        n_notes_aoc_hard=n_notes_aoc_hard,
        n_notes_aoc_contractor=n_notes_aoc_contractor,
        n_notes_mue=n_notes_mue,
        n_total_hard_ptp=n_total_hard_ptp,
        n_total_modifier_ptp=n_total_modifier_ptp,
        n_total_aoc_hard=n_total_aoc_hard,
        n_total_aoc_contractor=n_total_aoc_contractor,
        n_total_mue=n_total_mue,
        example_hard_ptp=example_hard_ptp,
        example_aoc_orphans=example_aoc_orphans,
        n_notes_hard_valid=n_notes_hard_valid,
    )
