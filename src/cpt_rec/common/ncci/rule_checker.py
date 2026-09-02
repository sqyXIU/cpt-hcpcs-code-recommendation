# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Unified NCCI rule checker.

Given a predicted code set, checks PTP incompatibilities, AOC dependency
satisfaction, and MUE feasibility.  Returns structured results suitable for
both constrained decoding and evaluation metrics.

Usage::

    checker = NCCIRuleChecker.from_data_dir("data/ncci")
    result  = checker.check({"29873", "29875", "64450"})
    print(result.hard_ptp_violations)
    print(result.aoc_violations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from cpt_rec.common.ncci.aoc import (
    build_aoc_lookup,
    filter_active_aoc,
    load_aoc_file,
)
from cpt_rec.common.ncci.mue import build_mue_lookup, load_mue_file
from cpt_rec.common.ncci.ptp import (
    build_ptp_lookup,
    filter_active_ptp,
    load_ptp_dir,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PTPViolation:
    """A single PTP edit violation between two codes."""

    col1: str
    col2: str
    ccmi: int
    rationale: str

    @property
    def is_hard(self) -> bool:
        return self.ccmi in (0, 9)

    @property
    def is_modifier_contingent(self) -> bool:
        return self.ccmi == 1


@dataclass
class AOCViolation:
    """An add-on code present without any acceptable primary code."""

    addon_code: str
    required_primaries: Set[str]
    is_contractor_defined: bool


@dataclass
class MUEViolation:
    """A code exceeding its MUE maximum units."""

    code: str
    reported_units: int
    max_units: int


@dataclass
class CheckResult:
    """Aggregated result of all NCCI rule checks on a code set."""

    code_set: Set[str]

    # PTP
    hard_ptp_violations: list[PTPViolation] = field(default_factory=list)
    modifier_contingent_ptp: list[PTPViolation] = field(default_factory=list)

    # AOC
    aoc_violations: list[AOCViolation] = field(default_factory=list)

    # MUE
    mue_violations: list[MUEViolation] = field(default_factory=list)

    @property
    def is_hard_valid(self) -> bool:
        """True if there are no hard PTP violations and no AOC Type-1 violations."""
        has_hard_aoc = any(not v.is_contractor_defined for v in self.aoc_violations)
        return len(self.hard_ptp_violations) == 0 and not has_hard_aoc

    @property
    def n_hard_ptp(self) -> int:
        return len(self.hard_ptp_violations)

    @property
    def n_modifier_contingent_ptp(self) -> int:
        return len(self.modifier_contingent_ptp)

    @property
    def n_aoc_violations(self) -> int:
        return len(self.aoc_violations)

    @property
    def n_mue_violations(self) -> int:
        return len(self.mue_violations)

    def summary(self) -> dict:
        return {
            "n_codes": len(self.code_set),
            "is_hard_valid": self.is_hard_valid,
            "hard_ptp_violations": self.n_hard_ptp,
            "modifier_contingent_ptp": self.n_modifier_contingent_ptp,
            "aoc_violations": self.n_aoc_violations,
            "mue_violations": self.n_mue_violations,
        }


class NCCIRuleChecker:
    """
    Facade that loads PTP, AOC, and MUE data and exposes a single
    :meth:`check` method.
    """

    def __init__(
        self,
        pair_to_ccmi: dict[Tuple[str, str], int],
        pair_to_rationale: dict[Tuple[str, str], str],
        addon_to_primaries: dict[str, Set[str]],
        contractor_defined_addons: Set[str],
        code_to_max_units: dict[str, int],
    ) -> None:
        self._pair_to_ccmi = pair_to_ccmi
        self._pair_to_rationale = pair_to_rationale
        self._addon_to_primaries = addon_to_primaries
        self._contractor_defined_addons = contractor_defined_addons
        self._code_to_max_units = code_to_max_units

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_data_dir(
        cls,
        ncci_dir: Path | str,
        reference_date: date | None = None,
    ) -> "NCCIRuleChecker":
        """
        Load all NCCI resources from a directory with the layout::

            ncci_dir/
              PTP_edits/          (subdirs with .TXT files)
              AOC_edits/          (AOC_*-F-MCR.txt)
              MUE/                (MCR_MUE_*.csv)

        Only edits active on *reference_date* (default: today) are kept.
        """
        ncci_dir = Path(ncci_dir)

        # PTP
        ptp_dir = ncci_dir / "PTP_edits"
        ptp_df = load_ptp_dir(ptp_dir)
        ptp_df = filter_active_ptp(ptp_df, reference_date)
        pair_to_ccmi, pair_to_rationale = build_ptp_lookup(ptp_df)
        LOGGER.info("Active PTP pairs: %d", len(pair_to_ccmi))

        # AOC
        aoc_files = list((ncci_dir / "AOC_edits").glob("AOC_*-F-*.txt"))
        if not aoc_files:
            raise FileNotFoundError(f"No AOC flat files found under {ncci_dir / 'AOC_edits'}")
        aoc_df = load_aoc_file(aoc_files[0])
        aoc_df = filter_active_aoc(aoc_df, reference_date)
        addon_to_primaries, contractor_defined = build_aoc_lookup(aoc_df)
        LOGGER.info(
            "Active AOC: %d addon codes with specific primaries, %d contractor-defined",
            len(addon_to_primaries),
            len(contractor_defined),
        )

        # MUE
        mue_files = list((ncci_dir / "MUE").glob("MCR_MUE_*.csv"))
        if not mue_files:
            raise FileNotFoundError(f"No MUE CSV files found under {ncci_dir / 'MUE'}")
        mue_df = load_mue_file(mue_files[0])
        code_to_max_units = build_mue_lookup(mue_df)
        LOGGER.info("MUE entries: %d", len(code_to_max_units))

        return cls(
            pair_to_ccmi=pair_to_ccmi,
            pair_to_rationale=pair_to_rationale,
            addon_to_primaries=addon_to_primaries,
            contractor_defined_addons=contractor_defined,
            code_to_max_units=code_to_max_units,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    def check_ptp(self, code_set: Set[str]) -> Tuple[list[PTPViolation], list[PTPViolation]]:
        """
        Check all unordered pairs in *code_set* against PTP edits.

        PTP edits are directional (col1, col2), so we check both orderings.
        Returns (hard_violations, modifier_contingent).
        """
        hard: list[PTPViolation] = []
        contingent: list[PTPViolation] = []

        for a, b in combinations(code_set, 2):
            for c1, c2 in [(a, b), (b, a)]:
                ccmi = self._pair_to_ccmi.get((c1, c2))
                if ccmi is None:
                    continue
                rationale = self._pair_to_rationale.get((c1, c2), "")
                v = PTPViolation(col1=c1, col2=c2, ccmi=ccmi, rationale=rationale)
                if v.is_hard:
                    hard.append(v)
                elif v.is_modifier_contingent:
                    contingent.append(v)

        return hard, contingent

    def check_aoc(self, code_set: Set[str]) -> list[AOCViolation]:
        """
        For each add-on code in *code_set*, verify that at least one
        acceptable primary code is also present.

        Type 1 add-ons require an explicit primary from the allowed list.
        Type 3 (contractor-defined) are flagged but treated as weaker.
        """
        violations: list[AOCViolation] = []

        for code in code_set:
            # Type 1: specific primary list
            required = self._addon_to_primaries.get(code)
            if required is not None:
                if not required & code_set:
                    violations.append(
                        AOCViolation(
                            addon_code=code,
                            required_primaries=required,
                            is_contractor_defined=False,
                        )
                    )

            # Type 3: contractor-defined (flag, not hard-block)
            if code in self._contractor_defined_addons:
                # We cannot verify contractor-defined primaries, so only flag
                # if no other code in the set is plausibly a primary
                # (conservative: flag always)
                violations.append(
                    AOCViolation(
                        addon_code=code,
                        required_primaries=set(),
                        is_contractor_defined=True,
                    )
                )

        return violations

    def check_mue(
        self, code_units: Dict[str, int]
    ) -> list[MUEViolation]:
        """
        Check each (code, reported_units) pair against MUE maximums.

        Parameters:
            code_units: {code: reported_units}  — only codes with explicit
                        unit counts need to be included.
        """
        violations: list[MUEViolation] = []
        for code, units in code_units.items():
            max_u = self._code_to_max_units.get(code)
            if max_u is not None and units > max_u:
                violations.append(
                    MUEViolation(code=code, reported_units=units, max_units=max_u)
                )
        return violations

    # ------------------------------------------------------------------
    # Unified check
    # ------------------------------------------------------------------
    def check(
        self,
        code_set: Set[str],
        code_units: Optional[Dict[str, int]] = None,
    ) -> CheckResult:
        """
        Run all applicable NCCI checks on *code_set*.

        Parameters:
            code_set:   set of predicted CPT/HCPCS codes (modifier-agnostic)
            code_units: optional {code: units} for MUE checking; if None,
                        MUE checks are skipped.
        """
        hard_ptp, contingent_ptp = self.check_ptp(code_set)
        aoc_violations = self.check_aoc(code_set)
        mue_violations = self.check_mue(code_units) if code_units else []

        return CheckResult(
            code_set=code_set,
            hard_ptp_violations=hard_ptp,
            modifier_contingent_ptp=contingent_ptp,
            aoc_violations=aoc_violations,
            mue_violations=mue_violations,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def n_ptp_pairs(self) -> int:
        return len(self._pair_to_ccmi)

    @property
    def n_addon_codes(self) -> int:
        return len(self._addon_to_primaries) + len(self._contractor_defined_addons)

    @property
    def n_mue_codes(self) -> int:
        return len(self._code_to_max_units)

    def get_mue(self, code: str) -> Optional[int]:
        """Return the MUE max-units for *code*, or None if not found."""
        return self._code_to_max_units.get(code)

    def get_ptp_ccmi(self, col1: str, col2: str) -> Optional[int]:
        """Return the CCMI for a directed (col1, col2) pair, or None."""
        return self._pair_to_ccmi.get((col1, col2))

    def get_acceptable_primaries(self, addon_code: str) -> Optional[Set[str]]:
        """Return the set of acceptable primary codes for *addon_code*, or None."""
        return self._addon_to_primaries.get(addon_code)
