#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Corpus registry + label-completeness contract for the two-corpus benchmark.

Why this file exists
--------------------
Every method in this repo already takes ``--notes`` / ``--kb`` / ``--train-stats``
as paths, so running a method on a second corpus is a matter of *pointing* it
somewhere else.  What is **not** interchangeable is what the resulting numbers
mean.  The two corpora differ on three axes that silently invalidate a naive
side-by-side table:

===================  ==============================  ==============================
axis                 VUMC operative notes            MIMIC-IV discharge summaries
===================  ==============================  ==============================
gold standard        billed CPT for the encounter    ``hcpcsevents`` rows, which
                     — reference-complete            cover only 145,364 of 331,793
                                                     noted admissions: **absent
                                                     code ≠ absent procedure**
split axis           temporal (train ≤ 2023,          per-patient date shifts make
                     val 2024, test 2025, drift       a temporal split meaningless
                     2026) — supports drift           → patient-disjoint split,
                                                     **no drift split exists**
document genre       operative note, sectionized      discharge summary, no
                     by 19 canonical headers          operative section headers
===================  ==============================  ==============================

The completeness axis is the dangerous one.  Under a partially recorded gold
standard a correct-but-unrecorded prediction is scored as a false positive, and
the penalty scales with how many codes a system emits — so precision, F1 and
exact-match are not merely noisy, they are **not comparable between systems**
with different output cardinality.  Recall-family and rank-position quantities
survive as *lower bounds* (they are gold-anchored; the only distortion is
displacement in the ranking, which every system suffers).  Prediction-only
quantities — NCCI validity, set size, latency — are unaffected entirely.

That reasoning is encoded here as :data:`PARTIAL_GOLD_SUPPRESSIONS` and applied
by :mod:`cpt_rec.benchmark.export`, so the guarantee is a
property of the pipeline rather than a sentence in a paper draft that a later
table can quietly contradict.

Usage
-----
::

    from cpt_rec.benchmark.corpora import get_corpus
    c = get_corpus("mimic")
    c.split_path("test")            # outputs/datasets/mimic_iv/test.csv
    c.quotable("metrics", "set.micro_f1")     # -> (False, "<reason>")
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from cpt_rec.common.constants import STANDARD_SECTIONS

# ---------------------------------------------------------------------------
# label-completeness contract
# ---------------------------------------------------------------------------

#: Dotted key paths (``fnmatch`` patterns, rooted at the JSON top level) that a
#: *partially recorded* gold standard cannot support, each with the reason that
#: is written into the export provenance.  ``kind`` is the artifact family:
#: ``metrics`` = ``metrics.json``, ``rank`` = ``rank_metrics.json``,
#: ``sibling`` = ``sibling_report.json``.
PARTIAL_GOLD_SUPPRESSIONS: Dict[str, Dict[str, str]] = {
    # These are GLOBS, not a list of key names, and that is deliberate.  The
    # 2026-08-27 export enumerated names and leaked 71 precision-derived cells
    # for MIMIC -- set.example_f1, bootstrap.macro_f1.*, bootstrap.example_f1.*
    # and set.per_category.*.{exact_match,example_f1,jaccard_mean} all had no
    # matching entry.  `*` crosses dots in fnmatch, so a token-shaped pattern
    # matches at ANY depth and a newly added metric is suppressed by default
    # rather than shipped by default.  Erring toward suppression is the safe
    # direction here: a wrongly hidden cell is visible in the export summary,
    # a wrongly shipped one is a number someone can quote.
    "metrics": {
        "set.*precision*": "precision counts correct-but-unrecorded codes as FP",
        "set.*f1*": "F1 inherits the precision bias, which scales with set size",
        "set.*exact_match*": "an exact set match is impossible against a partial gold set",
        "set.*jaccard*": "Jaccard penalises unrecorded-correct codes in the union",
        "set.n_fp": "an FP here may be a correct code the source table never recorded",
        "set.*.n_fp": "see set.n_fp",
        "per_bin.*precision*": "see set.*precision*",
        "per_bin.*f1*": "see set.*f1*",
        "per_bin.*exact_match*": "see set.*exact_match*",
        "per_bin.*jaccard*": "see set.*jaccard*",
        "per_bin.*.n_fp": "see set.n_fp",
        "bootstrap.*precision*": "CI of a suppressed point estimate",
        "bootstrap.*f1*": "CI of a suppressed point estimate",
        "bootstrap.*exact_match*": "CI of a suppressed point estimate",
        "bootstrap.*jaccard*": "CI of a suppressed point estimate",
    },
    "rank": {
        "precision_at": "precision@B charges the system for unrecorded-correct codes",
        "map_at": "MAP averages precision at gold ranks — same bias",
        "ndcg_at": "nDCG's gain is gold-anchored but its ideal ranking is not",
        "auc_pr": "the precision axis is not interpretable under a partial gold set",
        "bootstrap.*precision*": "CI of a suppressed point estimate",
        "bootstrap.map*": "CI of a suppressed point estimate",
        "bootstrap.ndcg*": "CI of a suppressed point estimate",
    },
    "sibling": {
        # The sibling RATES are the T2-5 read-out and carry caveats rather than
        # suppression (see PARTIAL_GOLD_CAVEATS below).  Its `micro` block is a
        # different thing: a plain precision/F1 pair with exactly the bias of
        # set.micro_precision.  The 2026-08-27 export shipped
        # micro.precision = 0.3889 for MIMIC with no annotation at all, because
        # this table was empty.  Suppressed on the same grounds as `metrics`.
        "*precision*": "precision counts correct-but-unrecorded codes as FP",
        "*f1*": "F1 inherits the precision bias, which scales with set size",
        # Every pattern below must be ROOT-FREE (leading ``*``).  The 2026-08-28
        # audit found that "*n_fp" caught ``by_train_bin.*.n_fp`` but missed the
        # FP-set *composition* fields, which do not end in ``n_fp``: the export
        # shipped confusion.fp, headline.n_fp_sibling / n_fp_substitution and
        # the whole primary_addon block for MIMIC with no annotation at all.
        # Do NOT collapse these into "*n_fp_*": that glob also matches
        # "substitutio(n_fp_)rate" and would suppress a rate we deliberately
        # caveat instead.  Accidental substring hits are the recurring bug here.
        "*n_fp": "an FP here may be a correct code the source table never recorded",
        "*n_fp_sibling": "an FP subset count inherits the FP-set bias, see *n_fp",
        "*n_fp_substitution": "an FP subset count inherits the FP-set bias, see *n_fp",
        "confusion.fp": "see *n_fp",
        "*addon_fp*": "an add-on FP share is a composition of the biased FP set",
        "gate_verdict": (
            "the verdict string asserts an FP-share conclusion this corpus "
            "cannot support"
        ),
    },
    # `aux` was added on 2026-08-27 with NO entry here, on the reasoning that an
    # aux payload is a cardinality or label-distribution aggregate and so cannot
    # carry a gold-matching quantity.  That reasoning was wrong within the hour:
    # `threshold.json` is a threshold SWEEP, and a sweep carries the objective it
    # was tuned on -- the first MIMIC aux export shipped best_micro_f1 = 0.4308
    # plus a 9-point grid of micro_f1 values.  An empty table is not a safe
    # default; it is the same defect as the 2026-08-27 rank/sibling tables.
    #
    # Aux holds arbitrary shapes, so every pattern here is ROOT-FREE and
    # token-shaped: it must fire at any depth, including inside a list, where
    # scrub() renders the path as `grid[].micro_f1`.  This deliberately mirrors
    # the token set a manual pre-commit scan would look for, so the exporter
    # enforces exactly that rather than something adjacent to it.
    "aux": {
        "*precision*": "precision counts correct-but-unrecorded codes as FP",
        "*f1*": "F1 inherits the precision bias, which scales with set size",
        "*exact_match*": "an exact set match is impossible against a partial gold set",
        "*jaccard*": "Jaccard penalises unrecorded-correct codes in the union",
        "*ndcg*": "nDCG's gain is gold-anchored but its ideal ranking is not",
        "*auc_pr*": "the precision axis is not interpretable under a partial gold set",
        "*map_at*": "MAP averages precision at gold ranks -- same bias",
        # explicit, not "*n_fp_*": that glob also matches
        # "substitutio(n_fp_)rate", a rate we caveat rather than suppress.
        "*n_fp": "an FP here may be a correct code the source table never recorded",
        "*n_fp_sibling": "an FP subset count inherits the FP-set bias, see *n_fp",
        "*n_fp_substitution": "an FP subset count inherits the FP-set bias, see *n_fp",
        "confusion.fp": "see *n_fp",
        "*addon_fp*": "an add-on FP share is a composition of the biased FP set",
    },
}

#: Quantities that are biased but still reportable, with the direction stated.
#: The export copies these through and attaches the caveat.
PARTIAL_GOLD_CAVEATS: Dict[str, Dict[str, str]] = {
    "metrics": {
        "set.micro_recall": "LOWER BOUND: unrecorded-correct codes displace gold in the ranking",
        "per_bin.*.micro_recall": "LOWER BOUND, see set.micro_recall",
    },
    "rank": {
        "recall_at": "LOWER BOUND: unrecorded-correct codes occupy top-B slots",
        "coverage_at": "LOWER BOUND, see recall_at",
        "family.*": "LOWER BOUND, see recall_at",
        "pool_ceiling": "LOWER BOUND, see recall_at",
    },
    "sibling": {
        "micro.recall": "LOWER BOUND, see set.micro_recall",
        "*sibling_fp_rate": (
            "BIASED UPWARD: unrecorded-correct codes are disproportionately "
            "siblings of a recorded code, so this over-states near-miss "
            "confusion. Read as an upper bound, never as a point estimate "
            "inside the VUMC band."
        ),
        "*substitution_fp_rate": "BIASED UPWARD, see sibling_fp_rate",
    },
}

#: Prediction-only quantities: unaffected by gold completeness, fully quotable
#: on either corpus.  Listed so the benchmark tables can say *why* these columns
#: are the ones that carry an absolute claim on both corpora.
COMPLETENESS_INDEPENDENT: Tuple[str, ...] = (
    "constraints.*",          # NCCI PTP / MUE / AOC validity of the emitted set
    "set.n_pred_codes",       # output cardinality
    "n_candidates_mean",
    "shown_at",               # codes actually shown at each budget
)


# ---------------------------------------------------------------------------
# corpus records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Corpus:
    """One benchmark corpus: where it lives, what it looks like, what it supports."""

    key: str
    display_name: str
    access: str                       # "private" | "credentialed-open"
    document_genre: str
    root: Path
    split_files: Mapping[str, str]
    stats_file: str
    note_id_col: Optional[str] = None      # None => autodetect (baselines.common)
    note_text_col: Optional[str] = None
    gold_code_col: str = "proc_codes"
    #: What the CORPUS offers, not what any one system reads (MIMIC offers
    #: none).  The systems disagree about which subset is "the evidence":
    #: M3/M5 read 3, M6 reads 6, and M1/M4 read the whole note.  Which
    #: subset M6 reads is a flag (``--section-cols``), not a separate
    #: system.  See ``reader_input_note``.
    section_cols: Tuple[str, ...] = ()
    #: Which text each system family actually reads on this corpus.  Stated
    #: because it is a design axis of the benchmark, not an accident.
    reader_input_note: str = ""
    patient_col: Optional[str] = None
    split_policy: str = "temporal"
    label_completeness: str = "reference-complete"
    completeness_note: str = ""
    #: What the gold label space actually contains, after any corpus-construction
    #: filter.  A source table that bills a level of service alongside the
    #: operation does not define the same task as a coder's procedure set, and
    #: the reader of a cross-corpus table has to be told which one they are
    #: looking at.
    label_space_note: str = ""
    #: may note text be sent to a third-party API endpoint?
    external_api_allowed: bool = False
    external_api_note: str = ""
    #: where note-free aggregates may be committed
    export_root: Path = Path("outputs/benchmark/top5")
    #: artifacts that must never leave ``root``
    restricted_tree: bool = False
    citation: str = ""

    # -- paths ------------------------------------------------------------
    def split_path(self, split: str) -> Path:
        if split not in self.split_files:
            raise KeyError(
                f"corpus '{self.key}' has no split '{split}' "
                f"(available: {sorted(self.split_files)})"
            )
        return self.root / self.split_files[split]

    @property
    def splits(self) -> Tuple[str, ...]:
        return tuple(self.split_files)

    @property
    def stats_path(self) -> Path:
        return self.root / self.stats_file

    def export_dir(self, system: str, split: str) -> Path:
        return self.export_root / self.key / system / split

    # -- metric contract --------------------------------------------------
    @property
    def partial_gold(self) -> bool:
        return self.label_completeness != "reference-complete"

    def quotable(self, kind: str, dotted_path: str) -> Tuple[bool, str]:
        """
        Is ``dotted_path`` of artifact family ``kind`` quotable for this corpus?

        Returns ``(ok, reason)``.  ``reason`` is the suppression reason when
        ``ok`` is False, the caveat when the value is quotable-with-a-caveat,
        and ``""`` when the value is unconditionally quotable.
        """
        if not self.partial_gold:
            return True, ""

        probes = [(kind, dotted_path)]
        # metrics.json nests the ENTIRE ranking contract under a top-level
        # `ranking` key, but the "rank" patterns are rooted at the top level of
        # rank_metrics.json, so they never fired for it.  That is why the
        # 2026-08-27 export dropped precision@B from rank_metrics.json and
        # shipped the identical numbers as metrics.json's ranking.precision_at
        # and ranking.auc_pr.  Probe the same path against the rank table with
        # the prefix stripped.
        if kind == "metrics" and dotted_path.startswith("ranking."):
            probes.append(("rank", dotted_path[len("ranking."):]))

        # every suppression is consulted before any caveat, so a suppression in
        # one table always beats a caveat in the other
        for probe_kind, path in probes:
            for pattern, reason in PARTIAL_GOLD_SUPPRESSIONS.get(probe_kind, {}).items():
                if _path_matches(path, pattern):
                    return False, reason
        for probe_kind, path in probes:
            for pattern, caveat in PARTIAL_GOLD_CAVEATS.get(probe_kind, {}).items():
                if _path_matches(path, pattern):
                    return True, caveat
        return True, ""


def _path_matches(dotted_path: str, pattern: str) -> bool:
    """``a.b.c`` matches ``a.b`` (prefix), ``a.*.c`` and ``a.b*`` (fnmatch)."""
    if dotted_path == pattern or dotted_path.startswith(pattern + "."):
        return True
    return fnmatch.fnmatch(dotted_path, pattern) or fnmatch.fnmatch(
        dotted_path, pattern + ".*"
    )


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

VUMC = Corpus(
    key="vumc",
    display_name="VUMC operative notes",
    access="private",
    document_genre="operative note (sectionized)",
    root=Path("outputs/datasets/vumc"),
    split_files={
        "train": "train_eval_sectioned.csv",
        "val": "val_eval_sectioned.csv",
        "test": "test_eval_sectioned.csv",
        "drift": "drift_eval_sectioned.csv",
    },
    stats_file="code_frequency_stats.csv",
    section_cols=tuple(STANDARD_SECTIONS),   # the sectionizer emits all 19
    reader_input_note=(
        "M3/M5 read 3 sections (Procedure(s) Performed + Detailed Description "
        "+ Findings) at 2048 tokens; M6 reads 6 "
        "(those 3 + Specimens Removed + Implants & Equipment + Indications "
        "for Surgery) when given --section-cols; M1/M4 and the shipped "
        "whole-note M6 arm read the whole note. 6-section vs whole-note inside "
        "M6 measured 0.47 pp apart, below the 0.70 pp nondeterminism floor."
    ),
    patient_col="PAT_MRN_ID",
    split_policy="temporal (train ≤2023-12 · val 2024 · test 2025 · drift 2026-01→02-05)",
    label_completeness="reference-complete",
    completeness_note="billed CPT/HCPCS for the encounter; the professional coder's set is the reference",
    label_space_note=(
        "as billed, unfiltered: 93.78% CPT-I surgery at 1.822 codes/note, with a "
        "procedural tail of intraoperative imaging, neuromonitoring, anaesthesia "
        "and J/L/S/T codes; E/M is 4 instances in 35,768"
    ),
    external_api_allowed=True,
    external_api_note="institutional Azure OpenAI deployment under the existing agreement",
    citation="Vanderbilt University Medical Center, 2017-11 → 2026-02 (not redistributable)",
)

MIMIC = Corpus(
    key="mimic",
    display_name="MIMIC-IV discharge summaries",
    access="credentialed-open",
    document_genre="discharge summary (no operative sections)",
    root=Path("outputs/datasets/mimic_iv"),
    split_files={
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
    },
    stats_file="code_frequency_stats.csv",
    note_id_col="note_id",
    note_text_col="note_text",
    # The evidence view for the sectioned ablation arm, chosen to
    # mirror VUMC's reader-side THREE (Procedure(s) Performed / Detailed
    # Description / Findings) rather than the verifier's six -- discharge
    # summaries have no analogue for Specimens Removed or Implants & Equipment.
    #   Procedures      -> the explicit procedure list (49 induced patterns)
    #   Hospital Course -> where operative and interventional events are
    #                      narrated in a discharge summary (100 patterns)
    #   Results         -> endoscopy and imaging reports; the most-confused
    #                      family in the sibling report is ERCP (43261/43262/
    #                      43264/43274), which is reported here
    # DELIBERATELY EXCLUDED: "Past Surgical History". It names procedures from
    # PRIOR admissions, which must not be coded for this encounter, and its 19
    # patterns fire cleanly -- it is the taxonomy's most likely FP source.
    # This populates columns for the ablation arm only; the shipped corpus is
    # still whole-note, and the parity claim still rests on that.  Feed them to
    # M6 with ``cptrec-verifier-predict --section-cols`` to run the arm.
    section_cols=("Procedures", "Hospital Course", "Results"),
    reader_input_note=(
        "every system reads the whole note, and the corpus ships no section "
        "columns by default, so the whole note — not the 3 evidence sections — is "
        "the only "
        "input policy the two corpora can share. Discharge summaries do carry "
        "headers: the sectionizer can build a sectioned copy against the curated "
        "26-section C-CDA taxonomy (constants.MIMIC_DISCHARGE_SECTIONS, LOINC "
        "18842-5). That is an ablation arm which populates section columns for "
        "itself; it does not change this default, and the parity claim rests on "
        "the shipped corpus. On the M6 side that arm is the same binary with "
        "--section-cols set, not a second system."
    ),
    patient_col="subject_id",
    split_policy="patient-disjoint 70/10/20 on subject_id, seed 42 "
                 "(per-patient date shifts make a temporal split meaningless)",
    label_completeness="partially-recorded",
    completeness_note=(
        "hcpcsevents covers 145,364 of 331,793 noted admissions; an absent code "
        "does not mean an absent procedure"
    ),
    label_space_note=(
        "hcpcsevents is an admission-level BILLING table: 61.6% surgical at 2.523 "
        "codes/enc as recorded. cptrec-bench-build-mimic --code-filter procedural "
        "(the default) drops CPT-I E/M 99202-99499 and E/M-equivalent HCPCS-II "
        "facility codes (G0378 observation-per-hour and kin) -> 99.48% surgical at "
        "1.562 codes/enc, comparable to VUMC. A level of service is not inferable "
        "from a procedure narrative; see corpus_stats.json codes.dropped_codes"
    ),
    external_api_allowed=True,
    external_api_note=(
        "Authorized by the data owner 2026-08-26: MIMIC-IV note text MAY be sent "
        "to Azure OpenAI, so the frontier rows (gpt-5.6-sol) run on this corpus "
        "exactly as they do on VUMC. PhysioNet names Azure OpenAI a compliant "
        "route for credentialed data PROVIDED human review of prompts and "
        "completions is disabled on the deployment (their Limited Access / "
        "responsible-use form) -- that condition is what makes this permitted, "
        "and it is a property of the deployment, not of this flag. Row-level "
        "MIMIC artifacts still never leave outputs/datasets/mimic_iv/ and still "
        "reach the repo only through cptrec-bench-export."
    ),
    restricted_tree=True,
    citation="MIMIC-IV v3.1 (PhysioNet, credentialed access, DUA)",
)

REGISTRY: Dict[str, Corpus] = {c.key: c for c in (VUMC, MIMIC)}

#: The methods that are meant to run on BOTH corpora under one recipe.  Anything
#: outside this list is a corpus-specific row and must be labelled as such in the
#: benchmark table.
#:
#: The frontier rows are portable: both corpora run the same API model, so the
#: strongest baseline in the study appears on both and enters the rank
#: correlation instead of being footnoted out of it.  ``m5_sft_local`` is
#: portable for the same reason -- the LoRA SFT is trained on each corpus's own
#: train split under one whole-note recipe and scored through the unmodified M3
#: CLI, so the supervision control exists on both sides.  Any locally-served
#: generative variant is an optional secondary, not a core row.
PORTABLE_CORE: Tuple[str, ...] = (
    "m1_bm25_knn",
    "m2_label_attention",
    "m3_zeroshot_frontier",
    "m4_rag_frontier",
    "m5_sft_local",
    "m6_retrieve_verify",
    "kb_index",
)


#: Paper label for each system key.  The keys are the on-disk contract: they
#: name the metrics directories every run writes and that ``cptrec-bench-export``
#: reads.  They are deliberately spelled the same way the paper labels the rows,
#: so a directory listing and a table read alike.
SYSTEM_LABELS: Dict[str, str] = {
    "m1_bm25_knn": "M1",
    "m2_label_attention": "M2",
    "m3_zeroshot_frontier": "M3",
    "m4_rag_frontier": "M4",
    "m5_sft_local": "M5",
    "m6_retrieve_verify": "M6",
}


def system_label(key: str) -> str:
    """Paper label for *key*, or the key itself when it has no label.

    Unlabelled keys are things that run but are not compared rows in the paper
    (``kb_index``, the KB retrieval index M6 reuses; and any corpus-specific
    secondary, such as a locally-served variant of an API baseline).
    """
    return SYSTEM_LABELS.get(key, key)


def get_corpus(key: str) -> Corpus:
    try:
        return REGISTRY[key]
    except KeyError:
        raise SystemExit(
            f"unknown corpus '{key}' (known: {', '.join(sorted(REGISTRY))})"
        ) from None


def describe(key: str) -> str:
    """One-paragraph corpus card, for the paper's data section."""
    c = get_corpus(key)
    lines = [
        f"{c.display_name} [{c.key}]",
        f"  access            : {c.access}",
        f"  genre             : {c.document_genre}",
        f"  root              : {c.root}",
        f"  splits            : {', '.join(c.splits)}",
        f"  split policy      : {c.split_policy}",
        f"  gold completeness : {c.label_completeness}",
        f"                      {c.completeness_note}",
        f"  gold label space  : {c.label_space_note}",
        f"  section columns   : {len(c.section_cols)} (offered by the corpus)",
        f"  what systems read : {c.reader_input_note or 'n/a'}",
        f"  external API      : {'allowed' if c.external_api_allowed else 'FORBIDDEN'}"
        f" — {c.external_api_note}",
        f"  citation          : {c.citation}",
    ]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Describe a registered benchmark corpus.")
    ap.add_argument("corpus", nargs="?", choices=sorted(REGISTRY) + ["all"], default="all")
    args = ap.parse_args()
    keys = sorted(REGISTRY) if args.corpus == "all" else [args.corpus]
    for k in keys:
        print(describe(k))
        print()


if __name__ == "__main__":  # pragma: no cover
    main()
