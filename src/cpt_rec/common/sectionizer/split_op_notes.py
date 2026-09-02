# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
import argparse
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from cpt_rec.common.constants import (
    MIMIC_DISCHARGE_SECTIONS,
    STANDARD_SECTIONS,
)


@dataclass(frozen=True)
class SectionRegexRule:
    """A regex rule that identifies the start of a section label/header."""
    section_name: str
    regex: str


@dataclass(frozen=True)
class SectionTaxonomy:
    """The three things this splitter needs to know about a corpus's sections.

    ``sections`` alone is not enough: :func:`extract_sections` also has to put
    the text BEFORE the first marker somewhere, and the whole note somewhere
    when no marker matches at all.  Those were the string literals
    ``"Facility Information"`` and ``"Detailed Description"`` -- both VUMC
    operative-note bucket names, and both invisible in a `sections` list.
    Naming them as roles is what makes a second corpus possible.
    """
    key: str
    sections: Tuple[str, ...]
    prefix_section: str      # text before the first marker
    fallback_section: str    # the whole note, when nothing matched

    def __post_init__(self) -> None:
        for role, name in (("prefix_section", self.prefix_section),
                           ("fallback_section", self.fallback_section)):
            if name not in self.sections:
                raise ValueError(
                    f"taxonomy {self.key!r}: {role}={name!r} is not one of its "
                    f"own sections, so extract_sections would write to a column "
                    f"that never reaches the output frame"
                )


VUMC_OP_NOTE = SectionTaxonomy(
    key="vumc_op_note",
    sections=tuple(STANDARD_SECTIONS),
    prefix_section="Facility Information",
    fallback_section="Detailed Description",
)

MIMIC_DISCHARGE = SectionTaxonomy(
    key="mimic_discharge",
    sections=tuple(MIMIC_DISCHARGE_SECTIONS),
    # The block above MIMIC's first header is the admin banner (Name, Unit No,
    # Admission Date, Sex), not a facility line.
    prefix_section="Patient Identification",
    # An unsectioned discharge summary is one narrative, and Hospital Course is
    # the narrative bucket -- the counterpart of VUMC's Detailed Description.
    fallback_section="Hospital Course",
)

TAXONOMIES = {t.key: t for t in (VUMC_OP_NOTE, MIMIC_DISCHARGE)}


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pattern_config(
    pattern_config_path: str,
    taxonomy: SectionTaxonomy = VUMC_OP_NOTE,
) -> List[SectionRegexRule]:
    """
    Load regex rules from a JSON config file.

    Expected schema:
    {
      "version": 1,
      "patterns": [
        {"section": "Pre-operative Diagnosis", "regex": "..."},
        ...
      ]
    }

    ``taxonomy`` must be the one the config was BUILT with.  The mismatch is
    worth erroring loudly on: a MIMIC config read under the VUMC taxonomy is
    exactly the failure that blocks a port to a new note genre, and the message below is what
    tells you which of the two you passed wrongly.
    """
    with open(pattern_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    known = set(taxonomy.sections)
    patterns = config.get("patterns", [])
    rules: List[SectionRegexRule] = []
    for item in patterns:
        section = item.get("section")
        regex = item.get("regex")
        if not section or not regex:
            continue
        if section not in known:
            raise ValueError(
                f"Unknown section in pattern config: {section!r} is not in "
                f"taxonomy {taxonomy.key!r}. Either the config was built with a "
                f"different --sections-file, or this call needs --taxonomy "
                f"{'mimic_discharge' if taxonomy.key == 'vumc_op_note' else 'vumc_op_note'}."
            )
        rules.append(SectionRegexRule(section_name=section, regex=regex))

    if not rules:
        raise ValueError(f"No usable patterns found in config: {pattern_config_path}")

    return rules


def is_single_word_pattern(regex_str: str) -> bool:
    """
    Heuristic classifier:
    - Consider a pattern "single-word" if it does NOT contain multi-token whitespace joins.
    - i.e., it does not contain '\\s+' and does not contain literal spaces.
    This matches your config style where multi-word headers are usually token-joined with \\s+.
    """
    if "\\s+" in regex_str:
        return False
    if " " in regex_str:
        return False
    return True


def compile_rules(rules: Sequence[SectionRegexRule]) -> List[Tuple[str, str, bool, re.Pattern]]:
    """
    Case-sensitive compilation (NO re.IGNORECASE).
    Keep re.MULTILINE so config patterns using ^/$ can work per line.
    """
    compiled: List[Tuple[str, str, bool, re.Pattern]] = []
    for rule in rules:
        single_word = is_single_word_pattern(rule.regex)
        compiled.append(
            (rule.section_name, rule.regex, single_word, re.compile(rule.regex, flags=re.MULTILINE))
        )
    return compiled


def _is_safe_header_match_for_single_word(text: str, start: int, end: int) -> bool:
    """
    Safety filter used ONLY for single-word patterns to avoid splitting mid-narrative.

    Allow if:
      - matched text contains ':' (typical header form), OR
      - match starts at the beginning of the note or the beginning of a line (ignoring leading spaces),
        and the next non-space character after the match is ':' or '\\n' or end-of-string.
    """
    matched = text[start:end]
    if ":" in matched:
        return True

    # Determine start of current line
    line_start = text.rfind("\n", 0, start) + 1  # 0 if not found
    i = line_start
    while i < len(text) and text[i] == " ":
        i += 1
    if i != start:
        return False  # not at trimmed line start

    # Next non-space character after the match
    j = end
    while j < len(text) and text[j] == " ":
        j += 1
    if j >= len(text):
        return True
    if text[j] in [":", "\n"]:
        return True

    return False


def find_non_overlapping_markers(
    text: str,
    compiled_rules: Sequence[Tuple[str, str, bool, re.Pattern]],
) -> List[Tuple[int, int, str, int]]:
    """
    Return ordered, non-overlapping markers: (start, end, section_name, priority_index).

    NOTE: safe-header filtering is applied ONLY to single-word patterns.
    """
    candidates: List[Tuple[int, int, str, int]] = []
    for priority, (section_name, _regex_str, is_single_word, pattern) in enumerate(compiled_rules):
        for m in pattern.finditer(text):
            if is_single_word:
                if not _is_safe_header_match_for_single_word(text, m.start(), m.end()):
                    continue
            candidates.append((m.start(), m.end(), section_name, priority))

    candidates.sort(key=lambda x: (x[0], x[3], -(x[1] - x[0])))

    kept: List[Tuple[int, int, str, int]] = []
    for start, end, section_name, priority in candidates:
        if kept and start < kept[-1][1]:
            prev_start, prev_end, _prev_sec, prev_pri = kept[-1]
            if start == prev_start and priority < prev_pri and end > prev_end:
                kept[-1] = (start, end, section_name, priority)
            continue
        kept.append((start, end, section_name, priority))

    kept.sort(key=lambda x: x[0])
    return kept


def extract_sections(
    note_text: str,
    compiled_rules: Sequence[Tuple[str, str, bool, re.Pattern]],
    taxonomy: SectionTaxonomy = VUMC_OP_NOTE,
) -> Dict[str, str]:
    """
    Split a note into standardized sections using ONLY config-based markers.
    Prefix before the first detected marker -> ``taxonomy.prefix_section``.
    No markers at all -> whole note into ``taxonomy.fallback_section``.
    """
    text = normalize_whitespace(note_text)
    markers = find_non_overlapping_markers(text, compiled_rules)

    sections: Dict[str, str] = {name: "" for name in taxonomy.sections}
    if not markers:
        sections[taxonomy.fallback_section] = text
        return sections

    prefix = normalize_whitespace(text[: markers[0][0]])
    if prefix:
        sections[taxonomy.prefix_section] = prefix

    for i, (start, end, section_name, _priority) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        label = normalize_whitespace(text[start:end])
        content = normalize_whitespace(text[end:next_start])
        if not content:
            continue
        combined = f"{label} {content}".strip()
        sections[section_name] = (
            sections[section_name] + ("\n\n" if sections[section_name] else "") + combined
        ).strip()

    return sections


def write_wide_output(
    input_csv_path: str,
    output_csv_path: str,
    note_text_column: str,
    pattern_config_path: str,
    taxonomy: SectionTaxonomy = VUMC_OP_NOTE,
) -> None:
    df = pd.read_csv(input_csv_path)
    if note_text_column not in df.columns:
        raise ValueError(f"Expected column '{note_text_column}'. Found: {df.columns.tolist()}")

    if not os.path.exists(pattern_config_path):
        raise FileNotFoundError(
            f"Pattern config file not found: {pattern_config_path}\n"
            "Pass the correct path via --pattern-config."
        )

    # Keep ALL original columns, including NOTE_TEXT
    out_base_df = df.copy()
    if "note_index" not in out_base_df.columns:
        out_base_df.insert(0, "note_index", range(len(df)))

    rules = load_pattern_config(pattern_config_path, taxonomy=taxonomy)
    compiled = compile_rules(rules)

    extracted_rows: List[Dict[str, str]] = []
    for note_text in df[note_text_column].fillna("").astype(str).tolist():
        extracted_rows.append(extract_sections(note_text, compiled, taxonomy=taxonomy))

    sections_df = pd.DataFrame(extracted_rows, columns=list(taxonomy.sections))

    # Avoid column name collisions with existing columns
    rename_map = {col: f"{col} (Section)" for col in sections_df.columns if col in out_base_df.columns}
    if rename_map:
        sections_df = sections_df.rename(columns=rename_map)

    out_df = pd.concat([out_base_df.reset_index(drop=True), sections_df.reset_index(drop=True)], axis=1)
    out_df.to_csv(output_csv_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split operative notes into standardized sections (wide CSV output).")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path (wide format).")
    parser.add_argument("--note-text-column", default="NOTE_TEXT", help="Column name containing the raw note text.")
    parser.add_argument(
        "--pattern-config",
        default="outputs/datasets/vumc/sectionizer/header_patterns_config.json",
        help=(
            "Path to the learned header regex config JSON produced by "
            "`cptrec-build-pattern-config`. It is induced from a sample of the "
            "corpus's own headers, so it is a per-corpus artifact and lives "
            "beside that corpus: outputs/datasets/<corpus>/sectionizer/. The "
            "default is VUMC's; MIMIC's discharge taxonomy is in-code "
            "(constants.MIMIC_DISCHARGE_SECTIONS)."
        ),
    )
    parser.add_argument(
        "--taxonomy",
        choices=sorted(TAXONOMIES),
        default="vumc_op_note",
        help=(
            "Which section taxonomy the pattern config was built with. Must "
            "match the --sections-file given to `cptrec-extract-headers` and "
            "`cptrec-build-pattern-config`. Default: vumc_op_note (the 19 "
            "operative-note sections), which leaves existing runs unchanged."
        ),
    )
    parser.add_argument(
        "--sections-file",
        default=None,
        help=(
            "Escape hatch for a taxonomy not in the registry: a JSON list of "
            "section names. Requires --prefix-section and --fallback-section, "
            "since a bare list cannot say where the pre-header block and an "
            "unsectioned note should go."
        ),
    )
    parser.add_argument("--prefix-section", default=None,
                        help="With --sections-file: bucket for text before the first header.")
    parser.add_argument("--fallback-section", default=None,
                        help="With --sections-file: bucket for a note with no header at all.")
    return parser


def resolve_taxonomy(args: argparse.Namespace) -> SectionTaxonomy:
    """Registry name, or an explicit --sections-file triple."""
    if not args.sections_file:
        return TAXONOMIES[args.taxonomy]
    sections = json.loads(Path(args.sections_file).read_text())
    if not isinstance(sections, list) or not all(isinstance(x, str) for x in sections):
        raise SystemExit(f"--sections-file must be a JSON list of strings: {args.sections_file}")
    missing = [f"--{r.replace('_', '-')}" for r, v in
               (("prefix_section", args.prefix_section),
                ("fallback_section", args.fallback_section)) if not v]
    if missing:
        raise SystemExit(
            f"--sections-file also needs {' and '.join(missing)}: a list of section "
            f"names cannot say where the text above the first header goes, nor where "
            f"a note with no header at all goes. Both were VUMC bucket names hard-"
            f"wired into this file until 2026-08-26."
        )
    return SectionTaxonomy(
        key=f"custom:{Path(args.sections_file).name}",
        sections=tuple(sections),
        prefix_section=args.prefix_section,
        fallback_section=args.fallback_section,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    write_wide_output(
        input_csv_path=args.input,
        output_csv_path=args.output,
        note_text_column=args.note_text_column,
        pattern_config_path=args.pattern_config,
        taxonomy=resolve_taxonomy(args),
    )


if __name__ == "__main__":
    main()