# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cpt_rec.common.constants import STANDARD_SECTIONS


def normalize_header_for_counting(header: str) -> str:
    """
    Canonicalize header for counting:
      - strip ends
      - collapse internal whitespace to single spaces
    Keeps spelling/case/punctuation otherwise.
    """
    header = header.strip()
    header = re.sub(r"\s+", " ", header)
    return header


def header_to_regex(header: str) -> str:
    """
    Convert a (possibly misspelled) header string into a robust regex that matches it
    in the splitter’s normalized-note text.

    Strategy:
      - Collapse whitespace for building the regex
      - If header ends with colon (allowing whitespace before it), convert trailing colon to '\\s*:'
      - Escape tokens literally with re.escape
      - Join tokens with '\\s+' to allow any whitespace runs between tokens
      - Add a leading word boundary if the header starts with an alphanumeric char
      - Add trailing '\\s*' so the match can consume spaces after the header but not content
    """
    raw = header.strip()
    raw = re.sub(r"\s+", " ", raw)

    # Treat trailing colon flexibly: "DX:" or "DX :" -> same regex ending
    has_trailing_colon = raw.endswith(":") or raw.endswith(" :")
    base = re.sub(r"\s*:\s*$", "", raw) if has_trailing_colon else raw

    tokens = [t for t in base.split(" ") if t]
    escaped_tokens = [re.escape(t) for t in tokens]

    if escaped_tokens:
        core = r"\s+".join(escaped_tokens)
    else:
        core = re.escape(base)

    if has_trailing_colon:
        core = core + r"\s*:"

    # Optional leading word boundary if starts with alnum (safer vs matching mid-word)
    if re.match(r"^[A-Za-z0-9]", base):
        core = r"\b" + core

    # Allow whitespace after header label (but not content)
    return core + r"\s*"


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


def build_pattern_config(
    input_jsonl_path: str,
    min_count: int,
    top_k_per_section: int,
    resolve_conflicts: bool,
    sections: Optional[Sequence[str]] = None,
) -> Tuple[dict, dict]:
    """
    Returns:
      - config_json: dict with keys {version, generated_at, patterns}
      - stats: dict with useful summaries (counts) for debugging/QA

    ``sections`` defaults to :data:`STANDARD_SECTIONS`, the 19 VUMC
    operative-note buckets.  It is a parameter because this builder allocates
    one Counter per section name up front: with the constant hard-wired it can
    only ever emit the first corpus's vocabulary, so a second corpus with a
    different note genre would have its headers silently dropped (no matching
    key) rather than counted.  Pass the taxonomy the extractor ran with.
    """
    sections = list(sections) if sections else list(STANDARD_SECTIONS)
    # counts[section][canonical_header] = count
    counts: Dict[str, Counter] = {sec: Counter() for sec in sections}
    # representative[section][canonical_header] = first_seen_original_header
    representative: Dict[str, Dict[str, str]] = {sec: {} for sec in sections}

    total_records = 0

    for record in iter_jsonl(input_jsonl_path):
        total_records += 1
        mapping = record.get("standard_section_headers", {})
        if not isinstance(mapping, dict):
            continue

        for sec in sections:
            headers = mapping.get(sec, [])
            if not isinstance(headers, list):
                continue

            for h in headers:
                if not isinstance(h, str):
                    continue
                canon = normalize_header_for_counting(h)
                if not canon:
                    continue

                counts[sec][canon] += 1
                if canon not in representative[sec]:
                    representative[sec][canon] = h  # keep exact first-seen form

    # Optional: resolve conflicts where the same canonical header appears in multiple sections.
    # Keep it only in the section where it appears most frequently.
    if resolve_conflicts:
        header_global: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for sec in sections:
            for canon, c in counts[sec].items():
                header_global[canon].append((sec, c))

        for canon, sec_counts in header_global.items():
            if len(sec_counts) <= 1:
                continue
            sec_counts.sort(key=lambda x: (-x[1], x[0]))
            winner = sec_counts[0][0]
            for sec, _c in sec_counts[1:]:
                if canon in counts[sec]:
                    del counts[sec][canon]
                if canon in representative[sec]:
                    del representative[sec][canon]

    # Build patterns list
    patterns: List[dict] = []
    per_section_kept: Dict[str, List[Tuple[str, int]]] = {}

    for sec in sections:
        items = [(canon, c) for canon, c in counts[sec].items() if c >= min_count]
        # Sort: higher frequency first, then longer header (more specific), then lex
        items.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))

        if top_k_per_section > 0:
            items = items[:top_k_per_section]

        per_section_kept[sec] = items

        for canon, _c in items:
            # use canonical (whitespace-collapsed) for regex building, but store representative exact header in stats
            patterns.append(
                {
                    "section": sec,
                    "regex": header_to_regex(canon),
                }
            )

    config_json = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patterns": patterns,
    }

    # Extra stats (NOT required by downstream loader; just useful for QA)
    stats = {
        "total_records": total_records,
        "min_count": min_count,
        "top_k_per_section": top_k_per_section,
        "resolve_conflicts": resolve_conflicts,
        "kept_counts_by_section": {
            sec: [{"header": representative[sec].get(canon, canon), "count": c} for canon, c in per_section_kept[sec]]
            for sec in sections
        },
    }

    return config_json, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize LLM-extracted headers (JSONL) into a splitter pattern config JSON."
    )
    p.add_argument("--input-jsonl", required=True, help="Path to JSONL from header extraction step.")
    p.add_argument("--output-config", required=True, help="Output JSON config path (for downstream splitter).")
    p.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Keep headers that appear at least this many times (default: 2).",
    )
    p.add_argument(
        "--top-k-per-section",
        type=int,
        default=100,
        help="Max number of patterns to keep per section (0 = no limit). Default: 100.",
    )
    p.add_argument(
        "--resolve-conflicts",
        action="store_true",
        help="If a header appears in multiple sections, keep it only in the most frequent section.",
    )
    p.add_argument(
        "--output-stats",
        default=None,
        help="Optional path to write a QA stats JSON (counts + examples).",
    )
    p.add_argument(
        "--sections-file",
        default=None,
        help="JSON list of section names. Must match the taxonomy the header "
             "extractor ran with. Default: the 19 STANDARD_SECTIONS.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sections = None
    if args.sections_file:
        sections = json.loads(Path(args.sections_file).read_text())
        if not isinstance(sections, list) or not all(isinstance(x, str) for x in sections):
            raise SystemExit(
                f"--sections-file must be a JSON list of strings: {args.sections_file}"
            )

    config_json, stats = build_pattern_config(
        input_jsonl_path=args.input_jsonl,
        min_count=args.min_count,
        top_k_per_section=args.top_k_per_section,
        resolve_conflicts=args.resolve_conflicts,
        sections=sections,
    )

    with open(args.output_config, "w", encoding="utf-8") as f:
        json.dump(config_json, f, ensure_ascii=False, indent=2)

    if args.output_stats:
        with open(args.output_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()