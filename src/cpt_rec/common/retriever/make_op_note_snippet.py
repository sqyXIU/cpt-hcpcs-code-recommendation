#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Wide (section columns) -> Long (snippet rows) for de-identified operative notes.

What it does (for ONE chosen section header/column):
1) Build a GLOBAL vocabulary of procedure codes from the dataset column (default: "CPT_CODES"),
   recognizing:
   - CPT Category I: 5 digits (e.g., 43235)
   - CPT Category II: 4 digits + 'F' (e.g., 2029F)
   - CPT Category III: 4 digits + 'T' (e.g., 0123T)
   - HCPCS Level II: 1 letter + 4 digits (e.g., A0428, J1885, G0283)
   (HCPCS letters fixed to A–Z per request)

2) If ANY of those codes appear in the selected section text, mask/remove them.

3) Sentence/clause split that does NOT rely on newlines (works with "wild string" storage),
   then create overlapping, sentence-aligned windows.

4) Output long-format CSV (one row per snippet), keeping note_id and the *code list*.
"""

from __future__ import annotations

import argparse
import math
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from cpt_rec.common.constants import STANDARD_SECTIONS
from cpt_rec.common.preprocess.code_utils import parse_proc_codes


def build_code_mask_pattern(codes: List[str]) -> Optional[re.Pattern]:
    """Regex that matches any provided code as a standalone token (not inside alphanumeric strings)."""
    codes = [c.upper() for c in codes if c]
    if not codes:
        return None

    codes_sorted = sorted(set(codes), key=lambda x: (-len(x), x))
    return re.compile(
        r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % "|".join(map(re.escape, codes_sorted)),
        flags=re.IGNORECASE,
    )


# -----------------------------
# 2) Sentence/clause splitting
# -----------------------------
def sent_tokenize_clinical(text: str) -> List[str]:
    """
    Robust splitter for op notes stored as a single long string ("wild string").
    - Does NOT rely on newline boundaries.
    - Normalizes escaped newlines / HTML breaks if present.
    - Splits on . ! ? and also ; : when followed by likely new sentence starts.
    - Protects common abbreviations to reduce false splits.
    - Fallback: if almost no punctuation, creates pseudo-sentences by word count.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []

    t = str(text)

    # Normalize common artifacts (escaped newlines, HTML breaks)
    t = re.sub(r"(\\r\\n|\\n|\\r|<br\s*/?>|</p>|<p>)", " ", t, flags=re.IGNORECASE)
    t = t.replace("\u2028", " ").replace("\u2029", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []

    # Protect abbreviations (extend if needed)
    abbrev_patterns = [
        r"\bDr\.", r"\bMr\.", r"\bMs\.", r"\bMrs\.", r"\bSt\.", r"\bvs\.", r"\betc\.",
        r"\be\.g\.", r"\bi\.e\.", r"\ba\.m\.", r"\bp\.m\.", r"\bNo\.", r"\bPt\."
    ]
    for pat in abbrev_patterns:
        t = re.sub(pat, lambda m: m.group(0).replace(".", "<DOT>"), t, flags=re.IGNORECASE)

    # Split on sentence ends OR strong clause boundaries common in op notes
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\[])|(?<=[;:])\s+(?=[A-Z0-9\[])", t
    )
    sents = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]

    # Fallback if minimal punctuation: force sentence-like chunks
    if len(sents) == 1 and len(re.findall(r"\w+", sents[0])) > 300:
        words = re.findall(r"\S+", sents[0])
        step = 40
        sents = [" ".join(words[i:i + step]) for i in range(0, len(words), step)]

    return sents


# -----------------------------
# 3) Overlapping, sentence-aligned windowing
# -----------------------------
def make_overlapping_sentence_windows(
    sentences: List[str],
    max_words: int = 180,
    overlap_words: int = 60,
) -> List[dict]:
    """
    Pack consecutive sentences into overlapping windows based on a word budget.
    Window boundaries are always sentence boundaries.
    """
    wc = [len(re.findall(r"\w+", s)) for s in sentences]
    n = len(sentences)

    windows: List[dict] = []
    start = 0
    idx = 0

    while start < n:
        total = 0
        end = start

        # Ensure at least one sentence per window
        while end < n and (total + wc[end] <= max_words or end == start):
            total += wc[end]
            end += 1
            if total >= max_words:
                break

        windows.append(
            dict(
                snippet_index=idx,
                start_sentence=start,
                end_sentence=end - 1,
                n_sentences=end - start,
                n_words=total,
                snippet_text=" ".join(sentences[start:end]).strip(),
            )
        )
        idx += 1

        if end >= n:
            break

        # Move start backward to create overlap (sentence-aligned).
        # Guarantee forward progress: if the new start would not advance
        # past the current start (which happens when a single sentence
        # already exceeds overlap_words, e.g. a 200-word run-on sentence
        # with max_words=180), skip overlap and jump to ``end`` so the
        # outer loop terminates.
        overlap = 0
        new_start = end
        while new_start > start + 1 and overlap < overlap_words:
            new_start -= 1
            overlap += wc[new_start]
        if new_start <= start:
            new_start = end
        start = new_start

    return windows


# -----------------------------
# 4) Wide -> Long for one section
# -----------------------------
def section_snippets_wide_to_long(
    wide_csv_path: str,
    section_name: str,
    output_csv_path: str,
    note_id_col: str = "note_index",
    code_col: str = "CPT_CODES",
    mask_mode: str = "mask",          # "mask" or "remove"
    mask_token: str = "[PROC_CODE]",  # ignored if mask_mode == "remove"
    max_words: int = 180,
    overlap_words: int = 60,
) -> pd.DataFrame:
    """
    1) Read wide-format CSV.
    2) Choose ONE section (column).
    3) Mask/remove ANY codes found in the dataset's code column if they appear in section text.
    4) Sentence/clause split + overlapping sentence-aligned windows.
    5) Write long-format CSV (one row per snippet) including note_id + code list.
    """
    df = pd.read_csv(wide_csv_path)

    if section_name not in df.columns:
        raise ValueError(
            f"Section '{section_name}' not found in CSV columns.\n"
            f"Expected one of these standardized headers:\n{STANDARD_SECTIONS}"
        )
    if note_id_col not in df.columns:
        raise ValueError(f"note_id_col '{note_id_col}' not found in CSV columns.")
    if code_col not in df.columns:
        raise ValueError(f"code_col '{code_col}' not found in CSV columns.")

    # Global code vocabulary from dataset (CPT + HCPCS)
    all_codes: List[str] = []
    for s in df[code_col].dropna():
        all_codes.extend(parse_proc_codes(s))
    code_pat = build_code_mask_pattern(all_codes)

    records: List[dict] = []
    section_slug = re.sub(r"[^A-Za-z0-9]+", "_", section_name).strip("_").lower()

    for _, r in df.iterrows():
        note_id = r[note_id_col]
        codes_raw = "" if pd.isna(r.get(code_col, np.nan)) else str(r.get(code_col))
        codes_list = parse_proc_codes(codes_raw)
        codes_norm = "|".join(codes_list)  # keep the list (pipe-separated) for downstream

        text = r.get(section_name, "")
        if pd.isna(text) or str(text).strip() == "":
            continue
        text = str(text)

        # 1) Mask/remove codes appearing in section text
        if code_pat is not None:
            if mask_mode == "remove":
                text = code_pat.sub("", text)
                text = re.sub(r"\s{2,}", " ", text).strip()
            else:
                text = code_pat.sub(mask_token, text)

        # 2) Sentence/clause split (no newline reliance)
        sentences = sent_tokenize_clinical(text)
        if not sentences:
            continue

        # 3) Overlapping, sentence-aligned windows
        windows = make_overlapping_sentence_windows(
            sentences, max_words=max_words, overlap_words=overlap_words
        )

        for w in windows:
            snippet_id = f"{note_id}_{section_slug}_{w['snippet_index']:03d}"
            records.append(
                dict(
                    note_id=note_id,
                    section=section_name,
                    codes_raw=codes_raw,      # original string from code_col
                    proc_codes=codes_norm,    # parsed list as pipe-separated
                    snippet_index=w["snippet_index"],
                    snippet_id=snippet_id,
                    start_sentence=w["start_sentence"],
                    end_sentence=w["end_sentence"],
                    n_sentences=w["n_sentences"],
                    n_words=w["n_words"],
                    snippet_text=w["snippet_text"],
                )
            )

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(output_csv_path, index=False)
    return out_df


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create sentence-aligned overlapping snippets from one section column."
    )
    parser.add_argument("--input", required=True, help="Path to wide-format CSV")
    parser.add_argument("--output", required=True, help="Path to output long-format CSV")
    parser.add_argument("--section", required=True, help="One standardized section header (column name)")
    parser.add_argument("--note_id_col", default="note_index", help="Column for note id (default: note_index)")
    parser.add_argument("--code_col", default="CPT_CODES", help="Column containing CPT+HCPCS codes (default: CPT_CODES)")
    parser.add_argument("--mask_mode", choices=["mask", "remove"], default="mask", help="Mask or remove codes in section text")
    parser.add_argument("--mask_token", default="[PROC_CODE]", help="Replacement token if mask_mode=mask")
    parser.add_argument("--max_words", type=int, default=180, help="Max words per snippet window")
    parser.add_argument("--overlap_words", type=int, default=60, help="Approx overlap size (words), sentence-aligned")

    args = parser.parse_args()

    df_long = section_snippets_wide_to_long(
        wide_csv_path=args.input,
        section_name=args.section,
        output_csv_path=args.output,
        note_id_col=args.note_id_col,
        code_col=args.code_col,
        mask_mode=args.mask_mode,
        mask_token=args.mask_token,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )

    print(f"Wrote {len(df_long)} snippet rows -> {args.output}")


if __name__ == "__main__":
    main()