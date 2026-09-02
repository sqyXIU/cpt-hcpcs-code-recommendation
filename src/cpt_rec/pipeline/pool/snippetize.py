# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Note -> evidence-rich snippets.

Reuses the sentence splitter and windowing from
:mod:`cpt_rec.common.retriever.make_op_note_snippet` but operates
in-memory (no CSV round-trip) and prefers a shortlist of evidence-rich
sections when the note has already been sectionized.

A :class:`Snippet` carries its section tag as a prefix token so the
scorer can learn section-conditioned code relevance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from cpt_rec.common.retriever.make_op_note_snippet import (
    make_overlapping_sentence_windows,
    sent_tokenize_clinical,
)

# Sections most likely to ground procedure codes.  Order matters: when we
# compose a prefix tag we pick the FIRST matching section so that a snippet
# lives under exactly one canonical label.
EVIDENCE_SECTIONS: List[str] = [
    "Procedure(s) Performed",
    "Detailed Description",
    "Findings",
    "Specimens Removed",
    "Implants & Equipment",
    "Indications for Surgery",
]


@dataclass
class Snippet:
    """One sentence-aligned window of note text plus its provenance."""

    note_id: str
    section: str  # canonical section name, or "__whole_note__" as fallback
    snippet_index: int
    start_sentence: int
    end_sentence: int
    n_words: int
    text: str  # *without* the section tag
    snippet_id: str = ""

    def __post_init__(self) -> None:
        if not self.snippet_id:
            slug = self.section.lower().replace(" ", "_").replace("(", "")\
                .replace(")", "").replace("&", "and")
            self.snippet_id = f"{self.note_id}_{slug}_{self.snippet_index:03d}"

    def tagged_text(self) -> str:
        """Text prefixed with a ``[SECTION=...]`` tag for the encoder."""
        return f"[SECTION={self.section}] {self.text}"


def _windows_for_text(
    text: str,
    max_words: int,
    overlap_words: int,
) -> List[dict]:
    sents = sent_tokenize_clinical(text or "")
    if not sents:
        return []
    return make_overlapping_sentence_windows(
        sents, max_words=max_words, overlap_words=overlap_words
    )


def snippets_for_note(
    note_id: str,
    note_text: Optional[str] = None,
    sections: Optional[Dict[str, str]] = None,
    evidence_sections: Sequence[str] = EVIDENCE_SECTIONS,
    max_words: int = 180,
    overlap_words: int = 60,
    max_snippets: Optional[int] = None,
) -> List[Snippet]:
    """
    Build snippets for a single note.

    If ``sections`` (a dict of canonical-section -> text) is provided, we
    iterate through ``evidence_sections`` in order and gather snippets only
    from those sections that have non-empty text.  Otherwise we fall back to
    windowing the whole note under the tag ``__whole_note__``.

    ``max_snippets`` caps the total number returned (per-section allocation
    is proportional).
    """
    out: List[Snippet] = []
    used_any_section = False

    if sections:
        for sec in evidence_sections:
            sec_text = sections.get(sec, "")
            if not sec_text or not str(sec_text).strip():
                continue
            used_any_section = True
            windows = _windows_for_text(str(sec_text), max_words, overlap_words)
            base_idx = len(out)
            for w in windows:
                out.append(
                    Snippet(
                        note_id=str(note_id),
                        section=sec,
                        snippet_index=base_idx + int(w["snippet_index"]),
                        start_sentence=int(w["start_sentence"]),
                        end_sentence=int(w["end_sentence"]),
                        n_words=int(w["n_words"]),
                        text=str(w["snippet_text"]),
                    )
                )

    if not used_any_section:
        text = note_text or (sections or {}).get("Detailed Description", "") or ""
        windows = _windows_for_text(str(text), max_words, overlap_words)
        for w in windows:
            out.append(
                Snippet(
                    note_id=str(note_id),
                    section="__whole_note__",
                    snippet_index=int(w["snippet_index"]),
                    start_sentence=int(w["start_sentence"]),
                    end_sentence=int(w["end_sentence"]),
                    n_words=int(w["n_words"]),
                    text=str(w["snippet_text"]),
                )
            )

    if max_snippets is not None and len(out) > max_snippets:
        out = _reserve_balanced_snippets(out, max_snippets)

    return out


def select_snippet_indices(scores: Sequence[float], k: Optional[int]) -> List[int]:
    """Keep the ``k`` highest-scoring snippets, restoring reading order.

    The S2 evidence-selection stage.  Unlike :func:`_reserve_balanced_snippets`
    -- which ranks by ``n_words`` and therefore encodes no notion of relevance
    -- this ranks by a caller-supplied score, so the evidence budget can be
    spent on the windows that actually look like procedure text rather than on
    the longest ones.

    Returned indices are sorted ascending so ``[SECTION=...]`` provenance and
    snippet ordering still read naturally downstream.
    """
    n = len(scores)
    if k is None or k <= 0 or n <= k:
        return list(range(n))
    ranked = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    return sorted(ranked[:k])


def _reserve_balanced_snippets(snippets: List[Snippet], max_snippets: int) -> List[Snippet]:
    """Down-select ``snippets`` to ``max_snippets`` via a 2-pass reservation.

    1. **Per-section reserve** — keep the single longest window of each
       non-blank section (in section first-appearance order), so a short but
       distinctive section (e.g. a one-line "Implants & Equipment") is never
       starved by a verbose one.
    2. **Global top-up** — fill the remaining budget with the globally longest
       remaining windows.

    Original order is preserved in the result so section provenance (and the
    ``[SECTION=...]`` tag) still reads naturally.
    """
    by_section: Dict[str, List[int]] = {}
    for idx, snippet in enumerate(snippets):
        by_section.setdefault(snippet.section, []).append(idx)

    kept: set = set()
    # Pass 1: one longest window per section.
    for section_idxs in by_section.values():
        if len(kept) >= max_snippets:
            break
        kept.add(max(section_idxs, key=lambda i: snippets[i].n_words))

    # Pass 2: top up with the globally longest remaining windows.
    if len(kept) < max_snippets:
        remaining = sorted(
            (i for i in range(len(snippets)) if i not in kept),
            key=lambda i: -snippets[i].n_words,
        )
        for i in remaining[: max_snippets - len(kept)]:
            kept.add(i)

    return [snippets[i] for i in sorted(kept)]
