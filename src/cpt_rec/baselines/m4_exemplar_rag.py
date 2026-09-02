#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
M4 (``m4_rag_frontier``) — LLM + BM25 exemplars (retrieval-augmented).

The reported M4 row is the API backend.  ``--backend local`` runs the same
prompt against a locally-served model; that is an optional secondary, not a
core row, so give it its own key (e.g. ``m4_rag_local``) if you run it.

Augment the GPT prompt with:

* the top-``k`` BM25 training-note neighbors (truncated and tagged with
  their gold code sets), diversified by code-set Jaccard so we don't
  burn prompt budget on near-duplicate exemplars, and
* the descriptions of every code that appears in those neighbors' gold
  sets — passed through the canonical procedure-code shape filter
  (``code_utils.is_valid_proc_code``) — i.e. the union of candidate
  codes the LLM is allowed to choose from.

The LLM is then asked to return the codes for the procedures actually
performed in the *target* note, restricted to the candidate set.

This is a clean RAG comparator for M3 (zero-shot LLM): same model, same
prompt structure, but with retrieved evidence.

Concurrency
-----------
Like M3, M4 fans out across a thread pool gated by a global
``FixedIntervalRateLimiter``.

Backends — see ``m3_zeroshot_llm`` for env-var requirements.

CLI
---

::

    cptrec-m4-rag \\
        --notes outputs/datasets/vumc/test_eval.csv \\
        --index outputs/baselines/m1_bm25/index.pkl \\
        --kb data/kb/codes_with_ranges.csv \\
        --out outputs/baselines/m4_rag_frontier/predictions/test.csv \\
        --top-k 5 --max-exemplar-tokens 400 --max-note-tokens 1500 \\
        --backend azure --deployment-name gpt-5.3-chat \\
        --max-workers 16 --rpm 250
"""

from __future__ import annotations

import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm

from cpt_rec.baselines.m3_zeroshot_llm import (
    assemble_sectionized_text,
)
from cpt_rec.baselines.bm25_index import TrainNoteBM25Index
from cpt_rec.baselines.common import (
    Prediction,
    apply_seed_and_limit,
    complete_shortlist,
    crosswalk_codes_for_date,
    dedupe_keep_order,
    load_notes_for_prediction,
    log_note_budget,
    log_prediction_stats,
    stats_sidecar,
    maybe_load_code_history,
    parse_as_of,
    rank_by_self_consistency,
    truncate_text_by_tokens,
    write_predictions,
    write_scores_npz,
)
from cpt_rec.baselines.llm import (
    AzureOpenAIBackend,
    EchoBackend,
    FixedIntervalRateLimiter,
    LLMBackend,
    parse_selected_codes,
)
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.common.preprocess.code_utils import is_valid_proc_code

LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert medical coder.  You will be given:\n"
    "  (a) a TARGET operative note,\n"
    "  (b) several similar operative notes from the training corpus, each "
    "labeled with its known CPT/HCPCS codes (EXAMPLES), and\n"
    "  (c) descriptions of every candidate CPT/HCPCS code that appears in "
    "any example (CANDIDATE CODES).\n\n"
    "Your task is to return ONLY the codes whose described procedure is "
    "actually performed in the TARGET note.\n\n"
    "Rules:\n"
    "- Choose only from the CANDIDATE CODES list.  Do NOT invent new codes.\n"
    "- Do NOT emit any modifier (e.g. -59, -RT).  Base codes only.\n"
    "- Multiple codes are common when multiple distinct procedures are done.\n"
    "- Omit codes you are unsure about.\n"
    'Return a JSON object: {"selected": ["<code1>", "<code2>", ...]}'
)


def _system_prompt(shortlist_k: Optional[int] = None) -> str:
    """System prompt, optionally in *matched-budget shortlist* mode.

    ``shortlist_k=None`` returns ``SYSTEM_PROMPT`` unchanged, so the default
    path is byte-identical to every previously published M4 run.  With a
    budget, the two rules that push the model toward a small, precision-
    favouring set ("omit codes you are unsure about") are replaced by a
    fixed-cardinality ranked instruction, so the system can be scored at the
    same review budget ``B`` as a ranker's top-B shortlist.
    """
    if shortlist_k is None:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        .replace(
            "Your task is to return ONLY the codes whose described procedure "
            "is actually performed in the TARGET note.\n\n",
            f"Your task is to return the {shortlist_k} candidate codes MOST "
            "LIKELY to be billed for the TARGET note, ranked most likely "
            "first.  A human coder will review your shortlist, so include "
            "plausible codes rather than only certain ones.\n\n",
        )
        .replace(
            "- Omit codes you are unsure about.\n",
            f"- Return EXACTLY {shortlist_k} codes, ordered most likely "
            "first.  Do not omit uncertain codes — rank them lower.\n",
        )
    )


def _selected_json_hint(shortlist_k: Optional[int] = None) -> str:
    if shortlist_k is None:
        return 'Return JSON: {"selected": ["<code>", ...]}'
    return (
        f'Return JSON with exactly {shortlist_k} codes, best first: '
        '{"selected": ["<code>", ...]}'
    )


def _diversify_exemplars(
    neighbors: Sequence[Tuple[str, float, Set[str]]],
    target_k: int,
    jaccard_cap: float = 0.8,
) -> List[Tuple[str, float, Set[str]]]:
    """
    Greedy code-set Jaccard diversification.

    BM25 top-k often returns several near-duplicate notes from the same
    encounter style; their gold code sets overlap heavily and they all
    push the LLM toward the same vote.  We accept the top neighbor and
    then admit a candidate only if its gold-code-set Jaccard against
    every already-accepted exemplar is ``<= jaccard_cap``.  Cheap and
    visibly diverse without changing the retrieval contract.
    """
    picked: List[Tuple[str, float, Set[str]]] = []
    for cand in neighbors:
        if len(picked) >= target_k:
            break
        _, _, c_codes = cand
        too_similar = False
        for _, _, p_codes in picked:
            union = c_codes | p_codes
            if not union:
                continue
            jacc = len(c_codes & p_codes) / len(union)
            if jacc > jaccard_cap:
                too_similar = True
                break
        if not too_similar:
            picked.append(cand)
    return picked


def _format_exemplars(
    neighbors: Sequence[Tuple[str, float, Set[str]]],
    note_text_lookup: Callable[[str], str],
    max_exemplar_tokens: int,
) -> str:
    blocks: List[str] = []
    for i, (nid, score, code_set) in enumerate(neighbors, start=1):
        text = note_text_lookup(nid) or ""
        snippet = truncate_text_by_tokens(
            text.strip(), max_tokens=max_exemplar_tokens
        )
        codes_str = ", ".join(sorted(code_set)) if code_set else "(none)"
        blocks.append(
            f"--- EXAMPLE {i} (similarity={score:.2f}) ---\n"
            f"Codes: {codes_str}\n"
            f"Note: {snippet}"
        )
    return "\n\n".join(blocks)


def _format_candidates(
    candidate_codes: Sequence[str],
    kb: CodeKnowledgeBase,
) -> str:
    lines: List[str] = []
    for code in sorted(candidate_codes):
        desc = kb.short_description(code) or kb.description(code) or ""
        desc = desc.replace("\n", " ").strip()
        if len(desc) > 220:
            desc = desc[:220] + "..."
        lay = (kb.lay_term(code) or "").replace("\n", " ").strip()
        lay_clip = ""
        if lay and lay.lower() != desc.lower():
            if len(lay) > 80:
                lay = lay[:80] + "..."
            lay_clip = f"  (lay: {lay})"
        lines.append(f"- {code}: {desc}{lay_clip}")
    return "\n".join(lines)


def build_user_prompt(
    target_note: str,
    exemplars_block: str,
    candidates_block: str,
    shortlist_k: Optional[int] = None,
) -> str:
    return (
        "TARGET OPERATIVE NOTE:\n"
        f"{target_note}\n\n"
        "EXAMPLES (similar past notes with known codes):\n"
        f"{exemplars_block}\n\n"
        "CANDIDATE CODES (candidates the model may choose from):\n"
        f"{candidates_block}\n\n"
        + _selected_json_hint(shortlist_k)
    )


# ---------------------------------------------------------------------------
# Prompt token budgeting — enforced before every LLM call so a single
# pathological note doesn't blow up an Azure deployment's TPM quota.
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Best-effort token count via ``tiktoken``; whitespace fallback."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def _fit_to_budget(
    target_note: str,
    nbrs: Sequence[Tuple[str, float, Set[str]]],
    note_text_lookup: Callable[[str], str],
    candidate_codes: Sequence[str],
    kb: CodeKnowledgeBase,
    *,
    max_total_tokens: int,
    max_exemplar_tokens: int,
    min_target_note_tokens: int = 600,
    min_candidates: int = 10,
    min_exemplars: int = 1,
) -> Tuple[str, str, str]:
    """
    Return ``(target_note, exemplars_block, candidates_block)`` whose
    total token count is below ``max_total_tokens``.

    Trim order (cheapest signal-loss first):
      1) drop trailing exemplars one at a time (down to min_exemplars);
      2) drop trailing candidate codes one at a time (down to min_candidates);
      3) shorten per-exemplar token budget;
      4) finally truncate the target note (down to min_target_note_tokens).

    The budget is intentionally conservative — it caps deployment TPM
    pressure, not the model's own context limit.  Real failures we've
    seen on Azure ``gpt-5.3-chat`` start clustering when concurrent
    workers push aggregate TPM past the deployment's quota; a per-call
    cap is the only thing that bounds it from the client side.
    """
    cur_exemplar_tokens = max_exemplar_tokens
    nbrs_list = list(nbrs)
    cands_list = list(candidate_codes)
    target = target_note

    def _build_blocks() -> Tuple[str, str, int]:
        ex_block = _format_exemplars(nbrs_list, note_text_lookup, cur_exemplar_tokens)
        cand_block = _format_candidates(cands_list, kb)
        prompt = build_user_prompt(target, ex_block, cand_block)
        return ex_block, cand_block, _estimate_tokens(prompt)

    # Initial estimate.
    ex_block, cand_block, total = _build_blocks()
    if total <= max_total_tokens:
        return target, ex_block, cand_block

    LOGGER.debug(
        "Prompt %d tokens > budget %d; trimming exemplars first",
        total, max_total_tokens,
    )

    # 1) Drop exemplars from the tail.
    while total > max_total_tokens and len(nbrs_list) > min_exemplars:
        nbrs_list.pop()
        ex_block, cand_block, total = _build_blocks()
    # 2) Drop candidate codes from the tail.
    while total > max_total_tokens and len(cands_list) > min_candidates:
        cands_list.pop()
        ex_block, cand_block, total = _build_blocks()
    # 3) Shorten per-exemplar budget.
    while total > max_total_tokens and cur_exemplar_tokens > 100:
        cur_exemplar_tokens = max(100, int(cur_exemplar_tokens * 0.7))
        ex_block, cand_block, total = _build_blocks()
    # 4) Last resort: truncate the target note.
    if total > max_total_tokens:
        # Shrink the target note in halves until we fit (or hit the floor).
        cur_target_tokens = _estimate_tokens(target)
        while total > max_total_tokens and cur_target_tokens > min_target_note_tokens:
            cur_target_tokens = max(min_target_note_tokens, int(cur_target_tokens * 0.7))
            target = truncate_text_by_tokens(target, max_tokens=cur_target_tokens)
            ex_block, cand_block, total = _build_blocks()

    if total > max_total_tokens:
        LOGGER.warning(
            "Prompt still %d tokens after full trim (budget=%d, "
            "exemplars=%d, candidates=%d); sending anyway",
            total, max_total_tokens, len(nbrs_list), len(cands_list),
        )
    else:
        LOGGER.debug(
            "Trimmed prompt to %d tokens (budget=%d, "
            "exemplars=%d, candidates=%d)",
            total, max_total_tokens, len(nbrs_list), len(cands_list),
        )
    return target, ex_block, cand_block


def _sectionized_text_lookup(
    sectionized_csv: Path,
    label: str,
) -> Dict[str, str]:
    """Map note_id -> the three evidence sections, exactly as M3 assembles them.

    M4 has always been pointed at ``*_eval_sectioned.csv``, but that file is a
    **superset** of the raw export: ``split_op_notes.write_wide_output`` keeps
    every original column (``NOTE_TEXT`` included) and *appends* the section
    columns.  ``load_notes_for_prediction`` resolves ``NOTE_TEXT`` first, so
    M4 has been reading the **whole raw note** — head-truncated — while M3 and
    M5 read three sections.  This lookup is what makes the two comparable.
    """
    sec_df = pd.read_csv(sectionized_csv, dtype=str)
    if "NOTE_ID" in sec_df.columns:
        sec_df = sec_df.rename(columns={"NOTE_ID": "note_id"})
    if "note_id" not in sec_df.columns:
        raise SystemExit(
            f"{sectionized_csv} has no NOTE_ID/note_id column; it does not "
            "look like a wide-format sectionized CSV."
        )
    out: Dict[str, str] = {}
    for nid, group in sec_df.groupby(sec_df["note_id"].astype(str)):
        text = assemble_sectionized_text(group.iloc[0])
        if text:
            out[str(nid)] = text
    if not out:
        raise SystemExit(
            f"{sectionized_csv} yielded no evidence-section text for any note; "
            "expected the wide-format output of `cptrec-split-op-notes`."
        )
    LOGGER.info(
        "%s: evidence-section text for %d/%d notes from %s "
        "(notes with no evidence section keep their raw text)",
        label, len(out), len(sec_df), sectionized_csv,
    )
    return out


def _build_exemplar_text_lookup(
    index: TrainNoteBM25Index,
    train_csv_for_text: Optional[Path],
    note_id_col: Optional[str],
    note_text_col: Optional[str],
    train_sectionized_csv: Optional[Path] = None,
) -> Callable[[str], str]:
    """
    Resolve neighbor note IDs to readable text for the LLM prompt.

    Preference order:
      1) ``--train-csv-for-text`` (original scrubbed text from disk),
      2) ``index.raw_texts`` baked into the BM25 index pickle (v2+),
      3) Re-joined tokenized corpus — lossy fallback we warn about.
    """
    nid_to_text: Dict[str, str] = {}
    if train_csv_for_text is not None:
        train_df = load_notes_for_prediction(
            train_csv_for_text,
            note_id_col=note_id_col,
            note_text_col=note_text_col,
        )
        nid_to_text = dict(
            zip(train_df["note_id"].astype(str), train_df["note_text"])
        )
        LOGGER.info("Loaded raw exemplar text for %d train notes", len(nid_to_text))
    elif index.raw_texts is not None:
        nid_to_text = dict(zip(index.note_ids, index.raw_texts))
        LOGGER.info(
            "Using exemplar text from index (raw_texts present, %d entries)",
            len(nid_to_text),
        )
    else:
        LOGGER.warning(
            "M4 has no raw exemplar text available (index lacks raw_texts "
            "and --train-csv-for-text was not provided); falling back to a "
            "lossy bag-of-words reconstruction.  Rebuild the index with "
            "the current `cptrec-m1-bm25-knn build-index` to fix this."
        )

    if train_sectionized_csv is not None:
        nid_to_text = {
            **nid_to_text,
            **_sectionized_text_lookup(train_sectionized_csv, "M4 exemplars"),
        }

    nid_to_corpus_idx = {nid: i for i, nid in enumerate(index.note_ids)}

    def lookup(nid: str) -> str:
        if nid in nid_to_text:
            return str(nid_to_text[nid])
        i = nid_to_corpus_idx.get(nid)
        if i is None:
            return ""
        return " ".join(index.corpus[i])

    return lookup


#: Response-filter accounting, mirroring M3's ``_KB_FILTER_TALLY``.  ``named``
#: is what the model wrote (post-dedupe); ``out_of_candidates`` / ``out_of_kb``
#: are what each filter level would remove.  Both are tallied on every run
#: regardless of which level is active, so the cost of the filter is reported
#: even when the filter is in force.
_RESP_FILTER_TALLY: Dict[str, int] = {
    "named": 0, "out_of_candidates": 0, "out_of_kb": 0, "notes_hit": 0,
}

#: ``candidates`` — keep only what retrieval offered (the historical default,
#: and M4's definition).  ``kb`` — keep anything in the KB vocabulary, i.e.
#: exactly M3's filter.  That equalises the FILTER; equalising the INPUT also
#: needs ``--sectionized-csv`` (M4 otherwise reads the whole raw note).
#: ``none`` — keep whatever the model named; a code retrieval never surfaced
#: scores as a false positive instead of vanishing.
CANDIDATE_FILTER_LEVELS = ("candidates", "kb", "none")


def reset_response_filter_tally() -> None:
    for k in _RESP_FILTER_TALLY:
        _RESP_FILTER_TALLY[k] = 0


def _apply_response_filter(
    codes: Sequence[str],
    candidate_codes: Set[str],
    kb_codes: Set[str],
    level: str,
) -> List[str]:
    """Filter a parsed response at ``level``; tally all levels either way."""
    n_out_cand = sum(1 for c in codes if c not in candidate_codes)
    n_out_kb = sum(1 for c in codes if c not in kb_codes)
    _RESP_FILTER_TALLY["named"] += len(codes)
    _RESP_FILTER_TALLY["out_of_candidates"] += n_out_cand
    _RESP_FILTER_TALLY["out_of_kb"] += n_out_kb
    if n_out_cand:
        _RESP_FILTER_TALLY["notes_hit"] += 1
    if level == "candidates":
        return [c for c in codes if c in candidate_codes]
    if level == "kb":
        return [c for c in codes if c in kb_codes]
    return list(codes)


def log_response_filter_tally(
    label: str = "M4", level: str = "candidates"
) -> Dict[str, float]:
    """Report what the response filter removed (or would have removed)."""
    named = _RESP_FILTER_TALLY["named"]
    n_cand = _RESP_FILTER_TALLY["out_of_candidates"]
    n_kb = _RESP_FILTER_TALLY["out_of_kb"]
    LOGGER.info(
        "[%s] response filter=%s: model named %d codes; %d (%.1f%%) outside "
        "the retrieved candidates, %d (%.1f%%) outside the KB; %d "
        "note-samples had at least one out-of-candidate code",
        label, level, named,
        n_cand, 100.0 * n_cand / max(1, named),
        n_kb, 100.0 * n_kb / max(1, named),
        _RESP_FILTER_TALLY["notes_hit"],
    )
    return {
        "candidate_filter_level": level,
        "codes_named_by_model": float(named),
        "codes_outside_candidates": float(n_cand),
        "pct_codes_outside_candidates": round(100.0 * n_cand / max(1, named), 2),
        "codes_outside_kb": float(n_kb),
        "pct_codes_outside_kb": round(100.0 * n_kb / max(1, named), 2),
        "note_samples_with_a_drop": float(_RESP_FILTER_TALLY["notes_hit"]),
    }


def _score_one_note(
    note_id: str,
    note_text: str,
    index: TrainNoteBM25Index,
    kb: CodeKnowledgeBase,
    backend: LLMBackend,
    lookup: Callable[[str], str],
    top_k: int,
    max_note_tokens: int,
    max_exemplar_tokens: int,
    diversify_jaccard_cap: float,
    kb_codes: set,
    max_prompt_tokens: int,
    history=None,
    as_of=None,
    shortlist_k: Optional[int] = None,
    self_consistency: int = 1,
    pad_to_k: bool = False,
    candidate_filter: str = "candidates",
) -> Prediction:
    target = truncate_text_by_tokens(note_text, max_tokens=max_note_tokens)

    # Over-fetch and then diversify down to top_k so we have headroom for
    # the Jaccard cap to actually filter out near-duplicates.
    raw_nbrs = index.neighbors(target, top_k=max(top_k * 3, top_k + 5))
    nbrs = _diversify_exemplars(
        raw_nbrs, target_k=top_k, jaccard_cap=diversify_jaccard_cap
    )

    # Crosswalk neighbor gold codes into the target's date frame so a
    # 2018 retired-by-2026 code is replaced by its current substitute
    # (when one is resolvable) before reaching the LLM as a candidate.
    if history is not None and as_of is not None:
        nbrs = [
            (nid, score, crosswalk_codes_for_date(
                code_set, history=history, as_of=as_of, keep_unresolved=False,
            ))
            for nid, score, code_set in nbrs
        ]

    candidate_codes: Set[str] = set()
    for _nid, _score, code_set in nbrs:
        candidate_codes.update(code_set)
    # Two layers: KB membership AND the canonical procedure-code shape gate.
    candidate_codes = {
        c for c in candidate_codes
        if c in kb_codes and is_valid_proc_code(c)
    }

    if not candidate_codes:
        return Prediction(note_id=note_id, codes=[], scores=None)

    # Retrieval ranking over the same candidates the model is shown: each code
    # scores the summed similarity of the neighbours that carry it (a kNN
    # vote).  Used ONLY as the --pad-to-k tail source; the prompt block itself
    # stays code-sorted so the model is given no ordering hint.
    _vote: Dict[str, float] = {}
    for _nid, _score, code_set in nbrs:
        for _c in code_set:
            if _c in candidate_codes:
                _vote[_c] = _vote.get(_c, 0.0) + float(_score)
    ranked_candidates = sorted(
        candidate_codes, key=lambda c: (-_vote.get(c, 0.0), c)
    )

    # Token-budgeted prompt assembly: trim exemplars → candidates → note
    # until the full prompt fits below ``max_prompt_tokens``.  This caps
    # per-call TPM so a pathological note can't blow up the deployment.
    target, exemplars_block, candidates_block = _fit_to_budget(
        target_note=target,
        nbrs=nbrs,
        note_text_lookup=lookup,
        candidate_codes=sorted(candidate_codes),
        kb=kb,
        max_total_tokens=max_prompt_tokens,
        max_exemplar_tokens=max_exemplar_tokens,
    )
    user_prompt = build_user_prompt(
        target, exemplars_block, candidates_block, shortlist_k=shortlist_k
    )

    n_samples = max(1, int(self_consistency))
    try:
        if n_samples > 1:
            responses = backend.chat_n(
                _system_prompt(shortlist_k), user_prompt, n_samples
            )
        else:
            responses = [backend.chat(_system_prompt(shortlist_k), user_prompt)]
    except Exception as exc:
        LOGGER.error("LLM call failed for note %s: %s", note_id, exc)
        responses = [""]

    if n_samples > 1:
        samples = [
            _apply_response_filter(
                dedupe_keep_order(parse_selected_codes(r)),
                candidate_codes, kb_codes, candidate_filter,
            )
            for r in responses
        ]
        kept, scores = rank_by_self_consistency(samples)
        if shortlist_k is not None:
            kept, scores = complete_shortlist(
                kept, scores, shortlist_k,
                pad_pool=ranked_candidates if pad_to_k else None,
            )
        return Prediction(note_id=note_id, codes=kept, scores=scores or None)

    selected = dedupe_keep_order(parse_selected_codes(responses[0]))
    kept = _apply_response_filter(
        selected, candidate_codes, kb_codes, candidate_filter
    )
    if shortlist_k is not None:
        # Truncate always; pad only when --pad-to-k is explicitly asked for.
        # The default stays “report the size the generator actually emitted”,
        # because filling the tail from the candidate pool measures the
        # retriever rather than the generator.
        if pad_to_k:
            _s = [float(len(kept) - i) for i in range(len(kept))]
            kept, _s = complete_shortlist(
                kept, _s, shortlist_k, pad_pool=ranked_candidates
            )
            return Prediction(note_id=note_id, codes=kept, scores=_s or None)
        kept = kept[:shortlist_k]
    return Prediction(note_id=note_id, codes=kept, scores=None)


def _dump_generation_order_npz(
    predictions: List[Prediction], out_npz: Path
) -> None:
    """Write a ranked NPZ whose order is the model's generation order.

    Generation order IS the model's ranking (in shortlist mode it is
    instructed to be; in set mode it is the best available proxy).
    Descending integer scores preserve that order for the review-budget
    suite; ``pool_ceiling`` next to R@B exposes the small pool.

    Called on BOTH exits of :func:`predict_b5` -- including the resume
    "nothing to do" early return.  A completed run re-launched only to add
    ``--dump-scores-npz`` must still produce the NPZ, otherwise the only way
    to obtain rankings for an already-paid-for pass is to re-issue every API
    call.  Generation order survives the CSV round-trip because
    ``write_predictions`` joins ``pred.codes`` in order and never sorts.
    """
    write_scores_npz(
        (
            (
                pred.note_id,
                pred.codes,
                pred.scores if pred.scores is not None
                else [float(len(pred.codes) - i)
                      for i in range(len(pred.codes))],
            )
            for pred in predictions
        ),
        out_npz,
    )


def predict_b5(
    notes_csv: Path,
    index: TrainNoteBM25Index,
    kb: CodeKnowledgeBase,
    out_csv: Path,
    backend: LLMBackend,
    top_k: int = 3,
    max_note_tokens: int = 1000,
    max_exemplar_tokens: int = 250,
    max_prompt_tokens: int = 6000,
    diversify_jaccard_cap: float = 0.8,
    seed: int = 42,
    limit: Optional[int] = None,
    max_workers: int = 16,
    train_csv_for_text: Optional[Path] = None,
    history_changes: Optional[Path] = None,
    history_deleted: Optional[Path] = None,
    kb_csv: Optional[Path] = None,
    as_of_col: str = "PROCEDURE_DATE",
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    checkpoint_every: int = 25,
    resume: bool = True,
    shortlist_k: Optional[int] = None,
    self_consistency: int = 1,
    pad_to_k: bool = False,
    dump_scores_npz: Optional[Path] = None,
    candidate_filter: str = "candidates",
    sectionized_csv: Optional[Path] = None,
    train_sectionized_csv: Optional[Path] = None,
) -> int:
    """Drive M4 end-to-end with a thread pool.

    ``sectionized_csv`` / ``train_sectionized_csv`` swap the target notes' and
    the exemplars' text for M3's three evidence sections.  Both default to
    ``None``, which is M4's historical behaviour: the **whole raw note**,
    head-truncated to ``max_note_tokens`` (target) / ``max_exemplar_tokens``
    (exemplars).  Pointing ``--notes`` at ``*_eval_sectioned.csv`` does *not*
    do this on its own — that file still carries the original ``NOTE_TEXT``.

    ``candidate_filter`` sets how hard the model's response is filtered:
    ``candidates`` (default, historical) keeps only what retrieval offered,
    ``kb`` keeps anything in the KB vocabulary (exactly M3's filter), and
    ``none`` keeps whatever the model named.
    """
    if candidate_filter not in CANDIDATE_FILTER_LEVELS:
        raise SystemExit(
            f"--candidate-filter must be one of {CANDIDATE_FILTER_LEVELS}, "
            f"got {candidate_filter!r}"
        )
    reset_response_filter_tally()
    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)
    LOGGER.info("Loaded %d target notes from %s", len(df), notes_csv)

    lookup = _build_exemplar_text_lookup(
        index=index,
        train_csv_for_text=train_csv_for_text,
        note_id_col=note_id_col,
        note_text_col=note_text_col,
        train_sectionized_csv=train_sectionized_csv,
    )

    history, _ = maybe_load_code_history(
        history_changes=history_changes,
        history_deleted=history_deleted,
        kb_csv=kb_csv,
    )
    if history is not None and as_of_col not in df.columns:
        LOGGER.warning(
            "M4 history requested but as_of column %r not in CSV; "
            "candidate crosswalk will be a no-op", as_of_col,
        )

    # Validation vocabulary: when history is loaded, accept any code that
    # is *either* in the active 2026 KB *or* in CodeHistory.all_codes()
    # (i.e. ever existed).  Without this, a code that was valid on the
    # target note's PROCEDURE_DATE but retired by 2026 — and which the
    # neighbor crosswalk left as-is because it's still active on that
    # date — would be dropped by the final candidate gate.
    kb_codes = set(kb.codes)
    if history is not None:
        before = len(kb_codes)
        kb_codes = kb_codes | set(history.all_codes())
        LOGGER.info(
            "M4: validation vocab = active KB ∪ CodeHistory.all_codes() "
            "(%d → %d codes; +%d historical/deleted)",
            before, len(kb_codes), len(kb_codes) - before,
        )
    all_note_ids = df["note_id"].astype(str).tolist()
    note_texts = dict(zip(all_note_ids, df["note_text"].astype(str).tolist()))
    log_note_budget(
        "M4", max_note_tokens,
        "sections:3" if sectionized_csv is not None else "whole-note",
        covered_by=f"exemplars capped separately at {max_exemplar_tokens} tok",
    )
    if sectionized_csv is not None:
        # Notes with no evidence section keep their raw text, mirroring M3.
        note_texts.update(_sectionized_text_lookup(sectionized_csv, "M4 target"))
    else:
        LOGGER.info(
            "M4 target text = the WHOLE RAW NOTE, head-truncated to %d tokens "
            "(no --sectionized-csv).  M3 and M5 read 3 evidence sections, so "
            "any M3/M4 comparison at this setting differs in INPUT as well as "
            "in prompt.", max_note_tokens,
        )
    note_as_of: Dict[str, "object"] = {}
    if history is not None and as_of_col in df.columns:
        note_as_of = {
            str(nid): parse_as_of(d)
            for nid, d in zip(df["note_id"].astype(str), df[as_of_col])
        }

    # Resume: if the output CSV already has rows, treat them as DONE so
    # a crashed long-running job can be re-launched without re-issuing
    # successful calls.  Permanent-failure rows (empty pred_codes from a
    # previous attempt) are also treated as done — pass --no-resume to
    # redo them after Azure stabilizes.
    order: Dict[str, Prediction] = {}
    if resume and Path(out_csv).is_file():
        try:
            existing = pd.read_csv(out_csv, dtype=str).fillna("")
            for _, r in existing.iterrows():
                nid = str(r["note_id"])
                cells = str(r.get("pred_codes", ""))
                codes = [c for c in cells.split("|") if c]
                order[nid] = Prediction(note_id=nid, codes=codes, scores=None)
            LOGGER.info(
                "M4 resume: loaded %d previously-completed notes from %s "
                "(use --no-resume to redo them)",
                len(order), out_csv,
            )
        except Exception as exc:
            LOGGER.warning(
                "M4 resume: could not parse %s (%s); starting fresh",
                out_csv, exc,
            )

    todo_ids = [nid for nid in all_note_ids if nid not in order]
    if not todo_ids:
        LOGGER.info(
            "M4: nothing to do — all %d notes already done.",
            len(all_note_ids),
        )
        predictions = [order[nid] for nid in all_note_ids]
        log_prediction_stats(predictions, label="M4", budget_k=shortlist_k,
                         out_path=stats_sidecar(out_csv))
        if dump_scores_npz is not None:
            _dump_generation_order_npz(predictions, dump_scores_npz)
        return len(order)

    LOGGER.info(
        "M4: %d/%d notes to score (%d already done from previous run)",
        len(todo_ids), len(all_note_ids), len(order),
    )
    write_lock = threading.Lock()

    def _job(note_id: str) -> Prediction:
        return _score_one_note(
            note_id=note_id,
            note_text=note_texts.get(note_id, ""),
            index=index,
            kb=kb,
            backend=backend,
            lookup=lookup,
            top_k=top_k,
            max_note_tokens=max_note_tokens,
            max_exemplar_tokens=max_exemplar_tokens,
            diversify_jaccard_cap=diversify_jaccard_cap,
            candidate_filter=candidate_filter,
            kb_codes=kb_codes,
            max_prompt_tokens=max_prompt_tokens,
            history=history,
            as_of=note_as_of.get(note_id),
            shortlist_k=shortlist_k,
            self_consistency=self_consistency,
            pad_to_k=pad_to_k,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {pool.submit(_job, nid): nid for nid in todo_ids}
        for fut in tqdm(as_completed(future_to_id),
                        total=len(future_to_id), desc="M4 GPT+RAG"):
            nid = future_to_id[fut]
            try:
                order[nid] = fut.result()
            except Exception as exc:
                LOGGER.exception("Hard failure on note %s: %s", nid, exc)
                order[nid] = Prediction(note_id=nid, codes=[], scores=None)
            if checkpoint_every and len(order) % checkpoint_every == 0:
                with write_lock:
                    snapshot = [order[k] for k in all_note_ids if k in order]
                    write_predictions(snapshot, out_csv, include_scores=False)

    predictions = [order[nid] for nid in all_note_ids]
    write_predictions(predictions, out_csv, include_scores=False)
    log_prediction_stats(predictions, label="M4", budget_k=shortlist_k,
                         out_path=stats_sidecar(out_csv))
    log_response_filter_tally(label="M4", level=candidate_filter)

    if dump_scores_npz is not None:
        _dump_generation_order_npz(predictions, dump_scores_npz)
    return len(predictions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M4: GPT + BM25 exemplars (RAG).")
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path,
                   help="Path to BM25 train-note index (from M1 build-index).")
    p.add_argument("--kb", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of BM25 neighbors to use as exemplars. "
                        "Lower values mean shorter prompts and lower "
                        "Azure deployment TPM pressure.")
    p.add_argument("--max-note-tokens", type=int, default=1000)
    p.add_argument("--max-exemplar-tokens", type=int, default=250)
    p.add_argument("--max-prompt-tokens", type=int, default=6000,
                   help="Hard cap on total prompt token count.  When "
                        "exceeded, M4 progressively trims exemplars → "
                        "candidates → exemplar lengths → target note "
                        "until the prompt fits.  Cap exists to bound "
                        "deployment TPM pressure (one pathological note "
                        "can otherwise blow Azure quota and trigger "
                        "500-error storms across all parallel workers).")
    p.add_argument("--diversify-jaccard-cap", type=float, default=0.8,
                   help="Drop a candidate exemplar whose code-set Jaccard "
                        "with any already-accepted exemplar exceeds this. "
                        "Set to 1.0 to disable diversification.")
    p.add_argument("--train-csv-for-text", type=Path, default=None,
                   help="Optional: train CSV providing raw exemplar text. "
                        "Defaults to using the raw_texts baked into the "
                        "BM25 index pickle (v2+).")
    p.add_argument("--sectionized-csv", type=Path, default=None,
                   help="Wide-format CSV from `cptrec-split-op-notes`.  Replaces "
                        "each TARGET note's text with M3's three evidence "
                        "sections (Procedure(s) Performed + Detailed "
                        "Description + Findings).  DEFAULT OFF, which is "
                        "M4's historical behaviour: the whole raw note, "
                        "head-truncated to --max-note-tokens.  Note that "
                        "passing `*_eval_sectioned.csv` to --notes does NOT "
                        "do this -- that file keeps the original NOTE_TEXT "
                        "column and only APPENDS the section columns.")
    p.add_argument("--train-sectionized-csv", type=Path, default=None,
                   help="Same, for the EXEMPLAR notes.  Without it each "
                        "exemplar is the first --max-exemplar-tokens (250) "
                        "of a raw note, which is mostly header.")
    p.add_argument("--backend", default="azure", choices=["azure", "local", "echo"],
                   help="'azure' = AzureOpenAIBackend (default, unchanged); "
                        "'local' = LocalOpenAIBackend against an OpenAI-"
                        "compatible endpoint (e.g. vllm serve) — the "
                        "open-weights RAG comparator; 'echo' = offline test.")
    p.add_argument("--deployment-name", default="gpt-5.3-chat",
                   help="Azure deployment name — or, with --backend local, "
                        "the served model id (whatever was passed to "
                        "`vllm serve`, e.g. google/medgemma-27b-text-it).")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint for --backend local "
                        "(default: $CPT_REC_LOCAL_LLM_BASE_URL or "
                        "http://localhost:8000/v1).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--rpm", type=int, default=250)
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Atomically write the (partial) predictions CSV "
                        "every N completed notes.  Lower = safer, more "
                        "I/O.  Used together with --resume.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Disable resume.  Default: if the output CSV "
                        "already exists, treat its rows as DONE and "
                        "skip them.")
    p.set_defaults(resume=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--history-changes", type=Path, default=None,
                   help="Optional code_changes.csv for date-aware "
                        "candidate-pool crosswalk; pair with --history-"
                        "deleted.")
    p.add_argument("--history-deleted", type=Path, default=None)
    p.add_argument("--as-of-col", default="PROCEDURE_DATE",
                   help="Date column on the test CSV used as the "
                        "crosswalk anchor.")
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)
    p.add_argument(
        "--shortlist-k", type=int, default=None,
        help=(
            "Matched-budget shortlist mode: instruct the model to return "
            "EXACTLY K codes ranked most-likely-first, so M4 can be scored "
            "at the same review budget B as a ranker's top-B shortlist "
            "(R@B / Coverage@B).  Omit for the published M4 behaviour — the "
            "prompt is then byte-identical to every previous run."
        ),
    )
    p.add_argument(
        "--self-consistency", type=int, default=1, metavar="N",
        help=(
            "Sample the SAME prompt N times and rank codes by how often the "
            "samples agree.  This replaces the synthetic generation-rank "
            "integers with a real marginal probability from the generator "
            "itself, and raises the achievable cardinality so R@5 / R@10 "
            "become comparable with a scorer's top-B.  N=1 (default) is the "
            "historical single-call path, byte-identical."
        ),
    )
    p.add_argument(
        "--sc-temperature", type=float, default=0.7,
        help="Sampling temperature when --self-consistency >1.  At 0.0 every "
             "sample is identical and the agreement score is vacuous.",
    )
    p.add_argument(
        "--candidate-filter", choices=list(CANDIDATE_FILTER_LEVELS),
        default="candidates",
        help=(
            "How hard to filter the model's RESPONSE.  'candidates' (default) "
            "keeps only codes the retrieved block offered -- M4's historical "
            "definition, and a much tighter filter than M3's.  'kb' keeps "
            "anything in the KB vocabulary, i.e. exactly M3's filter -- pair "
            "it with --sectionized-csv to also equalise the INPUT, or M4 "
            "still reads the whole raw note while M3 reads 3 sections.  "
            "'none' keeps whatever "
            "the model named, so a code retrieval never surfaced scores as a "
            "false positive instead of vanishing.  All three levels are "
            "tallied and logged whichever is active."
        ),
    )
    p.add_argument(
        "--pad-to-k", action="store_true",
        help=(
            "Guarantee EXACTLY --shortlist-k codes by filling any shortfall "
            "from the retrieved candidate pool, ranked by kNN vote.  OFF by "
            "default and never implicit: padded codes score strictly below "
            "every generated one, but the padded tail measures the RETRIEVER, "
            "not the generator, so a padded run must be reported as its own "
            "labelled row and never as the M4 generative result."
        ),
    )
    p.add_argument(
        "--dump-scores-npz",
        type=Path,
        default=None,
        help="Write the emitted codes as a ranked NPZ (generation order = "
             "rank) so cptrec-evaluate --scores-npz can report R@B for this "
             "generative row. Off by default; output is unchanged.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # Silence the noise.  httpx logs every HTTP request at INFO; the
    # openai SDK logs every internal retry on transient 5xx at INFO.
    # These look alarming during a normal run (Azure routinely returns
    # 500 / 502 / 503 a few times per thousand calls and the SDK
    # transparently retries) — bumping them to WARNING means we still
    # see *real* failures (ones that exhaust retries and reach our
    # wrapper) without the chatter.
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Self-consistency is meaningless at temperature 0: every sample would be
    # the same completion and every agreement score would be 1.0.
    sampling_temperature = (
        args.sc_temperature if args.self_consistency > 1 else args.temperature
    )
    if args.self_consistency > 1:
        LOGGER.info(
            "self-consistency: %d samples/note at temperature %.2f%s",
            args.self_consistency, sampling_temperature,
            "; padding to --shortlist-k from the retrieved pool"
            if args.pad_to_k else "",
        )
    if args.pad_to_k and args.shortlist_k is None:
        raise SystemExit("--pad-to-k requires --shortlist-k")

    if args.backend == "azure":
        rate_limiter = FixedIntervalRateLimiter(args.rpm)
        backend = AzureOpenAIBackend(
            deployment_name=args.deployment_name,
            temperature=sampling_temperature,
            max_tokens=args.max_tokens,
            rate_limiter=rate_limiter,
        )
    elif args.backend == "local":
        from cpt_rec.baselines.llm import LocalOpenAIBackend
        # Local vLLM serving usually needs no request pacing: --rpm 0 = off
        # (mirrors llm_prior's local-backend semantics).
        rate_limiter = (
            FixedIntervalRateLimiter(args.rpm) if args.rpm > 0 else None
        )
        backend = LocalOpenAIBackend(
            model=args.deployment_name,
            base_url=args.base_url,
            temperature=sampling_temperature,
            max_tokens=args.max_tokens,
            rate_limiter=rate_limiter,
        )
    else:
        backend = EchoBackend()

    index = TrainNoteBM25Index.load(args.index)
    kb = CodeKnowledgeBase.from_csv(args.kb, build_index=False)

    n = predict_b5(
        notes_csv=args.notes,
        index=index,
        kb=kb,
        out_csv=args.out,
        backend=backend,
        top_k=args.top_k,
        max_note_tokens=args.max_note_tokens,
        max_exemplar_tokens=args.max_exemplar_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        diversify_jaccard_cap=args.diversify_jaccard_cap,
        seed=args.seed,
        limit=args.limit,
        max_workers=args.max_workers,
        train_csv_for_text=args.train_csv_for_text,
        history_changes=args.history_changes,
        history_deleted=args.history_deleted,
        kb_csv=args.kb,
        as_of_col=args.as_of_col,
        note_id_col=args.note_id_col,
        note_text_col=args.note_text_col,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        shortlist_k=args.shortlist_k,
        self_consistency=args.self_consistency,
        pad_to_k=args.pad_to_k,
        dump_scores_npz=args.dump_scores_npz,
        candidate_filter=args.candidate_filter,
        sectionized_csv=args.sectionized_csv,
        train_sectionized_csv=args.train_sectionized_csv,
    )
    LOGGER.info("M4 done: %d notes -> %s", n, args.out)


if __name__ == "__main__":
    main()
