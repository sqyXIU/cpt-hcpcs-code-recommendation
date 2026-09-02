#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
M3 (``m3_zeroshot_frontier``) — frontier LLM, zero-shot.

Prompt the LLM with the full operative note (truncated to a token budget)
and ask it to return the set of CPT/HCPCS base codes for the procedures
performed.  Parse a JSON response and validate every code against the KB
so that out-of-vocab hallucinations are dropped.

This is the *fastest path to first numbers*: no retrieval, no candidate
list, no fine-tuning.  It is the natural ceiling for what a frontier LLM
can do unaided on this dataset.

Concurrency
-----------
Calls fan out across a thread pool gated by a global
``FixedIntervalRateLimiter``.  Each worker holds its own
``AzureOpenAIBackend`` instance (the SDK client is thread-safe but we
prefer per-worker clients for clarity).  Predictions are checkpointed
atomically every ``--checkpoint-every`` results.

Sectionization
--------------
``--sectionized-csv`` points at the wide-format CSV emitted by
``cptrec-split-op-notes`` and switches the prompt to use only the
evidence-rich sections (Procedure(s) Performed + Detailed Description +
Findings) instead of the raw note.  This avoids losing the procedure
description to truncation when a long indications block sits at the top.

Backends
--------

* ``azure``  — Azure OpenAI (default).  Requires
  ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
  ``AZURE_OPENAI_API_VERSION``.
* ``echo``   — deterministic stub for offline tests; always returns
  ``{"selected": []}`` for M3 because there are no candidates in the
  prompt.

CLI
---

::

    cptrec-m3-zeroshot \\
        --notes outputs/datasets/vumc/test_eval.csv \\
        --kb data/kb/codes_with_ranges.csv \\
        --out outputs/baselines/m3_zeroshot_frontier/predictions/test.csv \\
        --backend azure \\
        --deployment-name gpt-5.3-chat \\
        --max-note-tokens 2048 \\
        --max-workers 16 --rpm 250
"""

from __future__ import annotations

import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from cpt_rec.baselines.common import (
    Prediction,
    apply_seed_and_limit,
    complete_shortlist,
    dedupe_keep_order,
    load_notes_for_prediction,
    log_note_budget,
    log_prediction_stats,
    stats_sidecar,
    maybe_load_code_history,
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
from cpt_rec.common.constants import STANDARD_SECTIONS
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert medical coder.  Read the operative note below and "
    "return the full set of CPT and HCPCS Level II procedure base codes "
    "for the procedures actually performed.\n\n"
    "Rules:\n"
    "- Return 5-digit CPT Cat I codes (e.g. 43235), 4-digit codes ending in "
    "A / F / M / T / U (vaccine admin, Cat II, MAAA, Cat III, PLA — e.g. "
    "0001A, 2029F, 0002M, 0479T, 0001U), or HCPCS Level II codes "
    "(one letter + 4 digits, e.g. J1885).\n"
    "- The note may be from any year between 2018 and the present; "
    "historical / retired codes that were valid on the procedure date "
    "(e.g. 49560 for hernia repair before 2023) are acceptable answers — "
    "do not crosswalk them to current substitutes yourself.\n"
    "- Do NOT emit modifiers (no -59, -RT, -LT, -26, etc.).\n"
    "- Emit each distinct procedure only once.\n"
    "- Omit codes you are unsure about.\n"
    'Return a JSON object: {"selected": ["<code1>", ...]}'
)


# Sections used by the --sectionized-csv mode.  Procedure(s) Performed +
# Detailed Description + Findings is where the codeable evidence lives.
_EVIDENCE_SECTIONS: List[str] = [
    s for s in STANDARD_SECTIONS
    if s in {"Procedure(s) Performed", "Detailed Description", "Findings"}
]


def assemble_sectionized_text(
    row: pd.Series,
    sections: List[str] = _EVIDENCE_SECTIONS,
) -> str:
    """Concatenate evidence-rich sections from a wide-format row."""
    parts: List[str] = []
    for sec in sections:
        if sec in row and isinstance(row[sec], str) and row[sec].strip():
            parts.append(f"## {sec}\n{row[sec].strip()}")
    return "\n\n".join(parts)


def zeroshot_system_prompt(shortlist_k: Optional[int] = None) -> str:
    """System prompt, optionally in *matched-budget shortlist* mode.

    ``shortlist_k=None`` returns ``SYSTEM_PROMPT`` unchanged, so the default
    path is byte-identical to every previously published M3 run.  With a
    budget, the set-emission instruction and the precision-favouring
    "omit if unsure" rule are replaced by a fixed-cardinality ranked
    instruction (mirrors ``m4_exemplar_rag._system_prompt``), so the zero-shot
    LLM can be scored at the same review budget ``B`` as a ranker's top-B
    shortlist.
    """
    if shortlist_k is None:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        .replace(
            "return the full set of CPT and HCPCS Level II procedure base "
            "codes for the procedures actually performed.",
            f"return the {shortlist_k} CPT / HCPCS Level II procedure base "
            "codes MOST LIKELY to be billed for the procedures performed, "
            "ranked most likely first.  A human coder will review your "
            "shortlist, so include plausible codes rather than only "
            "certain ones.",
        )
        .replace(
            "- Omit codes you are unsure about.\n",
            f"- Return EXACTLY {shortlist_k} codes, ordered most likely "
            "first.  Do not omit uncertain codes — rank them lower.\n",
        )
    )


def build_user_prompt(note_text: str, shortlist_k: Optional[int] = None) -> str:
    if shortlist_k is None:
        hint = 'Return JSON: {"selected": ["<code>", ...]}'
    else:
        hint = (
            f'Return JSON with exactly {shortlist_k} codes, best first: '
            '{"selected": ["<code>", ...]}'
        )
    return (
        "OPERATIVE NOTE:\n"
        f"{note_text}\n\n"
        + hint
    )


#: Running tally of the KB-membership filter, so its effect is a measured
#: quantity rather than an assumption.  ``emitted`` counts every distinct code
#: the model named (post-dedupe, pre-filter); ``dropped`` counts those the KB
#: rejected.  Incremented from worker threads -- CPython's GIL makes ``+=`` on
#: a dict entry safe enough for a diagnostic counter, and an exact count is
#: not load-bearing for any result.
_KB_FILTER_TALLY: Dict[str, int] = {"emitted": 0, "dropped": 0, "notes_hit": 0}


def reset_kb_filter_tally() -> None:
    for k in _KB_FILTER_TALLY:
        _KB_FILTER_TALLY[k] = 0


def _apply_kb_filter(
    codes: Sequence[str], kb_codes: set, enabled: bool
) -> List[str]:
    """Drop out-of-vocabulary codes, tallying what was removed.

    With ``enabled=False`` nothing is dropped and M3 becomes a genuinely
    unaided zero-shot baseline: a hallucinated code counts as a false
    positive rather than vanishing.  The tally is kept either way, so the
    cost of the filter is reported even when it is switched off.
    """
    out = [c for c in codes if c in kb_codes]
    n_drop = len(codes) - len(out)
    _KB_FILTER_TALLY["emitted"] += len(codes)
    _KB_FILTER_TALLY["dropped"] += n_drop
    if n_drop:
        _KB_FILTER_TALLY["notes_hit"] += 1
    return out if enabled else list(codes)


def log_kb_filter_tally(label: str = "M3", enabled: bool = True) -> Dict[str, float]:
    """Report what KB validation removed (or would have removed)."""
    emitted = _KB_FILTER_TALLY["emitted"]
    dropped = _KB_FILTER_TALLY["dropped"]
    rate = (dropped / emitted) if emitted else 0.0
    LOGGER.info(
        "[%s] KB filter %s: model named %d codes, %d not in KB (%.1f%%), "
        "affecting %d note-samples",
        label, "ON (dropped)" if enabled else "OFF (kept, counted only)",
        emitted, dropped, 100.0 * rate, _KB_FILTER_TALLY["notes_hit"],
    )
    return {
        "kb_filter_enabled": float(bool(enabled)),
        "codes_named_by_model": float(emitted),
        "codes_not_in_kb": float(dropped),
        "pct_codes_not_in_kb": round(100.0 * rate, 2),
        "note_samples_with_a_drop": float(_KB_FILTER_TALLY["notes_hit"]),
    }


def _score_one_note(
    note_id: str,
    note_text: str,
    backend: LLMBackend,
    kb_codes: set,
    max_note_tokens: int,
    shortlist_k: Optional[int] = None,
    self_consistency: int = 1,
    kb_filter: bool = True,
) -> Prediction:
    """Score one note; with ``self_consistency > 1``, sample and marginalise.

    ``self_consistency=1`` is the historical path exactly: one call, codes in
    generation order, ``scores=None``.  Above 1 the same prompt is sampled
    ``n`` times and codes are ranked by how often the samples agree, which
    gives the generator a calibrated score of its own instead of the
    synthetic ``len(codes) - i`` rank integers.

    ``kb_filter=False`` reports the model's own vocabulary: out-of-KB codes
    are counted but kept, so hallucinations show up as false positives.
    """
    truncated = truncate_text_by_tokens(note_text, max_tokens=max_note_tokens)
    user_prompt = build_user_prompt(truncated, shortlist_k)
    system_prompt = zeroshot_system_prompt(shortlist_k)
    n = max(1, int(self_consistency))
    try:
        if n > 1:
            responses = backend.chat_n(system_prompt, user_prompt, n)
        else:
            responses = [backend.chat(system_prompt, user_prompt)]
    except Exception as exc:
        LOGGER.error("LLM call failed for note %s: %s", note_id, exc)
        responses = [""]

    samples = [
        _apply_kb_filter(
            dedupe_keep_order(parse_selected_codes(r)), kb_codes, kb_filter
        )
        for r in responses
    ]
    if n > 1:
        kept, scores = rank_by_self_consistency(samples)
        if shortlist_k is not None:
            kept, scores = complete_shortlist(kept, scores, shortlist_k)
        return Prediction(note_id=note_id, codes=kept, scores=scores or None)

    kept = samples[0]
    if shortlist_k is not None:
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


def predict_b4(
    notes_csv: Path,
    kb: CodeKnowledgeBase,
    out_csv: Path,
    backend: LLMBackend,
    max_note_tokens: int = 2048,
    seed: int = 42,
    limit: Optional[int] = None,
    max_workers: int = 16,
    sectionized_csv: Optional[Path] = None,
    history_changes: Optional[Path] = None,
    history_deleted: Optional[Path] = None,
    kb_csv: Optional[Path] = None,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    checkpoint_every: int = 50,
    shortlist_k: Optional[int] = None,
    self_consistency: int = 1,
    dump_scores_npz: Optional[Path] = None,
    resume: bool = True,
    kb_filter: bool = True,
) -> int:
    """Drive M3 end-to-end with a thread pool."""
    reset_kb_filter_tally()
    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)
    LOGGER.info("Loaded %d notes from %s", len(df), notes_csv)

    # Validation vocabulary: when history is loaded, accept any code that
    # is *either* in the active 2026 KB *or* ever existed in CodeHistory.
    # Without this expansion the LLM's correct historical answers (e.g.
    # ``49560`` on a pre-2023 hernia case) get silently dropped.
    history, _ = maybe_load_code_history(
        history_changes=history_changes,
        history_deleted=history_deleted,
        kb_csv=kb_csv,
    )
    valid_codes = set(kb.codes)
    if history is not None:
        before = len(valid_codes)
        valid_codes = valid_codes | set(history.all_codes())
        LOGGER.info(
            "M3: validation vocab = active KB ∪ CodeHistory.all_codes() "
            "(%d → %d codes; +%d historical/deleted)",
            before, len(valid_codes), len(valid_codes) - before,
        )

    note_text_lookup: Dict[str, str] = dict(
        zip(df["note_id"].astype(str), df["note_text"].astype(str))
    )

    log_note_budget(
        "M3", max_note_tokens,
        f"sections:{len(_EVIDENCE_SECTIONS)}" if sectionized_csv is not None
        else "whole-note",
    )
    if sectionized_csv is not None:
        LOGGER.info("Using sectionized text from %s", sectionized_csv)
        sec_df = pd.read_csv(sectionized_csv, dtype=str)
        if "NOTE_ID" in sec_df.columns:
            sec_df = sec_df.rename(columns={"NOTE_ID": "note_id"})
        for nid, group in sec_df.groupby(sec_df["note_id"].astype(str)):
            row = group.iloc[0]
            text = assemble_sectionized_text(row)
            if text:
                note_text_lookup[str(nid)] = text
            elif str(nid) not in note_text_lookup:
                note_text_lookup[str(nid)] = ""

    kb_codes = valid_codes  # expanded above when history is provided
    predictions: List[Prediction] = []
    note_ids = df["note_id"].astype(str).tolist()
    write_lock = threading.Lock()

    # Resume: if the output CSV already has rows, treat them as DONE so
    # a crashed long-running job can be re-launched without re-issuing
    # successful calls.  Permanent-failure rows (empty pred_codes from a
    # previous attempt) are also treated as done — pass --no-resume to
    # redo them after Azure stabilizes.  Mirrors ``predict_b5``.
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
                "M3 resume: loaded %d previously-completed notes from %s "
                "(use --no-resume to redo them)",
                len(order), out_csv,
            )
        except Exception as exc:
            LOGGER.warning(
                "M3 resume: could not parse %s (%s); starting fresh",
                out_csv, exc,
            )

    todo_ids = [nid for nid in note_ids if nid not in order]
    if not todo_ids:
        LOGGER.info("M3: nothing to do — all %d notes already done.",
                    len(note_ids))
        predictions = [order[nid] for nid in note_ids]
        log_prediction_stats(predictions, label="M3", budget_k=shortlist_k,
                         out_path=stats_sidecar(out_csv))
        # No KB-filter tally here: this branch scored nothing this run, so the
        # counters are zero and printing them would read as "no hallucinations".
        if dump_scores_npz is not None:
            _dump_generation_order_npz(predictions, dump_scores_npz)
        return len(order)

    LOGGER.info("M3: %d/%d notes to score (%d already done from previous run)",
                len(todo_ids), len(note_ids), len(order))

    def _job(note_id: str) -> Prediction:
        return _score_one_note(
            note_id=note_id,
            note_text=note_text_lookup.get(note_id, ""),
            backend=backend,
            kb_codes=kb_codes,
            max_note_tokens=max_note_tokens,
            shortlist_k=shortlist_k,
            kb_filter=kb_filter,
            self_consistency=self_consistency,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {pool.submit(_job, nid): nid for nid in todo_ids}
        for fut in tqdm(as_completed(future_to_id),
                        total=len(future_to_id), desc="M3 zero-shot"):
            nid = future_to_id[fut]
            try:
                order[nid] = fut.result()
            except Exception as exc:
                LOGGER.exception("Hard failure on note %s: %s", nid, exc)
                order[nid] = Prediction(note_id=nid, codes=[], scores=None)
            if checkpoint_every and len(order) % checkpoint_every == 0:
                with write_lock:
                    snapshot = [order[k] for k in note_ids if k in order]
                    write_predictions(snapshot, out_csv, include_scores=False)

        # Re-emit in stable note_id order.
        predictions = [order[nid] for nid in note_ids]

    write_predictions(predictions, out_csv, include_scores=False)
    log_prediction_stats(predictions, label="M3", budget_k=shortlist_k,
                         out_path=stats_sidecar(out_csv))
    log_kb_filter_tally(label="M3", enabled=kb_filter)

    if dump_scores_npz is not None:
        _dump_generation_order_npz(predictions, dump_scores_npz)
    return len(predictions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3: GPT zero-shot baseline.")
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--kb", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--backend", default="azure",
                   choices=["azure", "local", "echo"],
                   help="'azure' = AzureOpenAIBackend; 'local' = "
                        "LocalOpenAIBackend against an OpenAI-compatible "
                        "endpoint (e.g. vllm serve) — the SFT-baseline "
                        "path serves the merged fine-tuned model this way; "
                        "'echo' = offline stub.")
    p.add_argument("--deployment-name", default="gpt-5.3-chat",
                   help="Azure OpenAI deployment name, or (backend=local) "
                        "the served model id passed to vllm serve.")
    p.add_argument("--base-url", default=None,
                   help="backend=local only: OpenAI-compatible base URL "
                        "(default http://localhost:8000/v1 or "
                        "$CPT_REC_LOCAL_LLM_BASE_URL).")
    p.add_argument("--shortlist-k", type=int, default=None,
                   help="Matched-budget shortlist mode: ask for EXACTLY K "
                        "codes ranked most-likely-first instead of a "
                        "precision-favouring set.  Default off = prompt "
                        "byte-identical to all previous M3 runs.  Run "
                        "separately per budget B; score with --scores-npz.")
    p.add_argument("--self-consistency", type=int, default=1, metavar="N",
                   help="Sample the SAME prompt N times and rank codes by "
                        "how often the samples agree, giving the generator a "
                        "real marginal probability instead of synthetic "
                        "generation-rank integers.  N=1 (default) is the "
                        "historical single-call path, byte-identical.  N>1 "
                        "also raises the achievable cardinality, which is "
                        "what makes R@5 / R@10 comparable with a scorer.  "
                        "Use --sc-temperature to set the sampling "
                        "temperature (0.0 would make all N samples "
                        "identical and defeat the method).")
    p.add_argument("--sc-temperature", type=float, default=0.7,
                   help="Sampling temperature used when --self-consistency "
                        ">1.  Ignored at N=1, where --temperature applies.")
    p.add_argument("--dump-scores-npz", type=Path, default=None,
                   help="Also write the generation-ordered ranking as a "
                        "ragged scores NPZ for cptrec-evaluate --scores-npz "
                        "(scores = descending rank integers).")
    p.add_argument("--no-kb-filter", dest="kb_filter", action="store_false",
                   help="Do NOT drop codes that are absent from the KB. The "
                        "default (filter ON) gives the model a free "
                        "out-of-vocabulary safety net that no unaided "
                        "deployment has; with this flag a hallucinated code "
                        "counts as a false positive. The drop count is "
                        "reported either way.")
    p.set_defaults(kb_filter=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-note-tokens", type=int, default=2048)
    p.add_argument("--max-workers", type=int, default=16,
                   help="ThreadPoolExecutor max workers (Azure backend).")
    p.add_argument("--rpm", type=int, default=250,
                   help="Global requests-per-minute ceiling (Azure backend).")
    p.add_argument("--sectionized-csv", type=Path, default=None,
                   help="Optional wide-format CSV from cptrec-split-op-notes; "
                        "when provided, the prompt uses Procedure(s) "
                        "Performed + Detailed Description + Findings "
                        "instead of the raw note.")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Disable resume.  Default: if the output CSV "
                        "already exists, its notes are treated as done "
                        "and only the remainder is re-issued.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--history-changes", type=Path, default=None,
                   help="Optional code_changes.csv.  When supplied with "
                        "--history-deleted, the validation vocabulary is "
                        "expanded from the active 2026 KB to "
                        "(active KB ∪ CodeHistory.all_codes()) so the "
                        "LLM's correct historical / deleted code "
                        "predictions (e.g. retired hernia codes on "
                        "pre-2023 notes) are no longer silently dropped.")
    p.add_argument("--history-deleted", type=Path, default=None)
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # Silence per-request and per-retry chatter from httpx + openai SDK.
    # See m4_exemplar_rag.main for the full rationale; in short, a normal run
    # produces several "HTTP/1.1 500 / Retrying" log lines that look
    # alarming but represent transient Azure failures the SDK
    # transparently recovers from.
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Self-consistency is meaningless at temperature 0: every sample would be
    # the same completion and the agreement score would be 1.0 everywhere.
    sampling_temperature = (
        args.sc_temperature if args.self_consistency > 1 else args.temperature
    )
    if args.self_consistency > 1:
        LOGGER.info(
            "self-consistency: %d samples/note at temperature %.2f",
            args.self_consistency, sampling_temperature,
        )
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
        # (mirrors M4's local-backend semantics).
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
    kb = CodeKnowledgeBase.from_csv(args.kb, build_index=False)
    n = predict_b4(
        notes_csv=args.notes,
        kb=kb,
        out_csv=args.out,
        backend=backend,
        max_note_tokens=args.max_note_tokens,
        seed=args.seed,
        limit=args.limit,
        max_workers=args.max_workers,
        sectionized_csv=args.sectionized_csv,
        history_changes=args.history_changes,
        history_deleted=args.history_deleted,
        kb_csv=args.kb,
        note_id_col=args.note_id_col,
        note_text_col=args.note_text_col,
        checkpoint_every=args.checkpoint_every,
        shortlist_k=args.shortlist_k,
        kb_filter=args.kb_filter,
        self_consistency=args.self_consistency,
        dump_scores_npz=args.dump_scores_npz,
        resume=args.resume,
    )
    LOGGER.info("M3 done: %d notes -> %s", n, args.out)


if __name__ == "__main__":
    main()
