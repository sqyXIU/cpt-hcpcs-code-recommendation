# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Header extraction for operative notes.

This script extracts section-header strings that appear in operative notes and
maps each header to one of a fixed set of standard sections using Azure OpenAI.

Input : CSV with a column containing note text
Output: JSONL where each line contains:
        {
          "row_index": <int>,
          "standard_section_headers": { "<Standard Section>": ["<header>", ...], ... }
        }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from cpt_rec.baselines.llm import (
    AzureOpenAIBackend,
    FixedIntervalRateLimiter,
    LocalOpenAIBackend,
)
from cpt_rec.common.constants import STANDARD_SECTIONS

LOGGER = logging.getLogger(__name__)


def _standard_sections_bullets(sections: Sequence[str] = STANDARD_SECTIONS) -> str:
    return "\n".join(f"- {s}" for s in sections)


def build_system_prompt(sections: Sequence[str] = STANDARD_SECTIONS) -> str:
    """
    The extraction prompt for one section taxonomy.

    ``sections`` defaults to :data:`STANDARD_SECTIONS` (VUMC operative notes),
    which is what every existing caller gets.  It is a parameter because the
    taxonomy is corpus-specific: a MIMIC-IV discharge summary has "Brief
    Hospital Course" and "Pertinent Results" and has no "Specimens Removed",
    so forcing its headers into the operative-note buckets would not be the
    same procedure applied to a second corpus -- it would be a lossy
    projection onto the first corpus's vocabulary.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(sections_bullets=_standard_sections_bullets(sections))


# Concise prompts: keep rules in system; keep note + section list in user.
SYSTEM_PROMPT_TEMPLATE: str = """
You extract section/field HEADER STRINGS from operative notes.

Return a single JSON object with exactly this top-level key:
  "standard_section_headers": an object mapping each Standard Section to a list of header strings.

Rules:
- Include ONLY headers that appear verbatim in the note; copy EXACTLY (case/punctuation/spacing).
- Return ONLY the header text, not its value/content.
  If the note has "Header: value", return "Header:" (include the colon).
- Map each extracted header to exactly ONE Standard Section.
- Within each Standard Section: deduplicate while preserving first-seen order.
- Always include ALL Standard Sections as keys (use [] when absent).
- Emit raw json only: no prose, no markdown code fences.

Standard Sections (use these exact names as keys):
{sections_bullets}
""".strip()

#: Backwards-compatible module constant: the VUMC operative-note prompt.
SYSTEM_PROMPT: str = build_system_prompt(STANDARD_SECTIONS)


USER_PROMPT_TEMPLATE: str = """
Operative note:
<<<NOTE
{note_text}
NOTE>>>
""".strip()


def build_user_prompt(note_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(note_text=note_text or "")


def empty_headers_payload(
    sections: Sequence[str] = STANDARD_SECTIONS,
) -> Dict[str, List[str]]:
    """Return the canonical empty mapping with all standard sections present."""
    return {section: [] for section in sections}


def validate_and_normalize_output(
    payload: Any,
    sections: Sequence[str] = STANDARD_SECTIONS,
) -> Dict[str, List[str]]:
    """
    Normalize model output into:
      { "<Standard Section>": ["<header1>", ...], ... }

    Enforces:
    - all standard sections exist
    - values are list[str]
    - strip whitespace
    - deduplicate within section preserving order
    """
    output = empty_headers_payload(sections)

    if not isinstance(payload, dict):
        return output

    mapping = payload.get("standard_section_headers")
    if not isinstance(mapping, dict):
        return output

    for section in sections:
        raw = mapping.get(section, [])
        if not isinstance(raw, list):
            continue

        seen: set[str] = set()
        cleaned: List[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            header = item.strip()
            if not header or header in seen:
                continue
            seen.add(header)
            cleaned.append(header)

        output[section] = cleaned

    return output


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def loads_lenient(content: str) -> Any:
    """
    ``json.loads`` that survives a response which is not bare JSON.

    Matters because ``response_format={"type": "json_object"}`` is not always
    in force: some Azure deployments reject it unless the messages contain the
    literal lowercase word "json", and the backend responds by dropping the
    constraint and retrying.  The model then usually still returns JSON, but is
    free to wrap it in a ```json fence or a sentence.  A bare ``json.loads``
    raises on that, and the caller's except-branch turns the raise into an
    *empty payload* -- so a whole run would score zero headers per note while
    only logging one line each.  Strip fences, then fall back to the outermost
    brace-balanced object.
    """
    text = (content or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    unfenced = _FENCE_RE.sub("", text).strip()
    if unfenced != text:
        try:
            return json.loads(unfenced)
        except json.JSONDecodeError:
            text = unfenced

    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object in response", text, 0)
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise json.JSONDecodeError("unbalanced JSON object in response", text, start)


def call_llm_extract_headers(
    note_text: str,
    backend: "AzureOpenAIBackend | LocalOpenAIBackend",
    system_prompt: Optional[str] = None,
    sections: Sequence[str] = STANDARD_SECTIONS,
) -> Dict[str, List[str]]:
    """
    Extract headers for one note via the shared chat backend.

    The backend handles JSON-mode kwargs (new-style vs legacy deployments),
    rate limiting, and retry/backoff.  Returns an empty payload on
    persistent failure so the caller never has to handle exceptions.
    """
    user_prompt = build_user_prompt(note_text)
    try:
        content = backend.chat(system_prompt or SYSTEM_PROMPT, user_prompt) or "{}"
        payload = loads_lenient(content)
        return validate_and_normalize_output(payload, sections)
    except Exception as exc:
        LOGGER.error("Header extraction failed: %s", exc)
        return empty_headers_payload(sections)


def assert_external_api_allowed(input_csv_path: str, allow_external_api: bool) -> None:
    """
    Refuse to send a restricted corpus to a third-party endpoint by accident.

    ``benchmark.corpora`` already declares ``external_api_allowed`` per corpus,
    but until now nothing read it -- it was a comment with no teeth, so a single
    mistyped ``--input`` could have put DUA-restricted note text on the wire.
    This resolves the input path against the registry and stops if the owning
    corpus forbids it.

    It is a guard rail, not a veto: ``--allow-external-api`` proceeds, and
    records the decision in the run log and the shell history, which is what a
    compliance story needs anyway.  PhysioNet does permit Azure OpenAI for MIMIC
    *provided* the deployment has human review disabled via the Limited Access
    form -- so passing the flag can be entirely legitimate.  It just has to be a
    decision somebody made on purpose.
    """
    try:
        from cpt_rec.benchmark.corpora import REGISTRY
    except Exception:  # registry is optional for standalone use
        return

    resolved = Path(input_csv_path).expanduser().resolve()
    for corpus in REGISTRY.values():
        if corpus.external_api_allowed:
            continue
        try:
            root = Path(corpus.root).expanduser().resolve()
        except Exception:
            continue
        if root not in resolved.parents and root != resolved:
            continue
        if allow_external_api:
            LOGGER.warning(
                "--allow-external-api: sending %s note text to a third-party "
                "endpoint even though corpus %r declares external_api_allowed="
                "False (%s)",
                corpus.key, corpus.key, corpus.external_api_note,
            )
            return
        raise SystemExit(
            f"REFUSING to send {corpus.display_name} note text to a third-party "
            f"endpoint.\n"
            f"  input  : {resolved}\n"
            f"  corpus : {corpus.key} (external_api_allowed=False)\n"
            f"  reason : {corpus.external_api_note}\n\n"
            f"Either run the DUA-safe route --\n"
            f"  --backend local --base-url http://localhost:8000/v1 --model <served-id>\n"
            f"-- or, if this endpoint is cleared for the corpus (for MIMIC that "
            f"means an Azure OpenAI deployment with human review disabled via "
            f"PhysioNet's Limited Access form), re-run with --allow-external-api."
        )


def choose_sample_indices(n_rows: int, n_sample: int, seed: int) -> List[int]:
    """Return deterministic row indices to process."""
    if n_rows <= 0:
        return []
    if n_sample <= 0 or n_sample >= n_rows:
        return list(range(n_rows))
    rng = random.Random(seed)
    return rng.sample(range(n_rows), n_sample)


def process_one_note(
    row_index: int,
    note_text: str,
    backend: "AzureOpenAIBackend | LocalOpenAIBackend",
    system_prompt: Optional[str] = None,
    sections: Sequence[str] = STANDARD_SECTIONS,
) -> Dict[str, Any]:
    extracted = call_llm_extract_headers(
        note_text=note_text, backend=backend,
        system_prompt=system_prompt, sections=sections,
    )
    return {"row_index": int(row_index), "standard_section_headers": extracted}


def _prepare_output_path(output_jsonl_path: str) -> Path:
    """
    Probe the output path *before* spending any LLM tokens.

    Creates the parent directory (``mkdir -p``), then tries opening the
    file in append mode and immediately closing it.  If anything goes
    wrong (permission denied, parent on a read-only mount, …) we surface
    the error in <1 second instead of at the end of an hour-long run.
    """
    out = Path(output_jsonl_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Touch + close to validate writability without truncating any
    # existing partial output the user may want to recover.
    with open(out, "a", encoding="utf-8"):
        pass
    return out


def _partial_path(out: Path) -> Path:
    """Sibling path used for incremental writes."""
    return out.with_suffix(out.suffix + ".partial")


def run_extraction(
    input_csv_path: str,
    output_jsonl_path: str,
    note_text_column: str,
    n_sample: int,
    seed: int,
    deployment_name: str,
    max_workers: int,
    requests_per_minute: int,
    backend_kind: str = "azure",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    sections: Optional[Sequence[str]] = None,
    allow_external_api: bool = False,
) -> None:
    if backend_kind == "azure":
        assert_external_api_allowed(input_csv_path, allow_external_api)

    sections = list(sections) if sections else list(STANDARD_SECTIONS)
    system_prompt = build_system_prompt(sections)
    LOGGER.info(
        "backend=%s sections=%d (%s)",
        backend_kind, len(sections),
        "STANDARD_SECTIONS" if sections == list(STANDARD_SECTIONS) else "custom",
    )

    # Fail fast on a missing parent directory, permission error, or
    # read-only mount BEFORE any LLM call is made.
    out_path = _prepare_output_path(output_jsonl_path)
    partial_path = _partial_path(out_path)
    LOGGER.info("Output path validated: %s", out_path)

    df = pd.read_csv(input_csv_path)
    if note_text_column not in df.columns:
        raise ValueError(
            f"Column '{note_text_column}' not found. Available: {df.columns.tolist()}"
        )

    sample_indices = choose_sample_indices(len(df), n_sample, seed)
    LOGGER.info("Processing %d/%d notes.", len(sample_indices), len(df))

    # One shared backend across workers — the underlying httpx client is
    # thread-safe and the rate limiter / retry policy live inside the
    # backend, so workers don't need their own state.
    rate_limiter = FixedIntervalRateLimiter(requests_per_minute=requests_per_minute)
    backend: "AzureOpenAIBackend | LocalOpenAIBackend"
    if backend_kind == "local":
        # Same chat(system, user) contract as the Azure backend, pointed at an
        # OpenAI-compatible server we host -- so note text never leaves the box.
        # This is the backend the MIMIC LLM rows already use.
        backend = LocalOpenAIBackend(
            model=model or deployment_name,
            base_url=base_url,
            temperature=0.0,
            max_tokens=800,
            max_retries=5,
            rate_limiter=rate_limiter,
        )
    else:
        backend = AzureOpenAIBackend(
            deployment_name=deployment_name,
            temperature=0.0,
            max_tokens=800,
            max_retries=5,
            rate_limiter=rate_limiter,
        )

    # Copy note text out of pandas to avoid concurrent dataframe access.
    notes: Dict[int, str] = {}
    for idx in sample_indices:
        val = df.at[idx, note_text_column]
        notes[idx] = "" if pd.isna(val) else str(val)

    results: Dict[int, Dict[str, Any]] = {}

    # Incremental write: every completed result is appended to a
    # ``.partial`` JSONL the moment it arrives.  If the process crashes,
    # ``.partial`` survives — re-run-time recovery is just renaming the
    # file or feeding it to whatever consumes the final JSONL.
    partial_lock = threading.Lock()
    with open(partial_path, "w", encoding="utf-8") as partial_f, \
            ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(
                process_one_note,
                idx,
                notes[idx],
                backend,
                system_prompt,
                sections,
            ): idx
            for idx in sample_indices
        }

        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Extracting"):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                LOGGER.exception("Hard failure on row %d: %s", idx, exc)
                results[idx] = {
                    "row_index": int(idx),
                    "standard_section_headers": empty_headers_payload(sections),
                    "error": str(exc),
                }
            # Append to partial as soon as the result is in hand.
            with partial_lock:
                partial_f.write(
                    json.dumps(results[idx], ensure_ascii=False) + "\n"
                )
                partial_f.flush()

    # Final, ordered write.  Atomic via tempfile + os.replace so we
    # don't end up with a half-written file if the disk fills mid-write.
    fd, tmp_name = tempfile.mkstemp(
        prefix=out_path.name + ".",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for idx in sample_indices:
                f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    # Final ordered file landed cleanly — drop the partial.
    try:
        partial_path.unlink()
    except FileNotFoundError:
        pass

    LOGGER.info("Wrote JSONL to %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract operative-note header strings mapped to standard sections (JSONL output)."
    )
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--note-text-column", default="NOTE_TEXT", help="Column containing note text")
    p.add_argument("--n-sample", type=int, default=0, help="Number of notes to sample (0 = all)")
    p.add_argument("--seed", type=int, default=13, help="Random seed for sampling")
    p.add_argument("--deployment-name", default="gpt-4.1", help="Azure model deployment name")
    p.add_argument(
        "--backend",
        default="azure",
        choices=["azure", "local"],
        help="Chat backend. 'local' targets an OpenAI-compatible server "
             "(vLLM/TGI/SGLang) via --base-url, so note text stays on the box. "
             "Default 'azure' preserves the existing behaviour exactly.",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="--backend local only: endpoint (default http://localhost:8000/v1 "
             "or $CPT_REC_LOCAL_LLM_BASE_URL).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="--backend local only: served model id, i.e. whatever was passed "
             "to `vllm serve` (defaults to --deployment-name).",
    )
    p.add_argument(
        "--sections-file",
        default=None,
        help="JSON list of section names to extract into. Default: the 19 "
             "STANDARD_SECTIONS (VUMC operative notes). Point this at a "
             "corpus-specific taxonomy when the note genre differs.",
    )
    p.add_argument(
        "--allow-external-api",
        action="store_true",
        help="Proceed even when the input's corpus declares "
             "external_api_allowed=False. Records the decision in the log.",
    )
    p.add_argument("--max-workers", type=int, default=16, help="ThreadPoolExecutor max workers")
    p.add_argument("--rpm", type=int, default=250, help="Requests per minute ceiling (global)")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    sections = None
    if args.sections_file:
        sections = json.loads(Path(args.sections_file).read_text())
        if not isinstance(sections, list) or not all(isinstance(x, str) for x in sections):
            raise SystemExit(
                f"--sections-file must be a JSON list of strings: {args.sections_file}"
            )

    run_extraction(
        input_csv_path=args.input,
        output_jsonl_path=args.output,
        note_text_column=args.note_text_column,
        n_sample=args.n_sample,
        seed=args.seed,
        deployment_name=args.deployment_name,
        max_workers=args.max_workers,
        requests_per_minute=args.rpm,
        backend_kind=args.backend,
        base_url=args.base_url,
        model=args.model,
        sections=sections,
        allow_external_api=args.allow_external_api,
    )


if __name__ == "__main__":
    main()