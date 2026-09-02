#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
M5 (``m5_sft_local``) — supervised fine-tuning (LoRA) of an
open-weight LLM.

Answers the reviewer question M3/M4 leave open: *"is the frontier-LLM gap
a knowledge gap or an adaptation gap?"* — by giving the open-weight model
(medgemma-27b-text-it in the reported run; any HF causal LM works)
the one thing the pipeline has and the prompted baselines lack: the
training split.

Protocol identity with M3 (deliberate, load-bearing)
----------------------------------------------------
Training examples are built with **M3's own prompt code** —
``zeroshot_system_prompt(None)`` + ``build_user_prompt`` +
``truncate_text_by_tokens(max_note_tokens)`` (and the same
``--sectionized-csv`` mode) — and the completion is M3's own answer
format ``{"selected": [...]}``.  At inference the merged model is served
with ``vllm serve`` and scored by running the *unmodified* ``cptrec-m3-zeroshot
--backend local`` against it.  So M5 vs M3-zero-shot isolates exactly one
variable: supervision.  No second prompt template, parser, or scorer to
defend.

Canonical completion order
--------------------------
Gold sets are unordered; a causal LM must emit them in *some* order.  We
fix a canonical order — descending training-split code frequency, ties
lexical — so generation order is deterministic and doubles as the ranking
for the review-budget suite (M3's ``--dump-scores-npz`` captures it).
The order map is saved next to the adapter as ``canonical_order.json``.

Subcommands
-----------
``train``   LoRA-SFT on the training split; completion-only loss.
            ``--out-dir`` is the HF Trainer ``output_dir``: it receives a
            ``checkpoint-<global_step>/`` dir per epoch (``save_strategy=
            "epoch"``) plus the end-of-run adapter in ``adapter-final/``.
``merge``   Fold a chosen adapter back into the base weights and save a
            plain HF checkpoint that ``vllm serve`` can load directly.
            Point ``--adapter`` at ``adapter-final`` for the last epoch, or
            at a ``checkpoint-<global_step>`` dir to keep an earlier one.

::

    cptrec-m5-sft train \
        --base-model ~/hf_models/google__medgemma-27b-text-it \
        --notes outputs/datasets/vumc/train.csv \
        --out-dir outputs/baselines/m5_sft/sft \
        --max-note-tokens 2048

    cptrec-m5-sft merge \
        --base-model ~/hf_models/google__medgemma-27b-text-it \
        --adapter outputs/baselines/m5_sft/sft/adapter-final \
        --out outputs/baselines/m5_sft/merged

GPU note: defaults (bf16 + LoRA + gradient checkpointing + batch 1) are
sized for a single 80 GB H100 at ``--max-seq-len 4096``; if it OOMs, pass
``--device-map auto`` to shard the base across 2 GPUs (slower, same math).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from cpt_rec.baselines.m3_zeroshot_llm import (
    assemble_sectionized_text,
    build_user_prompt,
    zeroshot_system_prompt,
)
from cpt_rec.baselines.common import (
    _ID_CANDIDATES,
    _pick_col,
    apply_seed_and_limit,
    load_notes_for_prediction,
    log_note_budget,
    truncate_text_by_tokens,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_gold_codes(
    gold_csv: Path,
    note_id_col: Optional[str] = None,
    gold_code_col: str = "proc_codes",
    gold_sep: str = "|",
) -> Dict[str, List[str]]:
    """note_id -> gold code list (order as stored; canonicalized later).

    The id column is resolved with the SAME candidate list and the SAME
    ``.str.strip()`` normalization that ``load_notes_for_prediction`` applies,
    so gold keys and note keys agree by construction. The split CSVs ship
    ``NOTE_ID`` (upper), not ``note_id`` — hardcoding either spelling here is
    how the two key spaces drift apart.
    """
    df = pd.read_csv(gold_csv, dtype=str)
    id_col = note_id_col or _pick_col(df, _ID_CANDIDATES, "note-id")
    if id_col not in df.columns or gold_code_col not in df.columns:
        raise SystemExit(
            f"--gold {gold_csv} needs columns {id_col!r} and "
            f"{gold_code_col!r}; found {list(df.columns)[:20]}"
        )
    out: Dict[str, List[str]] = {}
    ids = df[id_col].astype(str).str.strip()
    for nid, raw in zip(ids, df[gold_code_col]):
        codes = [c.strip().upper() for c in str(raw).split(gold_sep)
                 if c and str(c).strip() and str(c).strip().lower() != "nan"]
        if codes:
            out[nid] = codes
    return out


def canonical_order_map(gold: Dict[str, List[str]]) -> Dict[str, int]:
    """code -> rank (0 = most frequent in the training gold, ties lexical)."""
    freq = Counter(c for codes in gold.values() for c in codes)
    ordered = sorted(freq.keys(), key=lambda c: (-freq[c], c))
    return {c: i for i, c in enumerate(ordered)}


def completion_text(codes: Sequence[str], order: Dict[str, int]) -> str:
    ranked = sorted(set(codes), key=lambda c: (order.get(c, len(order)), c))
    return json.dumps({"selected": ranked})


def build_examples(
    notes_csv: Path,
    gold: Dict[str, List[str]],
    order: Dict[str, int],
    max_note_tokens: int,
    sectionized_csv: Optional[Path] = None,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    seed: int = 42,
    limit: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """[(note_id, user_prompt, completion)] with M3's exact prompt path."""
    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)
    text_lookup: Dict[str, str] = dict(
        zip(df["note_id"].astype(str), df["note_text"].astype(str))
    )
    if sectionized_csv is not None:
        sec_df = pd.read_csv(sectionized_csv, dtype=str)
        if "NOTE_ID" in sec_df.columns:
            sec_df = sec_df.rename(columns={"NOTE_ID": "note_id"})
        for nid, group in sec_df.groupby(sec_df["note_id"].astype(str)):
            text = assemble_sectionized_text(group.iloc[0])
            if text:
                text_lookup[str(nid)] = text

    examples: List[Tuple[str, str, str]] = []
    skipped_no_gold = 0
    for nid in df["note_id"].astype(str):
        codes = gold.get(nid)
        if not codes:
            skipped_no_gold += 1
            continue
        truncated = truncate_text_by_tokens(
            text_lookup.get(nid, ""), max_tokens=max_note_tokens
        )
        examples.append(
            (nid, build_user_prompt(truncated), completion_text(codes, order))
        )
    LOGGER.info(
        "Built %d SFT examples (%d notes had no gold codes and were skipped)",
        len(examples), skipped_no_gold,
    )
    n_notes = len(df)
    if n_notes and skipped_no_gold > n_notes // 2:
        # Overwhelmingly the signature of an id-column mismatch between the
        # notes CSV and the gold CSV, not of genuinely uncoded notes. Failing
        # here costs seconds; not failing costs a 16-30 h fine-tune on a
        # fraction of the split that no metric would flag afterwards.
        raise SystemExit(
            f"{skipped_no_gold}/{n_notes} notes matched no gold entry — the "
            f"note ids and the gold ids are probably not the same key space.\n"
            f"  first note ids: {list(df['note_id'].astype(str)[:3])}\n"
            f"  first gold ids: {list(gold)[:3]}"
        )
    return examples


# ---------------------------------------------------------------------------
# Tokenization (completion-only loss)
# ---------------------------------------------------------------------------

def _probe_system_role(tokenizer) -> bool:
    """True if the chat template accepts a system turn (medgemma does)."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "u"}],
            tokenize=False, add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def encode_examples(
    tokenizer,
    examples: List[Tuple[str, str, str]],
    max_seq_len: int,
) -> List[Dict[str, List[int]]]:
    system_prompt = zeroshot_system_prompt(None)
    use_system = _probe_system_role(tokenizer)
    if not use_system:
        LOGGER.warning(
            "Chat template rejects a system turn — folding the system "
            "prompt into the user turn (same net context)."
        )
    eos = tokenizer.eos_token_id
    encoded: List[Dict[str, List[int]]] = []
    n_trunc = 0
    for _nid, user_prompt, completion in examples:
        if use_system:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [
                {"role": "user",
                 "content": system_prompt + "\n\n" + user_prompt},
            ]
        prompt_ids: List[int] = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        comp_ids: List[int] = tokenizer(
            completion, add_special_tokens=False
        )["input_ids"] + [eos]
        budget = max_seq_len - len(comp_ids)
        if len(prompt_ids) > budget:
            # Keep the head (procedure line lives early) plus the last 96
            # prompt tokens (JSON hint + the template's generation marker,
            # which must survive for the model to learn where to answer).
            n_trunc += 1
            keep_tail = min(96, budget // 4)
            prompt_ids = prompt_ids[: budget - keep_tail] + prompt_ids[-keep_tail:]
        input_ids = prompt_ids + comp_ids
        labels = [-100] * len(prompt_ids) + list(comp_ids)
        encoded.append({"input_ids": input_ids, "labels": labels})
    lens = [len(e["input_ids"]) for e in encoded]
    LOGGER.info(
        "Encoded %d examples: len mean %.0f / p95 %d / max %d; %d hit the "
        "%d-token cap (head+tail truncation)",
        len(encoded), sum(lens) / max(len(lens), 1),
        sorted(lens)[int(0.95 * (len(lens) - 1))] if lens else 0,
        max(lens) if lens else 0, n_trunc, max_seq_len,
    )
    return encoded


class _PadCollator:
    """Right-pad input_ids/labels; labels padded with -100."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch
        width = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attn = [], [], []
        for ex in batch:
            ids = ex["input_ids"]
            lab = ex["labels"]
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# train / merge
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    gold = load_gold_codes(
        args.gold or args.notes,
        note_id_col=args.note_id_col,
        gold_code_col=args.gold_code_col,
        gold_sep=args.gold_sep,
    )
    order = canonical_order_map(gold)
    log_note_budget(
        "M5-train", args.max_note_tokens,
        "sections:3" if args.sectionized_csv is not None else "whole-note",
        covered_by=(
            f"--max-seq-len {args.max_seq_len} caps the FULL prompt in the "
            "model's own tokenizer; keep it above the note budget or the "
            "cl100k cut stops being the only cut"
        ),
    )
    examples = build_examples(
        notes_csv=args.notes,
        gold=gold,
        order=order,
        max_note_tokens=args.max_note_tokens,
        sectionized_csv=args.sectionized_csv,
        note_id_col=args.note_id_col,
        note_text_col=args.note_text_col,
        seed=args.seed,
        limit=args.limit,
    )
    if not examples:
        raise SystemExit("No training examples — check --notes/--gold.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "canonical_order.json").write_text(
        json.dumps(order, indent=0, sort_keys=False)
    )
    (out_dir / "training_config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = encode_examples(tokenizer, examples, args.max_seq_len)

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    # LoRA + gradient checkpointing: inputs must require grads or nothing
    # flows back through the frozen embedding layer.
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=None,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=encoded,
        data_collator=_PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    final_dir = out_dir / "adapter-final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    LOGGER.info("M5 train done: adapter -> %s (per-epoch checkpoints in %s)",
                final_dir, out_dir)


def cmd_merge(args: argparse.Namespace) -> None:
    # Pre-flight BEFORE the ~5-minute, ~70 GB base-model load. Without this,
    # a missing adapter surfaces as a peft/hub "repo id" error minutes later:
    # peft falls back to treating the local path as a Hub repo the moment the
    # directory has no adapter_config.json, so the traceback names
    # HFValidationError and never mentions that training did not finish.
    if not (args.adapter / "adapter_config.json").is_file():
        parent = args.adapter.parent
        # sort by step NUMBER, not name: lexicographic puts checkpoint-1226
        # before checkpoint-613, which would misidentify the earliest epoch.
        ckpts = sorted(
            (d for d in parent.glob("checkpoint-*")
             if (d / "adapter_config.json").is_file()),
            key=lambda d: int(d.name.rsplit("-", 1)[-1])
            if d.name.rsplit("-", 1)[-1].isdigit() else -1,
        ) if parent.is_dir() else []
        why = ("exists but holds no adapter_config.json"
               if args.adapter.is_dir() else "does not exist")
        hint = (
            "\n  Usable per-epoch checkpoints found alongside it:\n    "
            + "\n    ".join(str(d) for d in ckpts)
            + "\n  Merge one of those with --adapter, or re-run `train`."
            if ckpts else
            "\n  No usable checkpoint-*/ dirs alongside it either, so the "
            "`train` step never reached a save. Check its log (the training "
            "command is piped through `tee`, so a crash there leaves exit "
            "status 0 and the shell falls through to `merge`)."
        )
        raise SystemExit(
            f"--adapter {args.adapter} {why}.{hint}"
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("Loading base %s (%s)", args.base_model, args.device_map)
    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    merged = PeftModel.from_pretrained(base, str(args.adapter))
    merged = merged.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        str(args.out), safe_serialization=True, max_shard_size="5GB"
    )
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(str(args.out))
    LOGGER.info(
        "M5 merge done: %s — serve with `vllm serve %s` and score with "
        "`cptrec-m3-zeroshot --backend local --deployment-name %s`",
        args.out, args.out, args.out,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M5: LoRA-SFT an open-weight LLM on the M3 protocol."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="LoRA-SFT on the training split.")
    t.add_argument("--base-model", required=True,
                   help="HF id or local dir (medgemma-27b-text-it).")
    t.add_argument("--notes", required=True, type=Path,
                   help="Training-split CSV with note text.")
    t.add_argument("--gold", type=Path, default=None,
                   help="CSV with note_id + gold codes (default: --notes).")
    t.add_argument("--gold-code-col", default="proc_codes")
    t.add_argument("--gold-sep", default="|")
    t.add_argument("--out-dir", required=True, type=Path)
    t.add_argument("--sectionized-csv", type=Path, default=None,
                   help="Same semantics as cptrec-m3-zeroshot.")
    t.add_argument("--max-note-tokens", type=int, default=2048,
                   help="M3's note budget — keep equal to the M3 eval runs.")
    t.add_argument("--max-seq-len", type=int, default=4096,
                   help="Hard cap on prompt+completion tokens.")
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--grad-accum", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lora-dropout", type=float, default=0.05)
    t.add_argument("--device-map", default=None,
                   help="e.g. 'auto' to shard the 27B base over 2 GPUs; "
                        "default = single visible GPU.")
    t.add_argument("--note-id-col", default=None)
    t.add_argument("--note-text-col", default=None)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--limit", type=int, default=None,
                   help="Smoke-run cap on training notes.")
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("merge", help="Merge adapter into base for vllm serve.")
    m.add_argument("--base-model", required=True)
    m.add_argument("--adapter", required=True, type=Path)
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--device-map", default=None,
                   help="Default merges on CPU RAM; 'auto' uses GPUs.")
    m.set_defaults(func=cmd_merge)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
