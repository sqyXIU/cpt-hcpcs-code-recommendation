#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Train the candidate verifier.

The verifier is a **pair-level binary classifier**: one example per ``(note, candidate-code)``
pair, labelled ``code ∈ gold``.  Each example is the candidate's best evidence
snippet + its code description (cross-encoder) and/or its source-tagged φ(n, c)
features.

Pipeline
--------

1. Assemble per-note candidate pools with φ over the union sources
   (KB-BM25 ∪ KB-dense ∪ neighbor [∪ LLM-concept]) — cached to disk so the
   GPU-bound snippet encode is paid once.
2. Flatten pools into pair-level examples (optional negative subsampling).
3. Train the verifier with ``BCEWithLogitsLoss``.
4. Per epoch, score the val pools and report **micro-F1 against the full gold**,
   so the number is comparable across pools.  The best checkpoint and its
   argmax decision threshold are saved.

Single-GPU by design: the cross-encoder is a ~base-size encoder and the
examples are independent, so one H100 trains a few epochs in well under an
hour once pools are cached.  Pin the GPU with ``CUDA_VISIBLE_DEVICES``.

CLI
---

::

    cptrec-verifier-train \\
        --train-csv outputs/datasets/vumc/train_eval_sectioned.csv \\
        --val-csv   outputs/datasets/vumc/val_eval_sectioned.csv \\
        --kb data/kb/codes_with_ranges.csv \\
        --kb-index-dir outputs/indices/code_kb_faiss/default/ \\
        --bm25-index outputs/indices/note_bm25/bm25.pkl \\
        --model-out outputs/verifier/sections192_baseline/ \\
        --arch cross_encoder --feature-version v1a \\
        --encoder cambridgeltl/SapBERT-from-PubMedBERT-fulltext \\
        --epochs 3 --batch-size 64 --lr 2e-5
"""

from __future__ import annotations

import os

# Silence the HF fast-tokenizer fork warning: the DataLoader (num_workers>0)
# forks after the tokenizer has been used in the main process. We don't rely on
# Rust-level tokenizer parallelism here (the DataLoader provides the parallelism),
# so disable it before transformers/tokenizers is imported. setdefault lets the
# caller still override via the environment.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.pipeline.pool.candidate_gen import KBCandidateIndex
from cpt_rec.pipeline.pool.candidate_pool import (
    AssemblerConfig,
    FEATURE_ORDERS,
    feature_dim,
)
from cpt_rec.pipeline.crossencoder.verifier import build_verifier
from cpt_rec.pipeline.crossencoder.verifier_data import (
    PairCollator,
    PoolExampleDataset,
    build_note_pools,
    score_pool,
)

LOGGER = logging.getLogger(__name__)

_CPT_REC_ALLOW_CPU_TRAIN = os.environ.get("CPT_REC_ALLOW_CPU_TRAIN") == "1"

DEFAULT_ENCODER = "SapBERT-from-PubMedBERT-fulltext"


# ---------------------------------------------------------------------------
# Validation: micro-F1 over the val pools at the candidate level
# ---------------------------------------------------------------------------

def _micro_f1(
    probs_by_note: Dict[str, "tuple"],
    gold: Dict[str, set],
    thr: float,
) -> Dict[str, float]:
    tp = fp = fn = 0
    for nid, (codes, probs) in probs_by_note.items():
        g = gold.get(nid, set())
        pred = {c for c, p in zip(codes, probs) if p >= thr}
        tp += len(pred & g)
        fp += len(pred - g)
        fn += len(g - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_pools(
    model,
    tokenizer,
    pools,
    gold: Dict[str, set],
    order: Sequence[str],
    device,
    arch: str,
    max_pair_len: int,
    thresholds: Sequence[float],
    score_batch_size: int = 128,
    pair_truncation: str = "longest_first",
) -> Dict[str, object]:
    """Score val pools once, then sweep thresholds for the best micro-F1."""
    probs_by_note: Dict[str, tuple] = {}
    for pool in pools:
        probs = score_pool(
            model, pool, order, device, arch,
            tokenizer=tokenizer, max_length=max_pair_len, batch_size=score_batch_size,
            truncation=pair_truncation,
        )
        probs_by_note[pool.note_id] = ([r.code for r in pool.records], probs)

    best = {"f1": -1.0, "threshold": 0.5}
    at_half = _micro_f1(probs_by_note, gold, 0.5)
    for thr in thresholds:
        m = _micro_f1(probs_by_note, gold, float(thr))
        if m["f1"] > best["f1"]:
            best = {"threshold": float(thr), **m}
    return {
        "best_threshold": float(best["threshold"]),
        "best_micro_f1": float(best["f1"]),
        "best_micro_precision": float(best["precision"]),
        "best_micro_recall": float(best["recall"]),
        "f1_at_0.5": float(at_half["f1"]),
        "n_notes": len(probs_by_note),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    train_csv: Path,
    kb_csv: Path,
    kb_index_dir: Path,
    model_out: Path,
    val_csv: Optional[Path] = None,
    bm25_index: Optional[Path] = None,
    train_llm_sources: Optional[Path] = None,
    val_llm_sources: Optional[Path] = None,
    keep_llm_sources: Sequence[str] = ("llm_concept",),
    arch: str = "cross_encoder",
    feature_version: str = "v1a",
    encoder: str = DEFAULT_ENCODER,
    pooling: str = "cls",
    dropout: float = 0.1,
    feature_mlp_hidden: int = 128,
    max_pair_len: int = 192,
    bi_max_length: Optional[int] = None,
    bm25_top_k: int = 25,
    dense_top_k: int = 25,
    neighbor_top_k: int = 25,
    llm_norm_k: int = 50,
    max_snippets: int = 32,
    snippet_max_words: int = 180,
    snippet_overlap_words: int = 60,
    snippet_select: str = "length",
    snippet_pool_cap: int = 0,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    neg_per_pos: Optional[int] = None,
    max_neg_per_note: Optional[int] = None,
    use_pos_weight: bool = False,
    grad_clip: float = 1.0,
    num_workers: int = 8,
    prefetch_factor: Optional[int] = None,
    section_cols: Optional[List[str]] = None,
    gold_code_col: str = "proc_codes",
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    max_train_notes: Optional[int] = None,
    max_val_notes: Optional[int] = None,
    train_pool_cache: Optional[Path] = None,
    val_pool_cache: Optional[Path] = None,
    encode_batch_size: int = 64,
    score_batch_size: int = 128,
    seed: int = 42,
    device: Optional[str] = None,
    log_every: int = 50,
    local_models_dir: Optional[Path] = None,
    pair_truncation: str = "longest_first",
    single_gpu: bool = False,
    enable_tf32: bool = False,
) -> Dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # A silent CPU fallback here costs DAYS, not minutes, and nothing downstream
    # ever says so: a cross-encoder at max_pair_len=320 runs ~0.4 s/sample on CPU
    # against ~0.004 s/sample on an H100, so a 5-epoch MIMIC run goes from ~1.5 h
    # to ~4 days and looks, in the log, exactly like a slow GPU. Measured
    # 2026-08-26: the MIMIC verifier train sat at 27.5 s/step (batch 64) for
    # 250 steps before anyone asked why.
    #
    # So: always say which device was chosen, and refuse the fallback when the
    # caller plainly asked for a GPU. `CUDA_VISIBLE_DEVICES=""` -- what an empty
    # $G2/$G3 expands to -- is the exact way this happens, and it is invisible.
    if str(device) == "cpu":
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None and not _CPT_REC_ALLOW_CPU_TRAIN:
            raise SystemExit(
                "REFUSING TO TRAIN ON CPU.\n"
                f"  CUDA_VISIBLE_DEVICES={cvd!r} is set, so a GPU was intended, "
                "but torch.cuda.is_available() is False.\n"
                "  An empty value (from an unset $G1/$G2/$G3) hides every GPU. "
                "Re-source scripts/amc_env.sh and read the slot line.\n"
                "  This run would take ~100x longer on CPU and would not say so.\n"
                "  To train on CPU deliberately, set CPT_REC_ALLOW_CPU_TRAIN=1."
            )
        LOGGER.warning("TRAINING ON CPU -- expect ~100x the GPU wall-clock.")
    else:
        LOGGER.info("device=%s  visible GPUs=%d  CUDA_VISIBLE_DEVICES=%s",
                    device, torch.cuda.device_count(),
                    os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    # TF32 tensor-core matmuls (opt-in): ~1.3-1.6x on H100 for the fp32
    # transformer matmuls, at a sub-1e-3 numeric cost. Default OFF so the
    # champion / A-B comparisons stay byte-identical unless explicitly requested.
    if enable_tf32 and str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        LOGGER.info("TF32 matmul/cudnn enabled")
    model_out = Path(model_out)
    model_out.mkdir(parents=True, exist_ok=True)
    order = FEATURE_ORDERS[feature_version]
    n_features = feature_dim(feature_version)

    if neg_per_pos is None:
        LOGGER.warning(
            "neg_per_pos is None -> keeping the FULL candidate pool as training "
            "pairs (~hundreds of negatives/note). This matches the inference "
            "distribution exactly but is dramatically slower (e.g. ~12 h/epoch on "
            "~19k notes with the LLM-concept pool). Pass --neg-per-pos (the "
            "v1a_ce baseline used 10) to subsample for a faster, balanced run."
        )

    asm_cfg = AssemblerConfig(
        bm25_top_k=bm25_top_k, dense_top_k=dense_top_k,
        neighbor_top_k=neighbor_top_k, llm_norm_k=llm_norm_k,
        max_snippets=max_snippets, snippet_max_words=snippet_max_words,
        snippet_overlap_words=snippet_overlap_words, encode_batch_size=encode_batch_size,
        snippet_select=snippet_select, snippet_pool_cap=snippet_pool_cap,
    )

    LOGGER.info("Loading KB + KB index for pool assembly ...")
    kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
    # Snippet (query-side) bi-encoder budget for dense retrieval + best-snippet
    # selection. Historically this silently followed max_pair_len, which made
    # every --max-pair-len A/B also a *retrieval* A/B (different pools / φ at
    # 192 vs 320). --bi-max-length decouples them; None keeps the historical
    # coupling so existing recipes stay byte-identical.
    resolved_bi_len = int(bi_max_length) if bi_max_length is not None else int(max_pair_len)
    kb_index = KBCandidateIndex(kb_index_dir, bi_max_length=resolved_bi_len)
    assembler = asm_cfg.build_assembler(kb, kb_index, feature_version=feature_version)

    LOGGER.info("Assembling TRAIN pools ...")
    train_pools, train_gold = build_note_pools(
        notes_csv=train_csv, assembler=assembler,
        bm25_index_path=bm25_index, llm_sources_csv=train_llm_sources,
        keep_llm_sources=keep_llm_sources, is_train=True,
        gold_code_col=gold_code_col, note_id_col=note_id_col,
        note_text_col=note_text_col, section_cols=section_cols,
        include_gold=True, max_snippets=max_snippets,
        snippet_max_words=snippet_max_words, snippet_overlap_words=snippet_overlap_words,
        snippet_select=snippet_select, snippet_pool_cap=snippet_pool_cap,
        limit=max_train_notes, cache_path=train_pool_cache,
    )

    val_pools = None
    val_gold: Dict[str, set] = {}
    if val_csv is not None:
        LOGGER.info("Assembling VAL pools ...")
        val_pools, val_gold = build_note_pools(
            notes_csv=val_csv, assembler=assembler,
            bm25_index_path=bm25_index, llm_sources_csv=val_llm_sources,
            keep_llm_sources=keep_llm_sources, is_train=False,
            gold_code_col=gold_code_col, note_id_col=note_id_col,
            note_text_col=note_text_col, section_cols=section_cols,
            include_gold=True, max_snippets=max_snippets,
            snippet_max_words=snippet_max_words, snippet_overlap_words=snippet_overlap_words,
            snippet_select=snippet_select, snippet_pool_cap=snippet_pool_cap,
            limit=max_val_notes, cache_path=val_pool_cache,
        )

    # ---- dataset / loader ----
    dataset = PoolExampleDataset(
        train_pools, train_gold, feature_version=feature_version,
        neg_per_pos=neg_per_pos, max_neg_per_note=max_neg_per_note, seed=seed,
        kb=kb,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training examples assembled; check inputs.")

    tokenizer = None
    need_text = arch == "cross_encoder"
    if need_text:
        from cpt_rec.baselines.common import resolve_local_model

        tokenizer = AutoTokenizer.from_pretrained(
            resolve_local_model(encoder, models_dir=local_models_dir)
        )
    collator = PairCollator(
        tokenizer, max_length=max_pair_len, need_text=need_text,
        pair_truncation=pair_truncation,
    )
    # With forked DataLoader workers, HF fast tokenizers must disable their own
    # Rust parallelism (else a noisy fork warning + potential slowdown); with 0
    # workers we leave it on so the single-process tokenizer stays multi-threaded.
    if num_workers > 0:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Shared loader knobs. ``prefetch_factor`` and ``persistent_workers`` are only
    # valid when num_workers>0, so gate them. None of these change which examples
    # land in a batch (sampling is done once in the dataset ctor), so the loop is
    # result-identical regardless of worker count.
    loader_kw: Dict[str, object] = {
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": str(device).startswith("cuda"),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kw["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        **loader_kw,
    )

    # ---- model / optim ----
    # ``core_model`` is the real module (used for val scoring + checkpointing).
    # When several GPUs are visible we wrap a DataParallel view around it for the
    # training forward only — the cross-encoder's transformer matmuls are the
    # bottleneck and shard cleanly across replicas, while pool assembly /
    # validation stay single-process (no DDP cache race or rank coordination).
    core_model = build_verifier(
        arch=arch, encoder_name=encoder, n_features=n_features,
        pooling=pooling, dropout=dropout, feature_mlp_hidden=feature_mlp_hidden,
        local_models_dir=local_models_dir,
    ).to(device)

    n_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    use_dp = n_gpus > 1 and arch == "cross_encoder" and not single_gpu
    model = torch.nn.DataParallel(core_model) if use_dp else core_model
    if use_dp:
        LOGGER.info("DataParallel across %d GPUs (global batch=%d, ~%d/GPU)",
                    n_gpus, batch_size, max(1, batch_size // n_gpus))
    elif single_gpu and n_gpus > 1:
        LOGGER.info("single_gpu set: DataParallel disabled, training on 1 of %d "
                    "visible GPUs (run separate configs per GPU to use all)", n_gpus)

    optimizer = torch.optim.AdamW(core_model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    pos_weight = None
    if use_pos_weight:
        pos_weight = torch.tensor([dataset.pos_weight()], device=device)
        LOGGER.info("Using BCE pos_weight=%.3f", float(pos_weight.item()))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Sweep up to ~0.998 with extra resolution in the high tail. With a
    # subsampled / pos-weighted training prior applied to a heavily imbalanced
    # (~1:hundreds) validation pool the model over-predicts, so the optimal
    # operating point sits well above 0.5 — often >0.9. Capping the grid at 0.9
    # pins ``best_threshold`` to the boundary and understates micro-F1.
    thresholds = list(np.round(
        np.unique(np.concatenate([
            np.linspace(0.05, 0.90, 18),
            np.linspace(0.91, 0.998, 14),
        ])),
        4,
    ))
    history: List[Dict] = []
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    best_threshold: float = 0.5

    global_step = 0
    def _build_config(history, best_value, best_epoch, best_threshold):
        return {
        "model_type": "verifier",
        "arch": arch,
        "encoder": encoder,
        "pooling": pooling,
        "feature_version": feature_version,
        "feature_order": list(order),
        "n_features": n_features,
        "max_pair_len": max_pair_len,
        "bi_max_length": resolved_bi_len,
        "pair_truncation": pair_truncation,
        "feature_mlp_hidden": feature_mlp_hidden,
        "assembler": asm_cfg.to_config_dict(),
        "keep_llm_sources": list(keep_llm_sources),
        "kb_index_dir": str(kb_index_dir),
        "kb_csv": str(kb_csv),
        "training": {
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "neg_per_pos": neg_per_pos,
            "max_neg_per_note": max_neg_per_note,
            "use_pos_weight": use_pos_weight,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
            "num_workers": num_workers,
            "single_gpu": single_gpu,
            "enable_tf32": enable_tf32,
            "history": history,
            "best_metric": "val_micro_f1",
            "best_value": best_value,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "val_csv": str(val_csv) if val_csv else None,
            "bm25_index": str(bm25_index) if bm25_index else None,
            "train_llm_sources": str(train_llm_sources) if train_llm_sources else None,
            "val_llm_sources": str(val_llm_sources) if val_llm_sources else None,
        },
    }

    # A 6-10 h train that dies inside the epoch loop used to leave ONLY
    # best_state.pt -- no config.json, so predict_verifier could not load it and
    # the whole run was unrecoverable. Write the static half of the config up
    # front under a DIFFERENT name: config.json stays the completion marker
    # (check_train_runs.py keys off it), while this file makes a crashed run
    # recoverable by `mv config.partial.json config.json` -- the best_* fields
    # are null, which predict does not read.
    with open(model_out / "config.partial.json", "w") as f:
        json.dump(_build_config([], None, None, None), f, indent=2, sort_keys=True)
    LOGGER.info("Wrote %s (crash-recovery config; config.json is written at the end)",
                model_out / "config.partial.json")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for step, batch in enumerate(loader):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            if need_text:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    features,
                )
            else:
                logits = model(features=features)
            loss = bce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += float(loss.detach().cpu())
            n += 1
            if n % log_every == 0:
                LOGGER.info(
                    "epoch=%d step=%d loss=%.4f lr=%.2e",
                    epoch, global_step, running / max(1, n),
                    scheduler.get_last_lr()[0],
                )

        rec: Dict[str, object] = {
            "epoch": epoch, "train_mean_loss": running / max(1, n),
        }
        if val_pools is not None:
            val_metrics = evaluate_pools(
                core_model, tokenizer, val_pools, val_gold, order, device, arch,
                max_pair_len=max_pair_len, thresholds=thresholds,
                score_batch_size=score_batch_size,
                pair_truncation=pair_truncation,
            )
            rec.update(val_metrics)
            LOGGER.info(
                "Epoch %d val: micro_f1=%.4f @thr=%.3f (p=%.4f r=%.4f) f1@0.5=%.4f",
                epoch, val_metrics["best_micro_f1"], val_metrics["best_threshold"],
                val_metrics["best_micro_precision"], val_metrics["best_micro_recall"],
                val_metrics["f1_at_0.5"],
            )
            cur = val_metrics["best_micro_f1"]
            if best_value is None or cur > best_value:
                best_value = cur
                best_epoch = epoch
                best_threshold = val_metrics["best_threshold"]
                torch.save(core_model.state_dict(), model_out / "best_state.pt")
                LOGGER.info("Epoch %d: new best micro_f1=%.4f -> best_state.pt", epoch, cur)
        history.append(rec)

    # ---- save ----
    torch.save(core_model.state_dict(), model_out / "verifier.pt")
    if tokenizer is not None:
        tokenizer.save_pretrained(model_out / "tokenizer")
    with open(model_out / "config.json", "w") as f:
        json.dump(_build_config(history, best_value, best_epoch, best_threshold),
                  f, indent=2, sort_keys=True)
    # config.json is the completion marker (check_train_runs.py keys off it), so
    # the partial written before the epoch loop must not outlive a finished run.
    (model_out / "config.partial.json").unlink(missing_ok=True)
    LOGGER.info("Saved verifier artifacts -> %s", model_out)
    return {
        "best_value": best_value,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "n_train_examples": len(dataset),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the candidate verifier.")
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--kb", dest="kb_csv", required=True, type=Path)
    p.add_argument("--kb-index-dir", required=True, type=Path)
    p.add_argument("--model-out", required=True, type=Path)
    p.add_argument("--bm25-index", type=Path, default=None,
                   help="Train-note BM25 index (bm25.pkl) for the neighbor source.")
    p.add_argument("--train-llm-sources", type=Path, default=None,
                   help="LLM candidate-prior .sources.csv for the TRAIN split "
                        "(optional; the reported M6 recipe does not use one). "
                        "Format: docs/OWN_CORPUS.md.")
    p.add_argument("--val-llm-sources", type=Path, default=None,
                   help="LLM candidate-prior .sources.csv for the VAL split.")
    p.add_argument("--keep-llm-sources", nargs="*", default=["llm_concept"])
    p.add_argument("--arch", default="cross_encoder",
                   choices=["cross_encoder", "feature_mlp"])
    p.add_argument("--feature-version", default="v1a", choices=list(FEATURE_ORDERS))
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--feature-mlp-hidden", type=int, default=128)
    p.add_argument("--max-pair-len", type=int, default=192)
    p.add_argument("--bi-max-length", type=int, default=None,
                   help="Query-side token budget for the snippet bi-encoder "
                        "(dense retrieval + best-snippet selection) during pool "
                        "assembly. Default None = follow --max-pair-len, the "
                        "historical behavior (so all existing recipes are "
                        "byte-identical). Set explicitly to decouple the "
                        "cross-encoder pair window from retrieval, e.g. "
                        "--max-pair-len 320 --bi-max-length 192 isolates the "
                        "window effect on the champion's pools. Recorded in "
                        "config.json and honored at predict time.")
    p.add_argument("--pair-truncation", default="longest_first",
                   choices=["longest_first", "only_first", "only_second"],
                   help="HF truncation strategy for the (evidence, code) pair. "
                        "Evidence is text_a / first; code+card is text_b / second. "
                        "'longest_first' (default) trims whichever side is longer "
                        "and lets card text evict evidence; 'only_second' protects "
                        "evidence by trimming only the code side (use with a wider "
                        "--max-pair-len, e.g. 320, since evidence is the long side).")
    p.add_argument("--bm25-top-k", type=int, default=25)
    p.add_argument("--dense-top-k", type=int, default=25)
    p.add_argument("--neighbor-top-k", type=int, default=25)
    p.add_argument("--llm-norm-k", type=int, default=50)
    p.add_argument("--max-snippets", type=int, default=32)
    p.add_argument("--snippet-max-words", type=int, default=180)
    p.add_argument("--snippet-overlap-words", type=int, default=60)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--neg-per-pos", type=int, default=None,
                   help="Keep all positives + N negatives/positive per note "
                        "(default: keep full pool, matching inference).")
    p.add_argument("--max-neg-per-note", type=int, default=None,
                   help="Negatives kept for notes with no in-pool gold "
                        "(only used with --neg-per-pos).")
    p.add_argument("--use-pos-weight", action="store_true",
                   help="Weight the BCE positive term by n_neg/n_pos.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers for parallel tokenization. Result-"
                        "identical to 0 (negatives are sampled once in the dataset "
                        "ctor); higher just removes GPU input-starvation. Try 16.")
    p.add_argument("--prefetch-factor", type=int, default=None,
                   help="Batches each worker prefetches (only used when "
                        "--num-workers > 0; PyTorch default is 2).")
    p.add_argument("--single-gpu", action="store_true",
                   help="Disable DataParallel and train on a single GPU even when "
                        "several are visible. For this ~110M model DP overhead "
                        "often exceeds its benefit; run separate configs per GPU "
                        "(CUDA_VISIBLE_DEVICES=0/1/2/3) to use all four. Off = "
                        "byte-identical to the current DataParallel path.")
    p.add_argument("--tf32", dest="enable_tf32", action="store_true",
                   help="Enable TF32 tensor-core matmuls (~1.3-1.6x on H100, "
                        "sub-1e-3 numeric change). Off = byte-identical champion.")
    p.add_argument("--section-cols", nargs="*", default=None)
    p.add_argument(
        "--snippet-select", choices=["length", "cosmax", "cosmargin"], default="length",
        help="S2 evidence-snippet selection. 'length' (default) is the historical "
             "path: snippets_for_note keeps the longest windows, so the section "
             "list is doing all the relevance work. 'cosmax'/'cosmargin' instead "
             "rank windows by their similarity to the KB, which is what makes "
             "dropping --section-cols viable.",
    )
    p.add_argument(
        "--snippet-pool-cap", type=int, default=0,
        help="Max windows encoded before S2 selection (0 = uncapped). Bounds the "
             "extra encode cost of whole-note evidence. Ignored when "
             "--snippet-select length.",
    )
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)
    p.add_argument("--max-train-notes", type=int, default=None)
    p.add_argument("--max-val-notes", type=int, default=None)
    p.add_argument("--train-pool-cache", type=Path, default=None)
    p.add_argument("--val-pool-cache", type=Path, default=None)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--score-batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--local-models-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        kb_csv=args.kb_csv,
        kb_index_dir=args.kb_index_dir,
        model_out=args.model_out,
        bm25_index=args.bm25_index,
        train_llm_sources=args.train_llm_sources,
        val_llm_sources=args.val_llm_sources,
        keep_llm_sources=args.keep_llm_sources,
        arch=args.arch,
        feature_version=args.feature_version,
        encoder=args.encoder,
        pooling=args.pooling,
        dropout=args.dropout,
        feature_mlp_hidden=args.feature_mlp_hidden,
        max_pair_len=args.max_pair_len,
        bi_max_length=args.bi_max_length,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        neighbor_top_k=args.neighbor_top_k,
        llm_norm_k=args.llm_norm_k,
        max_snippets=args.max_snippets,
        snippet_max_words=args.snippet_max_words,
        snippet_overlap_words=args.snippet_overlap_words,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        neg_per_pos=args.neg_per_pos,
        max_neg_per_note=args.max_neg_per_note,
        use_pos_weight=args.use_pos_weight,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        section_cols=args.section_cols,
        snippet_select=args.snippet_select,
        snippet_pool_cap=args.snippet_pool_cap,
        gold_code_col=args.gold_code_col,
        note_id_col=args.note_id_col,
        note_text_col=args.note_text_col,
        max_train_notes=args.max_train_notes,
        max_val_notes=args.max_val_notes,
        train_pool_cache=args.train_pool_cache,
        val_pool_cache=args.val_pool_cache,
        encode_batch_size=args.encode_batch_size,
        score_batch_size=args.score_batch_size,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        local_models_dir=args.local_models_dir,
        pair_truncation=args.pair_truncation,
        single_gpu=args.single_gpu,
        enable_tf32=args.enable_tf32,
    )


if __name__ == "__main__":
    main()
