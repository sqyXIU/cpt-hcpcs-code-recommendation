#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Predict with a trained candidate verifier.

Loads a verifier run (``config.json`` + ``best_state.pt`` / ``verifier.pt`` +
tokenizer), reassembles each note's candidate pool exactly as at train time
(``include_gold=False``), scores every ``(evidence, code)`` pair, then keeps
candidates above a decision threshold (default = the trained
``best_threshold``) or the top-k highest-scoring candidates.

Outputs follow the shared prediction contract so the evaluator
consumes verifier predictions unchanged:

* CSV: ``note_id, pred_codes, pred_scores`` (pipe-separated, aligned).
* NPZ (on by default): ``note_ids``, ``candidate_codes``, ``probs`` — the
  **full** candidate pool with scores, accumulated before thresholding, so the
  shortlist-review metrics (``cptrec-evaluate --scores-npz``) and any alternative
  operating point are recoverable from one inference pass.
* NPZ (``--save-npz``): object arrays ``note_ids / candidate_codes / probs``
  carrying the *full* per-note candidate pool with raw probabilities (for
  threshold sweeps / calibration downstream).
* Ledger JSONL (``--emit-ledger``): one
  :class:`cpt_rec.evidence.ledger.CandidateLedgerEntry` per candidate (sources,
  ranks, best snippet, family, verifier score, decode decision), reconcilable
  1:1 with the NPZ. Additive and default-off — the CSV/NPZ are byte-identical
  whether or not it is requested; the import is lazy so the frozen verifier's
  import graph is untouched when the flag is absent.

Default output path: ``<model>/predictions/<split>/predictions.csv``.

An LLM candidate prior can be applied at inference via ``--llm-sources`` (a
``.sources.csv`` for the target split, format in ``docs/OWN_CORPUS.md``); this
is how a v1a verifier benefits from a concept generator without train-time LLM
generation, and how v1b reads its ``llm_*`` features.  The reported M6 result
does not use one — the flag is default-off and the generator is not shipped.

CLI
---

::

    cptrec-verifier-predict \\
        --notes outputs/datasets/vumc/test_eval_sectioned.csv \\
        --model outputs/verifier/sections192_baseline/ \\
        --kb data/kb/codes_with_ranges.csv \\
        --kb-index-dir outputs/indices/code_kb_faiss/default/ \\
        --bm25-index outputs/indices/note_bm25/bm25.pkl \\
        --split test
"""

from __future__ import annotations

import os

# Silence the HF fast-tokenizer fork warning (see train_verifier.py): disable
# Rust-level tokenizer parallelism before transformers/tokenizers is imported.
# setdefault lets the caller still override via the environment.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from cpt_rec.baselines.common import Prediction, write_predictions
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.pipeline.pool.candidate_gen import KBCandidateIndex
from cpt_rec.pipeline.pool.candidate_pool import AssemblerConfig, FEATURE_ORDERS
from cpt_rec.pipeline.crossencoder.verifier_data import build_note_pools, score_pool

# ``build_verifier`` pulls in torch.  It is imported inside ``predict`` instead of
# here so that the torch-free helpers in this module -- notably
# ``build_ledger_entries`` -- stay importable without the ``gpu`` extra installed.

LOGGER = logging.getLogger(__name__)


def _load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "verifier":
        LOGGER.warning(
            "config.json model_type=%r (expected 'verifier')", cfg.get("model_type")
        )
    return cfg


def _resolve_state_path(model_dir: Path, prefer_best: bool) -> Path:
    best = model_dir / "best_state.pt"
    final = model_dir / "verifier.pt"
    if prefer_best and best.exists():
        return best
    if final.exists():
        return final
    if best.exists():
        return best
    raise FileNotFoundError(
        f"No verifier weights in {model_dir} (looked for best_state.pt / verifier.pt)"
    )


def _default_out_paths(
    model_dir: Path,
    split: str,
    explicit_out: Optional[Path],
    save_npz: bool,
    explicit_npz: Optional[Path],
):
    if explicit_out is not None:
        out_csv = Path(explicit_out)
    else:
        out_csv = model_dir / "predictions" / split / "predictions.csv"
    if explicit_npz is not None:
        out_npz = Path(explicit_npz)
    elif save_npz:
        out_npz = out_csv.with_suffix(".npz")
    else:
        out_npz = None
    return out_csv, out_npz


def build_ledger_entries(pool, probs, keep_idx, kb):
    """Map one scored :class:`NotePool` to its per-candidate ledger entries.

    Evidence-ledger enrichment: dump exactly what the verifier saw — sources,
    normalized rank/score feats, best snippet, KB family, the verifier score,
    and whether the decode kept the candidate (``selected_pre_rule``). Later
    agents (A4/A7/A9/A10) fill the remaining fields. One entry per pool record,
    aligned to ``probs`` — so the JSONL reconciles 1:1 with the NPZ.

    Lazy-imports :class:`CandidateLedgerEntry` so the frozen verifier's import
    graph is untouched on the default (no-ledger) path; this helper is only ever
    called when ``--emit-ledger`` is set.
    """
    from cpt_rec.evidence.ledger import CandidateLedgerEntry  # noqa: WPS433

    kept = {int(i) for i in keep_idx}
    entries = []
    for i, rec in enumerate(pool.records):
        feats = rec.feats or {}
        ranks = {
            k[:-len("_rank")]: float(v)
            for k, v in feats.items() if k.endswith("_rank")
        }
        confs = {
            name: float(feats[key])
            for name, key in (
                ("kb_bm25", "kb_bm25_score"),
                ("kb_dense", "kb_dense_score"),
                ("nbr", "nbr_frac"),
                ("llm", "llm_confidence"),
            )
            if key in feats
        }
        snippet = None
        bsi = rec.best_snippet_idx
        if bsi is not None and 0 <= bsi < len(pool.snippet_texts):
            snippet = pool.snippet_texts[bsi]
        entries.append(
            CandidateLedgerEntry(
                note_id=pool.note_id,
                code=rec.code,
                candidate_sources={s: True for s in rec.sources},
                source_ranks=ranks,
                source_confidences=confs,
                best_snippet=snippet,
                code_description=rec.description,
                code_family=kb.family(rec.code),
                verifier_score=float(probs[i]),
                selected_pre_rule=(i in kept),
            )
        )
    return entries


def predict(
    notes_csv: Path,
    model_dir: Path,
    kb_csv: Path,
    kb_index_dir: Path,
    bm25_index: Optional[Path] = None,
    llm_sources_csv: Optional[Path] = None,
    split: str = "test",
    out_csv: Optional[Path] = None,
    save_npz: bool = True,
    out_npz: Optional[Path] = None,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    min_predictions: int = 0,
    section_cols: Optional[List[str]] = None,
    gold_code_col: str = "proc_codes",
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    limit: Optional[int] = None,
    pool_cache: Optional[Path] = None,
    score_batch_size: int = 128,
    use_final: bool = False,
    device: Optional[str] = None,
    local_models_dir: Optional[Path] = None,
    emit_ledger: bool = False,
    ledger_path: Optional[Path] = None,
) -> Path:
    import torch

    from cpt_rec.pipeline.crossencoder.verifier import build_verifier
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    cfg = _load_config(model_dir)
    arch = cfg.get("arch", "cross_encoder")
    feature_version = cfg.get("feature_version", "v1a")
    order = cfg.get("feature_order") or FEATURE_ORDERS[feature_version]
    n_features = int(cfg.get("n_features", len(order)))
    encoder = cfg.get("encoder")
    pooling = cfg.get("pooling", "cls")
    max_pair_len = int(cfg.get("max_pair_len", 192))
    # Query-side bi-encoder budget for pool assembly. Models trained before the
    # --bi-max-length flag have no such key; they assembled at max_pair_len, so
    # the fallback reproduces their pools byte-identically.
    bi_max_length = int(cfg.get("bi_max_length", max_pair_len))
    pair_truncation = cfg.get("pair_truncation", "longest_first")
    asm_cfg = AssemblerConfig.from_config_dict(cfg)
    keep_llm_sources = tuple(cfg.get("keep_llm_sources", ("llm_concept",)))

    if threshold is None:
        threshold = float(cfg.get("training", {}).get("best_threshold", 0.5))
        LOGGER.info("Using trained decision threshold %.3f", threshold)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Same silent-CPU-fallback hole as train_verifier (fixed 2026-08-26). A
    # CPU predict does not cost days, but on the MIMIC test split it is still
    # hours instead of minutes, and nothing in the output says which device ran.
    if str(device) == "cpu":
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if _cvd is not None and os.environ.get("CPT_REC_ALLOW_CPU_TRAIN") != "1":
            raise SystemExit(
                "REFUSING TO PREDICT ON CPU.\n"
                f"  CUDA_VISIBLE_DEVICES={_cvd!r} is set, so a GPU was intended, "
                "but torch.cuda.is_available() is False.\n"
                "  An empty value (from an unset $G1/$G2/$G3) hides every GPU.\n"
                "  Set CPT_REC_ALLOW_CPU_TRAIN=1 to predict on CPU deliberately."
            )
        LOGGER.warning("PREDICTING ON CPU -- expect a large wall-clock penalty.")
    else:
        LOGGER.info("device=%s  visible GPUs=%d  CUDA_VISIBLE_DEVICES=%s",
                    device, torch.cuda.device_count(),
                    os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))

    LOGGER.info("Loading KB + KB index ...")
    kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
    kb_index = KBCandidateIndex(kb_index_dir, bi_max_length=bi_max_length)
    assembler = asm_cfg.build_assembler(kb, kb_index, feature_version=feature_version)

    LOGGER.info("Assembling candidate pools for %s ...", notes_csv)
    pools, _gold = build_note_pools(
        notes_csv=notes_csv, assembler=assembler,
        bm25_index_path=bm25_index, llm_sources_csv=llm_sources_csv,
        keep_llm_sources=keep_llm_sources, is_train=False,
        gold_code_col=gold_code_col, note_id_col=note_id_col,
        note_text_col=note_text_col, section_cols=section_cols,
        include_gold=False,
        max_snippets=asm_cfg.max_snippets,
        snippet_max_words=asm_cfg.snippet_max_words,
        snippet_overlap_words=asm_cfg.snippet_overlap_words,
        # Restored from config.json's "assembler" block -- predict must select
        # evidence the same way train did or the pools diverge silently.
        snippet_select=asm_cfg.snippet_select,
        snippet_pool_cap=asm_cfg.snippet_pool_cap,
        limit=limit, cache_path=pool_cache,
    )

    # ---- model ----
    tokenizer = None
    if arch == "cross_encoder":
        tok_dir = model_dir / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            str(tok_dir) if tok_dir.exists() else encoder
        )
    model = build_verifier(
        arch=arch, encoder_name=encoder, n_features=n_features,
        pooling=pooling, dropout=0.0,
        feature_mlp_hidden=int(cfg.get("feature_mlp_hidden", 128)),
        local_models_dir=local_models_dir,
    )
    state_path = _resolve_state_path(model_dir, prefer_best=not use_final)
    LOGGER.info("Loading verifier weights from %s", state_path)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.to(device).eval()

    # ---- optional evidence-ledger sink (default-off so the frozen
    # outputs are byte-identical when not requested) ----
    ledger_entries = None if not emit_ledger else []

    # ---- score + decode ----
    predictions: List[Prediction] = []
    all_note_ids: List[str] = []
    all_codes: List[List[str]] = []
    all_probs: List[List[float]] = []

    for pool in tqdm(pools, desc="scoring pools", unit="note"):
        codes = [r.code for r in pool.records]
        probs = score_pool(
            model, pool, order, device, arch,
            tokenizer=tokenizer, max_length=max_pair_len, batch_size=score_batch_size,
            truncation=pair_truncation,
        )
        all_note_ids.append(pool.note_id)
        all_codes.append(list(codes))
        all_probs.append([float(p) for p in probs])

        order_idx = np.argsort(-probs)
        if top_k is not None:
            keep_idx = list(order_idx[:top_k])
        else:
            keep_idx = [int(i) for i in order_idx if probs[i] >= threshold]
            if min_predictions and len(keep_idx) < min_predictions:
                keep_idx = list(order_idx[:min_predictions])
        kept_codes = [codes[i] for i in keep_idx]
        kept_scores = [float(probs[i]) for i in keep_idx]
        predictions.append(
            Prediction(note_id=pool.note_id, codes=kept_codes, scores=kept_scores)
        )

        if ledger_entries is not None:
            ledger_entries.extend(
                build_ledger_entries(pool, probs, keep_idx, kb)
            )

    out_csv, out_npz = _default_out_paths(
        model_dir, split, out_csv, save_npz, out_npz
    )
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(predictions, out_csv, include_scores=True)
    LOGGER.info("Wrote %d predictions -> %s", len(predictions), out_csv)

    if out_npz is not None:
        out_npz = Path(out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            note_ids=np.array(all_note_ids, dtype=object),
            candidate_codes=np.array(all_codes, dtype=object),
            probs=np.array(all_probs, dtype=object),
        )
        LOGGER.info("Wrote raw scores NPZ -> %s", out_npz)

    if ledger_entries is not None:
        from cpt_rec.evidence.ledger import write_ledger  # noqa: WPS433

        lpath = Path(ledger_path) if ledger_path is not None else out_csv.with_name("ledger.jsonl")
        n = write_ledger(ledger_entries, lpath)
        LOGGER.info("Wrote %d ledger entries -> %s", n, lpath)

    return out_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with a trained verifier.")
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--model", dest="model_dir", required=True, type=Path)
    p.add_argument("--kb", dest="kb_csv", required=True, type=Path)
    p.add_argument("--kb-index-dir", required=True, type=Path)
    p.add_argument("--bm25-index", type=Path, default=None,
                   help="Train-note BM25 index for the neighbor source.")
    p.add_argument("--llm-sources", type=Path, default=None,
                   help="LLM candidate-prior .sources.csv for this split "
                        "(applies the LLM-concept source at inference).")
    p.add_argument("--split", default="test",
                   help="Names the default output subdir "
                        "(<model>/predictions/<split>/predictions.csv).")
    p.add_argument("--out", dest="out_csv", type=Path, default=None,
                   help="Override the predictions CSV path.")
    p.add_argument("--save-npz", action="store_true", default=True,
                   help="Write raw full-pool candidate scores as an NPZ companion "
                        "(default: on). Required by `cptrec-evaluate --scores-npz` for "
                        "the shortlist-review metrics (R@B etc.); the NPZ holds every "
                        "candidate's score regardless of the decision threshold, so "
                        "any operating point can be re-scored without re-running.")
    p.add_argument("--no-save-npz", dest="save_npz", action="store_false",
                   help="Suppress the NPZ companion (legacy behaviour).")
    p.add_argument("--out-npz", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold (default: config best_threshold).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Keep top-k by score instead of thresholding.")
    p.add_argument("--min-predictions", type=int, default=0,
                   help="Backfill top-scoring candidates when a note falls "
                        "below this many thresholded predictions.")
    p.add_argument("--section-cols", nargs="*", default=None)
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--pool-cache", type=Path, default=None)
    p.add_argument("--score-batch-size", type=int, default=128)
    p.add_argument("--use-final", action="store_true",
                   help="Use verifier.pt (last epoch) instead of best_state.pt.")
    p.add_argument("--device", default=None)
    p.add_argument("--local-models-dir", type=Path, default=None)
    p.add_argument("--emit-ledger", action="store_true",
                   help="Also write an evidence-ledger JSONL (one entry per "
                        "candidate, reconcilable 1:1 with the NPZ). Default-off; "
                        "the verifier's outputs are unchanged when omitted.")
    p.add_argument("--ledger-out", dest="ledger_path", type=Path, default=None,
                   help="Override the ledger path (default: ledger.jsonl beside "
                        "the predictions CSV).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    predict(
        notes_csv=args.notes,
        model_dir=args.model_dir,
        kb_csv=args.kb_csv,
        kb_index_dir=args.kb_index_dir,
        bm25_index=args.bm25_index,
        llm_sources_csv=args.llm_sources,
        split=args.split,
        out_csv=args.out_csv,
        save_npz=args.save_npz,
        out_npz=args.out_npz,
        threshold=args.threshold,
        top_k=args.top_k,
        min_predictions=args.min_predictions,
        section_cols=args.section_cols,
        gold_code_col=args.gold_code_col,
        note_id_col=args.note_id_col,
        note_text_col=args.note_text_col,
        limit=args.limit,
        pool_cache=args.pool_cache,
        score_batch_size=args.score_batch_size,
        use_final=args.use_final,
        device=args.device,
        local_models_dir=args.local_models_dir,
        emit_ledger=args.emit_ledger,
        ledger_path=args.ledger_path,
    )


if __name__ == "__main__":
    main()
