#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
M2 (``m2_label_attention``) — Clinical-Longformer multi-label
classifier.

Fine-tunes ``yikuan8/Clinical-Longformer`` (default) with a multi-label
BCE-with-logits head over CPT/HCPCS codes that pass a frequency floor.
This is the strong long-doc neural baseline; requires a GPU.

``train --head label-attention`` selects the variant M2 reports: a
PLM-ICD-style per-label attention head over all token states instead of
the stock [CLS] sequence-classification head.  The head choice is recorded
in ``labels.json``, so ``predict`` / ``dump-scores`` need no extra flags.

Three CLI subcommands
---------------------

::

    cptrec-m2-longformer train \\
        --train outputs/datasets/vumc/train_eval_sectioned.csv \\
        --val outputs/datasets/vumc/val_eval_sectioned.csv \\
        --model-out outputs/baselines/m2_label_attention/model/ \\
        --base-model yikuan8/Clinical-Longformer \\
        --max-length 4096 --batch-size 2 --grad-accum 8 \\
        --epochs 3 --lr 3e-5 --min-code-freq 5

    cptrec-m2-longformer predict \\
        --notes outputs/datasets/vumc/test_eval_sectioned.csv \\
        --model outputs/baselines/m2_label_attention/model/ \\
        --out outputs/baselines/m2_label_attention/predictions/test.csv \\
        --threshold 0.3

    cptrec-m2-longformer dump-scores \\
        --notes outputs/datasets/vumc/val_eval_sectioned.csv \\
        --model outputs/baselines/m2_label_attention/model/ \\
        --out-npz outputs/baselines/m2_label_attention/predictions/val_scores.npz

The predict step optionally takes ``--calibration <path>`` pointing at a
calibration JSON produced by
:mod:`cpt_rec.common.evaluation.calibration` to apply per-code
thresholds.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cpt_rec.baselines.common import (
    Prediction,
    apply_seed_and_limit,
    load_notes_for_prediction,
    log_prediction_stats,
    stats_sidecar,
    resolve_local_model,
    write_predictions,
)
from cpt_rec.common.preprocess.code_utils import (
    pipe_split_codes as _pipe_split,  # re-exported; tests import this name
)

LOGGER = logging.getLogger(__name__)


# Default refers to the basename used for the offline cache layout
# (``models/Clinical-Longformer/``).  The canonical Hub repo id is
# ``yikuan8/Clinical-Longformer`` — pass ``--base-model
# yikuan8/Clinical-Longformer`` on a machine with internet access if
# you want to pull from the Hub directly.
DEFAULT_BASE_MODEL = "Clinical-Longformer"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_label_matrix(
    code_lists: Sequence[Sequence[str]],
    label_to_idx: Dict[str, int],
) -> np.ndarray:
    n_docs, n_labels = len(code_lists), len(label_to_idx)
    Y = np.zeros((n_docs, n_labels), dtype=np.float32)
    for i, codes in enumerate(code_lists):
        for c in codes:
            j = label_to_idx.get(c)
            if j is not None:
                Y[i, j] = 1.0
    return Y


# ---------------------------------------------------------------------------
# Label-attention head (PLM-ICD style)
# ---------------------------------------------------------------------------

def _build_label_attention_model(encoder, n_labels: int):
    """
    Wrap a bare ``AutoModel`` encoder with a per-label attention head
    (CAML/PLM-ICD lineage): every label attends over ALL token states and
    scores its own attended vector.  This removes the single-[CLS]-vector
    bottleneck that the stock sequence-classification head funnels ~2000
    labels through — the suspected cause of the stock head's recall collapse.

    PLM-ICD chunks its 512-token encoder to reach long documents; the
    Longformer already encodes 4096 tokens natively, so the head sits
    directly on ``last_hidden_state`` and no chunking is needed.
    """
    import torch
    from torch import nn
    from transformers.modeling_outputs import SequenceClassifierOutput

    class LabelAttentionHead(nn.Module):
        def __init__(self, hidden_size: int, n_labels: int):
            super().__init__()
            self.proj = nn.Linear(hidden_size, hidden_size)
            self.query = nn.Linear(hidden_size, n_labels, bias=False)
            self.weight = nn.Parameter(torch.empty(n_labels, hidden_size))
            self.bias = nn.Parameter(torch.zeros(n_labels))
            nn.init.xavier_uniform_(self.weight)

        def forward(self, hidden_states, attention_mask):
            # hidden_states: (B, T, d); attention_mask: (B, T)
            z = torch.tanh(self.proj(hidden_states))
            scores = self.query(z)  # (B, T, L)
            mask = attention_mask.to(torch.bool).unsqueeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            alpha = torch.softmax(scores, dim=1)  # over tokens, per label
            v = torch.einsum("btl,btd->bld", alpha, hidden_states)
            return (v * self.weight.unsqueeze(0)).sum(-1) + self.bias

    class LabelAttentionModel(nn.Module):
        def __init__(self, encoder, n_labels: int):
            super().__init__()
            self.encoder = encoder
            self.config = encoder.config  # Trainer expects .config
            self.head = LabelAttentionHead(encoder.config.hidden_size, n_labels)
            import inspect as _inspect
            self._takes_gattn = (
                "global_attention_mask"
                in _inspect.signature(encoder.forward).parameters
            )

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            # Trainer calls this on the wrapper when args.gradient_checkpointing
            # is set; delegate to the encoder (the head holds no activations
            # worth checkpointing).
            kwargs = gradient_checkpointing_kwargs or {}
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=kwargs
            )

        def forward(self, input_ids=None, attention_mask=None,
                    global_attention_mask=None, labels=None, **kwargs):
            enc_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if self._takes_gattn:
                if global_attention_mask is None:
                    # Mirror the training dataset: global attention on token 0.
                    # (LongformerForSequenceClassification does this implicitly;
                    # the bare LongformerModel does not.)
                    global_attention_mask = torch.zeros_like(input_ids)
                    global_attention_mask[:, 0] = 1
                enc_kwargs["global_attention_mask"] = global_attention_mask
            hidden = self.encoder(**enc_kwargs).last_hidden_state
            logits = self.head(hidden, attention_mask)
            return SequenceClassifierOutput(logits=logits)

    return LabelAttentionModel(encoder, n_labels)


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------

def _build_dataset(texts, Y, tokenizer, max_length):
    import torch
    from torch.utils.data import Dataset

    class _MultiLabelDataset(Dataset):
        def __init__(self, texts, Y, tokenizer, max_length):
            self.texts = list(texts)
            self.Y = Y
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            # Longformer-specific: without explicit global attention on the
            # [CLS] token, [CLS] only sees the local 512-token sliding window
            # and the classifier head ends up reading from note headers
            # rather than the full document.  This is the documented
            # recommendation in `LongformerModel`.
            if "input_ids" in item:
                gattn = torch.zeros_like(item["input_ids"])
                gattn[0] = 1
                item["global_attention_mask"] = gattn
            item["labels"] = torch.tensor(self.Y[idx], dtype=torch.float32)
            return item

    return _MultiLabelDataset(texts, Y, tokenizer, max_length)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _pick_precision(fp16: bool, bf16: bool) -> Tuple[bool, bool]:
    """
    Pick the fp16/bf16 flag pair to hand to ``TrainingArguments``.

    Defaults to bf16 on H100/Ampere+ when ``--bf16`` is requested (better
    numerics, no loss-scaling); falls back to fp16 on CUDA where bf16
    isn't supported, and to neither on CPU.
    """
    try:
        import torch
    except ImportError:
        return (False, False)
    if not torch.cuda.is_available():
        return (False, False)
    bf16_ok = bf16 and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    if bf16_ok:
        return (False, True)
    return (bool(fp16), False)


def _compute_metrics(eval_pred):
    """
    Sweep a few thresholds and return the best micro-F1 — used as the
    ``metric_for_best_model`` so HF Trainer can pick the best epoch
    instead of the last one.  Threshold tuning happens later
    (calibration step), so we just need a checkpoint-selection signal.
    """
    if hasattr(eval_pred, "predictions"):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    labels_i = labels.astype(np.int8)
    best_f1 = 0.0
    best_t = 0.0
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        preds = (probs >= t).astype(np.int8)
        tp = float((preds & labels_i).sum())
        fp = float(preds.sum()) - tp
        fn = float(labels_i.sum()) - tp
        p = tp / max(tp + fp, 1.0)
        r = tp / max(tp + fn, 1.0)
        f1 = 2.0 * p * r / max(p + r, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return {"micro_f1_best": float(best_f1),
            "best_threshold": float(best_t)}


def train_longformer(
    train_csv: Path,
    model_out: Path,
    val_csv: Optional[Path] = None,
    base_model: str = DEFAULT_BASE_MODEL,
    max_length: int = 4096,
    min_code_freq: int = 5,
    batch_size: int = 2,
    grad_accum: int = 8,
    epochs: int = 3,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    gold_code_col: str = "proc_codes",
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    seed: int = 42,
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = False,
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    resume_from_checkpoint: Optional[Path] = None,
    test_csv: Optional[Path] = None,
    local_models_dir: Optional[Path] = None,
    focal_gamma: float = 2.0,
    head: str = "cls",
) -> Dict:
    try:
        import torch
        from transformers import (
            AutoModel,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "M2 training requires `torch` and `transformers`: "
            "`pip install torch transformers`"
        ) from exc

    model_out = Path(model_out)
    model_out.mkdir(parents=True, exist_ok=True)

    df_train = load_notes_for_prediction(
        train_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    if gold_code_col not in df_train.columns:
        raise ValueError(f"Train CSV missing gold column '{gold_code_col}'.")
    code_lists_train = [_pipe_split(s) for s in df_train[gold_code_col].fillna("")]

    freq = Counter(c for cs in code_lists_train for c in cs)
    labels = sorted(c for c, k in freq.items() if k >= min_code_freq)
    label_to_idx = {c: i for i, c in enumerate(labels)}
    LOGGER.info(
        "Training: %d docs, %d unique gold codes, %d after freq>=%d filter",
        len(df_train), len(freq), len(labels), min_code_freq,
    )
    if not labels:
        raise ValueError("No labels survived the frequency floor; lower --min-code-freq.")

    Y_train = _build_label_matrix(code_lists_train, label_to_idx)

    base_model_resolved = resolve_local_model(base_model, models_dir=local_models_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_resolved)
    if head == "label-attention":
        encoder = AutoModel.from_pretrained(base_model_resolved)
        model = _build_label_attention_model(encoder, n_labels=len(labels))
        LOGGER.info(
            "label-attention head: per-label attention over %d token "
            "states x %d labels (hidden=%d)", max_length, len(labels), encoder.config.hidden_size,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_resolved,
            num_labels=len(labels),
            problem_type="multi_label_classification",
            id2label={i: c for c, i in label_to_idx.items()},
            label2id=label_to_idx,
        )

    # Initialize the classifier head's output bias to the per-label log-odds
    # of the training marginal.  Under extreme imbalance, training otherwise
    # spends most of its gradient budget pushing each label's bias from 0
    # down to log(p_l) (~ -7 for the rare tail), starving the feature head.
    # Pre-baking the bias frees the optimizer to learn the input-dependent
    # contribution from step 0.
    import torch as _torch_init
    pos_counts_for_bias = Y_train.sum(axis=0).astype(np.float64)
    p_l = pos_counts_for_bias / float(max(Y_train.shape[0], 1))
    p_l_safe = np.clip(p_l, 1e-5, 1.0 - 1e-5)
    bias_init = np.log(p_l_safe / (1.0 - p_l_safe)).astype(np.float32)
    head_out_proj = None
    if head == "label-attention":
        head_out_proj = model.head  # per-label bias on the attention head
    elif hasattr(model, "classifier") and hasattr(model.classifier, "out_proj"):
        head_out_proj = model.classifier.out_proj  # Longformer / RoBERTa
    elif hasattr(model, "classifier") and hasattr(model.classifier, "bias"):
        head_out_proj = model.classifier  # BERT-style flat classifier
    if head_out_proj is not None and head_out_proj.bias is not None:
        with _torch_init.no_grad():
            head_out_proj.bias.copy_(_torch_init.tensor(bias_init))
        LOGGER.info(
            "Classifier bias <- log-marginal: range [%.2f, %.2f] median=%.2f",
            float(bias_init.min()), float(bias_init.max()),
            float(np.median(bias_init)),
        )
    else:
        LOGGER.warning(
            "Could not locate classifier output bias on this model; "
            "skipped log-marginal init."
        )

    if gradient_checkpointing:
        # `use_cache=False` is required for gradient checkpointing under HF;
        # without it, the cache + checkpointing combination raises.
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing on the model.")

    train_ds = _build_dataset(
        df_train["note_text"].tolist(), Y_train, tokenizer, max_length
    )
    eval_ds = None
    if val_csv is not None:
        df_val = load_notes_for_prediction(
            val_csv, note_id_col=note_id_col, note_text_col=note_text_col
        )
        code_lists_val = [_pipe_split(s) for s in df_val[gold_code_col].fillna("")]
        Y_val = _build_label_matrix(code_lists_val, label_to_idx)
        eval_ds = _build_dataset(
            df_val["note_text"].tolist(), Y_val, tokenizer, max_length
        )

    fp16_eff, bf16_eff = _pick_precision(fp16=fp16, bf16=bf16)
    LOGGER.info(
        "Precision: fp16=%s, bf16=%s (requested fp16=%s, bf16=%s)",
        fp16_eff, bf16_eff, fp16, bf16,
    )
    # Resolve the deprecated `tokenizer=` kwarg in transformers >= 4.46
    # by sniffing the Trainer signature.  Older versions still accept it,
    # newer versions want `processing_class=`.
    import inspect
    trainer_param = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )

    args_kwargs = dict(
        output_dir=str(model_out / "trainer_state"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=50,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        report_to=[],
        seed=seed,
        fp16=fp16_eff,
        bf16=bf16_eff,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=2,
    )
    # Checkpoint-best selection requires save and eval strategies to match,
    # and only makes sense when an eval set is present.
    if eval_ds is not None and save_strategy == "epoch":
        args_kwargs["load_best_model_at_end"] = True
        args_kwargs["metric_for_best_model"] = "micro_f1_best"
        args_kwargs["greater_is_better"] = True
    args = TrainingArguments(**args_kwargs)

    # ------------------------------------------------------------------
    # BCE pos_weight to counter extreme multi-label imbalance.
    #
    # With ~2000 labels and ~1.8 gold codes per note, the negative:positive
    # cell ratio is ~1000:1.  Vanilla BCEWithLogitsLoss collapses to
    # "always predict zero" under that imbalance.
    #
    # We use sqrt(neg/pos) capped at 50x rather than raw neg/pos capped
    # at 200x.  The raw-ratio version saturated for >50% of labels
    # (median pos_weight hit the 200 ceiling), which flattened per-label
    # calibration into a near-uniform weight and drove the model to a
    # second degenerate solution: all logits clustered near 0, gold and
    # non-gold score distributions overlapping, micro-F1 ~0.01.  Sqrt
    # scaling gives long-tail labels a moderate boost without that
    # saturation — a label with neg/pos=10000 gets weight ~50 instead
    # of the cap, while a label with neg/pos=4 gets ~2.
    # ------------------------------------------------------------------
    pos_counts = Y_train.sum(axis=0).astype(np.float64)
    neg_counts = float(Y_train.shape[0]) - pos_counts
    pos_counts_safe = np.clip(pos_counts, 1.0, None)
    pos_weight_cap = 50.0
    pos_weight = np.clip(
        np.sqrt(neg_counts / pos_counts_safe), 1.0, pos_weight_cap
    )
    import torch as _torch
    pos_weight_t = _torch.tensor(pos_weight, dtype=_torch.float32)
    LOGGER.info(
        "BCE pos_weight (sqrt scaling): median=%.1f  p90=%.1f  max=%.1f "
        "(capped at %.0fx); min pos_count=%d  max pos_count=%d",
        float(np.median(pos_weight)),
        float(np.percentile(pos_weight, 90)),
        float(pos_weight.max()),
        pos_weight_cap,
        int(pos_counts.min()),
        int(pos_counts.max()),
    )
    LOGGER.info(
        "Loss: weighted-BCE%s",
        f" + focal modulator (gamma={focal_gamma})" if focal_gamma > 0 else " (no focal)",
    )

    class WeightedBCETrainer(Trainer):
        """
        Trainer with pos_weight'd BCE optionally modulated by focal loss.

        The (1 - p_t)^gamma modulator amplifies gradient on hard examples
        and damps it on easy ones — breaks the "predict the marginal
        everywhere" equilibrium that vanilla weighted BCE settles into
        under extreme negative imbalance.  ``gamma=0`` disables the
        modulator and reduces this to plain weighted BCE.
        """

        _pos_weight = pos_weight_t  # captured at class definition time
        _gamma = float(focal_gamma)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            bce = _torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
                pos_weight=self._pos_weight.to(logits.device),
                reduction="none",
            )
            if self._gamma > 0.0:
                p = _torch.sigmoid(logits)
                p_t = _torch.where(labels > 0.5, p, 1.0 - p)
                loss = (((1.0 - p_t) ** self._gamma) * bce).mean()
            else:
                loss = bce.mean()
            return (loss, outputs) if return_outputs else loss

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        **{trainer_param: tokenizer},
    )
    if eval_ds is not None:
        trainer_kwargs["compute_metrics"] = _compute_metrics
    trainer = WeightedBCETrainer(**trainer_kwargs)
    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint
        else None,
    )

    # Persist model + tokenizer + label vocab in the same directory.
    if head == "label-attention":
        # The wrapper is a plain nn.Module: save the encoder in HF format
        # (so predict can AutoModel.from_pretrained the same dir) and the
        # attention head as a separate state dict.
        model.encoder.save_pretrained(model_out)
        _torch.save(model.head.state_dict(), model_out / "label_attention_head.pt")
    else:
        model.save_pretrained(model_out)
    tokenizer.save_pretrained(model_out)

    # Visibility on recall ceiling: how many val/test gold codes are out
    # of the trained label vocab?  Logged AND saved into labels.json.
    oov_audit: Dict[str, Dict[str, int]] = {}
    label_set = set(labels)
    for split_name, split_csv in (("val", val_csv), ("test", test_csv)):
        if split_csv is None:
            continue
        df_split = load_notes_for_prediction(
            split_csv, note_id_col=note_id_col, note_text_col=note_text_col
        )
        if gold_code_col not in df_split.columns:
            continue
        code_lists_split = [_pipe_split(s) for s in df_split[gold_code_col].fillna("")]
        gold_codes = {c for cs in code_lists_split for c in cs}
        oov = gold_codes - label_set
        oov_audit[split_name] = {
            "n_unique_gold": len(gold_codes),
            "n_oov": len(oov),
            "pct_oov": round(100.0 * len(oov) / max(1, len(gold_codes)), 2),
        }
        LOGGER.info(
            "M2 OOV audit on %s: %d/%d (%s%%) unique gold codes are outside "
            "the trained label vocab — recall on those is structurally 0",
            split_name, len(oov), len(gold_codes),
            oov_audit[split_name]["pct_oov"],
        )

    with open(model_out / "labels.json", "w") as f:
        json.dump(
            {
                "labels": labels,
                "config": {
                    "base_model": base_model,
                    "max_length": max_length,
                    "min_code_freq": min_code_freq,
                    "fp16": fp16_eff,
                    "bf16": bf16_eff,
                    "gradient_checkpointing": gradient_checkpointing,
                    "head": head,
                },
                "oov_audit": oov_audit,
            },
            f,
            indent=2,
        )
    LOGGER.info("Saved M2 model -> %s (n_labels=%d)", model_out, len(labels))
    return {"n_labels": len(labels), "n_train_docs": len(df_train),
            "oov_audit": oov_audit}


# ---------------------------------------------------------------------------
# Predict / dump-scores
# ---------------------------------------------------------------------------

def _load_model(model_dir: Path):
    import torch
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    model_dir = Path(model_dir)
    with open(model_dir / "labels.json") as f:
        meta = json.load(f)
    labels: List[str] = meta["labels"]
    cfg = meta.get("config", {})
    max_length = int(cfg.get("max_length", 4096))
    head = str(cfg.get("head", "cls"))  # absent in older models -> old path

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if head == "label-attention":
        encoder = AutoModel.from_pretrained(model_dir)
        model = _build_label_attention_model(encoder, n_labels=len(labels))
        model.head.load_state_dict(
            torch.load(model_dir / "label_attention_head.pt", map_location="cpu")
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, tokenizer, labels, max_length, device


def _score_batches(model, tokenizer, texts, max_length, device, batch_size):
    import torch

    sigmoid = torch.nn.Sigmoid()
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            all_probs.append(probs)
    return np.vstack(all_probs) if all_probs else np.zeros((0, 0), dtype=np.float32)


def predict_longformer(
    notes_csv: Path,
    model_dir: Path,
    out_csv: Path,
    threshold: float = 0.3,
    calibration_json: Optional[Path] = None,
    batch_size: int = 4,
    seed: int = 42,
    limit: Optional[int] = None,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
    mode: str = "threshold",
    top_k: int = 2,
    top_k_min_score: float = 0.0,
) -> int:
    model, tokenizer, labels, max_length, device = _load_model(model_dir)
    LOGGER.info("Loaded M2 model: %d labels, device=%s, max_len=%d",
                len(labels), device, max_length)

    per_code_thr: Dict[str, float] = {}
    default_thr = float(threshold)
    if calibration_json is not None:
        with open(calibration_json) as f:
            cal = json.load(f)
        per_code_thr = {str(k).upper(): float(v) for k, v in cal.get("per_code", {}).items()}
        default_thr = float(cal.get("default_threshold", default_thr))
        LOGGER.info("Calibration: %d per-code thresholds, default=%.4f",
                    len(per_code_thr), default_thr)
    thresholds_vec = np.array(
        [per_code_thr.get(c, default_thr) for c in labels], dtype=np.float32
    )

    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)
    probs = _score_batches(
        model, tokenizer, df["note_text"].tolist(),
        max_length=max_length, device=device, batch_size=batch_size,
    )

    if mode == "top-k":
        LOGGER.info("Decoding mode: top-k (k=%d, min_score=%.4f)",
                    top_k, top_k_min_score)
    else:
        LOGGER.info("Decoding mode: threshold (per-code or default=%.4f)",
                    default_thr)

    predictions: List[Prediction] = []
    note_ids = df["note_id"].astype(str).tolist()
    for note_id, scores in zip(note_ids, probs):
        if mode == "top-k":
            order = np.argsort(-scores)[:top_k]
            order = order[scores[order] >= top_k_min_score]
            idxs = order
        else:
            keep = scores >= thresholds_vec
            idxs = np.where(keep)[0]
            idxs = idxs[np.argsort(-scores[idxs])]
        kept_codes = [labels[i] for i in idxs]
        kept_scores = [float(scores[i]) for i in idxs]
        predictions.append(
            Prediction(note_id=note_id, codes=kept_codes, scores=kept_scores)
        )

    write_predictions(predictions, out_csv, include_scores=True)
    log_prediction_stats(predictions, label="M2",
                         out_path=stats_sidecar(out_csv))
    return len(predictions)


def fit_thresholds(
    scores_npz: Path,
    out_json: Path,
    grid: Optional[List[float]] = None,
    min_pos_for_per_code: int = 5,
    default_threshold: float = 0.5,
) -> None:
    z = np.load(scores_npz, allow_pickle=True)
    if "gold" not in z.files:
        raise ValueError(f"{scores_npz} has no 'gold' array (need val "
                         f"NPZ with proc_codes column).")
    probs = z["scores"]
    gold = z["gold"].astype(bool)
    labels = [str(c) for c in z["labels"]]

    if grid is None:
        grid = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25,
                0.275, 0.3, 0.325, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                0.65, 0.7, 0.75, 0.8]
    grid_arr = np.array(grid, dtype=np.float32)

    pos_per_code = gold.sum(axis=0)
    n_codes = probs.shape[1]

    per_code_thr: Dict[str, float] = {}
    n_fit = 0
    n_default = 0

    # Vectorized: for each code, sweep grid.
    for j in range(n_codes):
        pj = pos_per_code[j]
        if pj < min_pos_for_per_code:
            n_default += 1
            continue
        s_j = probs[:, j]
        g_j = gold[:, j]
        best_f1 = -1.0
        best_t = float(default_threshold)
        for t in grid_arr:
            preds = s_j >= t
            tp = float((preds & g_j).sum())
            n_pred = float(preds.sum())
            if n_pred == 0:
                continue
            p = tp / n_pred
            r = tp / float(pj)
            f1 = 2.0 * p * r / max(p + r, 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        if best_f1 > 0:
            per_code_thr[labels[j]] = best_t
            n_fit += 1
        else:
            n_default += 1

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {"per_code": per_code_thr,
             "default_threshold": float(default_threshold),
             "min_pos_for_per_code": int(min_pos_for_per_code)},
            f, indent=2,
        )
    LOGGER.info("Fit thresholds: %d per-code, %d using default=%.4f -> %s",
                n_fit, n_default, default_threshold, out_json)

    # Report what F1 this calibration would yield on the val NPZ.
    thr_vec = np.array(
        [per_code_thr.get(labels[j], default_threshold)
         for j in range(n_codes)], dtype=np.float32
    )
    preds = probs >= thr_vec
    tp = float((preds & gold).sum())
    n_pred = float(preds.sum())
    n_gold = float(gold.sum())
    p = tp / max(n_pred, 1.0)
    r = tp / max(n_gold, 1.0)
    f1 = 2.0 * p * r / max(p + r, 1e-9)
    print(f"per-label calibration on val: P={p:.4f}  R={r:.4f}  "
          f"F1={f1:.4f}  preds={int(n_pred)}")


def _print_score_diagnostics(
    probs: np.ndarray, gold: Optional[np.ndarray]
) -> None:
    n_notes, n_labels = probs.shape
    print(f"shape: notes={n_notes}, labels={n_labels}")
    print(
        f"score range: min={probs.min():.6f}, max={probs.max():.6f}, "
        f"mean={probs.mean():.6f}"
    )
    per_note_max = probs.max(axis=1)
    print(
        f"per-note max: mean={per_note_max.mean():.4f}, "
        f"median={np.median(per_note_max):.4f}"
    )
    print()

    if gold is None:
        print("(no gold column found -- skipping percentile/threshold "
              "diagnostics)")
        return

    gold_bool = gold.astype(bool)
    gold_scores = probs[gold_bool]
    nong_scores = probs[~gold_bool]

    print(f"GOLD cells (n={gold_scores.size}):")
    if gold_scores.size > 0:
        gq = np.percentile(gold_scores, [10, 25, 50, 75, 90, 99])
        print(
            f"  p10 = {gq[0]:.6f}   p25 = {gq[1]:.6f}   p50 = {gq[2]:.6f}   "
            f"p75 = {gq[3]:.6f}   p90 = {gq[4]:.6f}   p99 = {gq[5]:.6f}"
        )
    else:
        print("  (none)")

    print(f"NON-GOLD cells (n={nong_scores.size}):")
    if nong_scores.size > 0:
        nq = np.percentile(nong_scores, [50, 90, 99, 99.9])
        print(
            f"  p  50 = {nq[0]:.6f}   p  90 = {nq[1]:.6f}   "
            f"p  99 = {nq[2]:.6f}   p99.9 = {nq[3]:.6f}"
        )
    else:
        print("  (none)")

    print()
    print("Threshold sweep:")
    print(f"{'t':>8}{'preds':>11}{'P':>9}{'R':>9}{'F1':>9}")
    thresholds = (0.001, 0.003, 0.005, 0.01, 0.02, 0.03,
                  0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
                  0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8)
    n_gold = float(gold.sum())
    best_f1 = 0.0
    best_t = 0.0
    for t in thresholds:
        preds = (probs >= t)
        n_preds = int(preds.sum())
        if n_preds == 0:
            print(f"{t:>8.4f}  no preds")
            continue
        tp = float((preds & gold_bool).sum())
        p = tp / max(n_preds, 1)
        r = tp / max(n_gold, 1.0)
        f1 = 2.0 * p * r / max(p + r, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
        print(f"{t:>8.4f}{n_preds:>11d}{p:>9.4f}{r:>9.4f}{f1:>9.4f}")

    print()
    print(f"BEST: t={best_t:.4f}  micro-F1={best_f1:.6f}")


def dump_scores(
    notes_csv: Path,
    model_dir: Path,
    out_npz: Path,
    gold_code_col: str = "proc_codes",
    batch_size: int = 4,
    seed: int = 42,
    limit: Optional[int] = None,
    note_id_col: Optional[str] = None,
    note_text_col: Optional[str] = None,
) -> None:
    model, tokenizer, labels, max_length, device = _load_model(model_dir)
    label_to_idx = {c: i for i, c in enumerate(labels)}

    df = load_notes_for_prediction(
        notes_csv, note_id_col=note_id_col, note_text_col=note_text_col
    )
    df = apply_seed_and_limit(df, seed=seed, limit=limit)
    probs = _score_batches(
        model, tokenizer, df["note_text"].tolist(),
        max_length=max_length, device=device, batch_size=batch_size,
    )

    save_kwargs = {
        "note_ids": df["note_id"].astype(str).to_numpy(),
        "labels": np.array(labels, dtype=object),
        "scores": probs,
    }
    gold: Optional[np.ndarray] = None
    if gold_code_col in df.columns:
        code_lists = [_pipe_split(s) for s in df[gold_code_col].fillna("")]
        gold = _build_label_matrix(code_lists, label_to_idx).astype(np.uint8)
        save_kwargs["gold"] = gold

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **save_kwargs)
    LOGGER.info("Wrote scores NPZ (%s) -> %s", probs.shape, out_npz)

    _print_score_diagnostics(probs, gold)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_train_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("train",
                      help="Fine-tune Clinical-Longformer for multi-label.")
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--val", type=Path, default=None)
    p.add_argument("--test", type=Path, default=None,
                   help="Optional test CSV for OOV-audit logging.")
    p.add_argument("--model-out", required=True, type=Path)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--min-code-freq", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--bf16", dest="bf16", action="store_true",
                   help="Use bf16 (default on H100/Ampere+).")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.set_defaults(bf16=True)
    p.add_argument("--fp16", dest="fp16", action="store_true",
                   help="Use fp16 (only when bf16 is unavailable).")
    p.set_defaults(fp16=False)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable HF gradient checkpointing on the model "
                        "to halve activation memory at L=4096.")
    p.add_argument("--save-strategy", default="epoch",
                   choices=["no", "epoch", "steps"])
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--resume-from-checkpoint", type=Path, default=None,
                   help="Resume training from this checkpoint dir.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal-loss gamma applied on top of pos_weight'd "
                        "BCE.  0 = plain weighted BCE (the prior behavior).")
    p.add_argument("--head", choices=["cls", "label-attention"],
                   default="cls",
                   help="cls = stock sequence-classification head (the prior "
                        "behavior).  label-attention = the PLM-ICD-style "
                        "per-label attention over all token states that M2 "
                        "reports.  Recorded in labels.json; predict/"
                        "dump-scores pick it up automatically.")
    p.add_argument("--local-models-dir", type=Path, default=None,
                   help="Directory holding offline copies of the HF "
                        "models (e.g. ./models/).  Falls back to "
                        "$CPT_REC_LOCAL_MODELS_DIR or ./models/.  Set "
                        "CPT_REC_REQUIRE_LOCAL_MODELS=1 to fail when no "
                        "local copy is found.")


def _build_predict_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("predict",
                      help="Predict with a fine-tuned M2 model.")
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--mode", choices=["threshold", "top-k"],
                   default="threshold",
                   help="threshold = global/per-code thresholds; "
                        "top-k = always pick top-K codes per note.")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--calibration", type=Path, default=None,
                   help="JSON with per-code thresholds (mode=threshold).")
    p.add_argument("--top-k", type=int, default=2,
                   help="K for mode=top-k.")
    p.add_argument("--top-k-min-score", type=float, default=0.0,
                   help="Drop top-k picks below this score floor.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)


def _build_fit_thresholds_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("fit-thresholds",
                      help="Fit per-label thresholds from a val scores NPZ.")
    p.add_argument("--scores-npz", required=True, type=Path,
                   help="NPZ from `dump-scores` on val (must include gold).")
    p.add_argument("--out", required=True, type=Path,
                   help="JSON file to write (compatible with --calibration).")
    p.add_argument("--default-threshold", type=float, default=0.5,
                   help="Threshold for codes with too few positives.")
    p.add_argument("--min-pos", type=int, default=5,
                   help="Minimum positives in val to fit a per-code "
                        "threshold; below this, use --default-threshold.")


def _build_dump_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("dump-scores",
                      help="Score every (note,code) cell to an NPZ.")
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out-npz", required=True, type=Path)
    p.add_argument("--gold-code-col", default="proc_codes")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--note-id-col", default=None)
    p.add_argument("--note-text-col", default=None)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="M2: Clinical-Longformer multi-label baseline."
    )
    sp = parser.add_subparsers(dest="cmd", required=True)
    _build_train_parser(sp)
    _build_predict_parser(sp)
    _build_dump_parser(sp)
    _build_fit_thresholds_parser(sp)
    args = parser.parse_args()

    if args.cmd == "train":
        train_longformer(
            train_csv=args.train,
            val_csv=args.val,
            test_csv=args.test,
            model_out=args.model_out,
            base_model=args.base_model,
            max_length=args.max_length,
            min_code_freq=args.min_code_freq,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            save_strategy=args.save_strategy,
            save_total_limit=args.save_total_limit,
            resume_from_checkpoint=args.resume_from_checkpoint,
            seed=args.seed,
            gold_code_col=args.gold_code_col,
            local_models_dir=args.local_models_dir,
            focal_gamma=args.focal_gamma,
            head=args.head,
        )
    elif args.cmd == "predict":
        predict_longformer(
            notes_csv=args.notes,
            model_dir=args.model,
            out_csv=args.out,
            threshold=args.threshold,
            calibration_json=args.calibration,
            batch_size=args.batch_size,
            seed=args.seed,
            limit=args.limit,
            note_id_col=args.note_id_col,
            note_text_col=args.note_text_col,
            mode=args.mode,
            top_k=args.top_k,
            top_k_min_score=args.top_k_min_score,
        )
    elif args.cmd == "dump-scores":
        dump_scores(
            notes_csv=args.notes,
            model_dir=args.model,
            out_npz=args.out_npz,
            gold_code_col=args.gold_code_col,
            batch_size=args.batch_size,
            seed=args.seed,
            limit=args.limit,
            note_id_col=args.note_id_col,
            note_text_col=args.note_text_col,
        )
    elif args.cmd == "fit-thresholds":
        fit_thresholds(
            scores_npz=args.scores_npz,
            out_json=args.out,
            min_pos_for_per_code=args.min_pos,
            default_threshold=args.default_threshold,
        )


if __name__ == "__main__":
    main()
