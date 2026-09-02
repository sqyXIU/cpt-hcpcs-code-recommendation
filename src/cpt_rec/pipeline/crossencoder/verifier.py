# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Candidate verifiers.

Two architectures share the same ``(note, candidate-code)`` pair-classification
contract — given the candidate's best evidence snippet, the candidate's code
description, and the φ(n, c) feature vector, emit one relevance logit:

* :class:`CrossEncoderVerifier` — the main lever.  A HuggingFace encoder
  jointly attends over ``[CLS] evidence [SEP] code_description [SEP]`` (true
  token-level cross attention, unlike the bi-encoder it replaces), then a
  small head fuses the pooled representation with φ.
* :class:`FeatureMLPVerifier` — a cheap "3a-1" stacker over φ alone (no
  transformer).  Useful as a fast sanity baseline that isolates how much of
  the lift comes from the source-tagged features vs. the cross encoder.

Both return raw logits of shape ``(B,)``; the train/predict drivers apply
``BCEWithLogitsLoss`` / ``sigmoid``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------

class CrossEncoderVerifier(nn.Module):
    """Joint ``(evidence, code_description)`` encoder + φ-fusion head."""

    def __init__(
        self,
        encoder_name: str,
        n_features: int,
        pooling: str = "cls",
        dropout: float = 0.1,
        head_hidden: Optional[int] = None,
        local_models_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        from cpt_rec.baselines.common import resolve_local_model

        resolved = resolve_local_model(encoder_name, models_dir=local_models_dir)
        self.encoder = AutoModel.from_pretrained(resolved)
        self.pooling = pooling
        self.n_features = int(n_features)
        self.hidden_size = int(self.encoder.config.hidden_size)

        # LayerNorm the (heterogeneous-scale) φ features before fusion so the
        # raw bm25 / cosine / rank columns enter the head on comparable scales.
        self.feat_norm = nn.LayerNorm(self.n_features) if self.n_features > 0 else None

        fused_dim = self.hidden_size + self.n_features
        hid = head_hidden or self.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return last_hidden_state[:, 0, :]  # CLS

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        if self.feat_norm is not None and features is not None:
            x = torch.cat([pooled, self.feat_norm(features)], dim=-1)
        else:
            x = pooled
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature-only MLP (no transformer)
# ---------------------------------------------------------------------------

class FeatureMLPVerifier(nn.Module):
    """MLP stacker over φ(n, c) alone — the cheap 3a-1 baseline."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.net = nn.Sequential(
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        **_ignored,
    ) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def build_verifier(
    arch: str,
    encoder_name: str,
    n_features: int,
    pooling: str = "cls",
    dropout: float = 0.1,
    feature_mlp_hidden: int = 128,
    local_models_dir: Optional[Path] = None,
) -> nn.Module:
    """Factory used by both the train and predict drivers."""
    if arch == "cross_encoder":
        return CrossEncoderVerifier(
            encoder_name=encoder_name,
            n_features=n_features,
            pooling=pooling,
            dropout=dropout,
            local_models_dir=local_models_dir,
        )
    if arch == "feature_mlp":
        return FeatureMLPVerifier(
            n_features=n_features, hidden=feature_mlp_hidden, dropout=dropout
        )
    raise ValueError(f"unknown verifier arch {arch!r}")
