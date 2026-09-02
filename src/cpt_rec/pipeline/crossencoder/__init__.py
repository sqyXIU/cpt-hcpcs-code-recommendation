# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""The cross-encoder verifier — M6's scoring stage.

One example per ``(note, candidate-code)`` pair, labelled ``code ∈ gold``:
a HuggingFace encoder attends jointly over ``[CLS] evidence [SEP] code text
[SEP]`` and a small head fuses the pooled vector with ``φ(n, c)``.  Trained
against hard sibling negatives drawn from the candidate's own code family.

This is what ``cptrec-verifier-train`` / ``cptrec-verifier-predict`` run: the
scorer inside M6, the retrieve-and-verify system, which writes under
``m6_retrieve_verify``.
"""
