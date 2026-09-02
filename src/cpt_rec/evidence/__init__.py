# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Evidence ledger — one auditable record per recommendation.

Every code the pipeline emits carries the evidence window that produced it, so a
recommendation can be inspected rather than merely scored.  :mod:`ledger`
defines that record and is written by ``cptrec-verifier-predict --emit-ledger``;
:mod:`grounding_audit` samples a stratified slice of a ledger and asks an LLM
judge whether the retained window actually supports the code, which is the
evidence-grounding analysis reported in the paper.

The ledger is a data structure, not a control flow: the pipeline is a fixed
sequence of stages (sectionize, retrieve, verify, decode), not a set of agents
that decide what to do next.
"""
