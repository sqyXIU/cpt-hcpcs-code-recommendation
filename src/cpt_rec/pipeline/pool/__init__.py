# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Candidate-pool assembly — the retrieval half, shared by both scorers.

Everything here runs *before* a model sees anything: cut the note into
snippets, retrieve a compact per-note candidate code pool from the KB index
(BM25 + dense + the LLM concept prior), and attach the source-tagged
``φ(n, c)`` features that record where each candidate came from.

The pool is the benchmark's recall ceiling: no scorer can recover a code the
pool never proposed.
"""
