# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
The retrieve-and-verify recommender.

Three subpackages, in the order a note passes through them:

.. code-block:: text

    note  ->  pool/          evidence windows, KB retrieval (lexical + dense),
                 |           similar-note transfer, and the phi(n, c) features
                 |           = the candidate pool, and the recall ceiling
                 v
              crossencoder/  pair verifier: P(code supported | window, code, phi)
                 |
                 v
              decode/        scores -> a ranked shortlist and an emitted set

The verifier is a ~110M-parameter cross-encoder; no LLM is called at note time.
Section headers are mapped offline by :mod:`cpt_rec.common.sectionizer`, whose
pattern config is induced once per corpus and then frozen.
"""

from cpt_rec.pipeline.pool.snippetize import (
    Snippet,
    snippets_for_note,
)

__all__ = ["Snippet", "snippets_for_note"]
