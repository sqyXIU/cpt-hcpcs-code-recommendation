# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Compared systems M1--M5, named as the paper's Table III labels.

=====  ==============================  ========================  =========================================
label  module                          metrics-directory key     system
=====  ==============================  ========================  =========================================
M1     :mod:`m1_bm25_knn`              ``m1_bm25_knn``           BM25 neighbor voting
M2     :mod:`m2_longformer_label_attn` ``m2_label_attention``    Clinical-Longformer + PLM-ICD label attn
M3     :mod:`m3_zeroshot_llm`          ``m3_zeroshot_frontier``  zero-shot frontier LLM (JSON output)
M4     :mod:`m4_exemplar_rag`          ``m4_rag_frontier``       exemplar RAG: BM25 exemplars -> LLM
M5     :mod:`m5_lora_sft`              ``m5_sft_local``          LoRA SFT of a local open-weights model
=====  ==============================  ========================  =========================================

M6 -- the retrieve-and-verify system -- lives in :mod:`cpt_rec.pipeline` and
writes under ``m6_retrieve_verify``.

The key in the middle column is the on-disk contract: it names the metrics
directory a run writes and that ``cptrec-bench-export`` reads.  It is spelled
the same way the paper labels the row, so a directory listing and a table read
alike; :data:`cpt_rec.benchmark.corpora.SYSTEM_LABELS` maps key -> label and the
collated tables print both.  Nothing else in this package carries a system
number, so ``M<n>`` in a docstring always means the same row as ``M<n>`` in the
paper.

:mod:`kb_index` is not a compared system.  It builds the BM25 + dense indexes
over knowledge-base code descriptions that M6's candidate generation reads, and
also carries a retrieve-and-rerank CLI that no reported table uses.

:mod:`common` and :mod:`llm` hold the shared note loading, prediction-writing,
and LLM-backend plumbing that every system above goes through, so a new system
inherits the same input contract for free.
"""
