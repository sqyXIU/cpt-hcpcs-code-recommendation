# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Two-corpus benchmark harness (T3).

The paper's pivot from *system* to *benchmark* needs three things the rest of
the repo does not provide:

1. a **corpus registry** so every method can be pointed at a second corpus
   without editing the method (:mod:`.corpora`),
2. a **label-completeness contract** that refuses to emit metric cells the
   corpus cannot support — MIMIC-IV's HCPCS table is partially recorded, so
   precision-family numbers there are not comparable across systems
   (:mod:`.corpora`, enforced by :mod:`.export`),
3. a **DUA-safe export path**: MIMIC-derived artifacts stay under the ignored
   ``outputs/datasets/mimic_iv/`` tree and only aggregate, note-free JSON is copied
   into the repo (:mod:`.export`), then collated into paper tables
   (:mod:`.collate`).
"""

from __future__ import annotations

__all__ = ["corpora", "build_mimic", "export", "collate"]
