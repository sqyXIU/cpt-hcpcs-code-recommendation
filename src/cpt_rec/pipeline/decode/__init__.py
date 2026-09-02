# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Turning scores into a code set — the decision step, kept separate.

Scoring a candidate and *choosing* the emitted set are different problems, so
they are different modules: threshold calibration on val, NCCI-aware
constrained repair (validity, not F1 -- it cost ~1 F1 point where measured),
and set-size
decoders (top-k / expected-cardinality) as alternatives to a global
threshold.

All three read a prediction NPZ and are agnostic about which scorer wrote it.
"""
