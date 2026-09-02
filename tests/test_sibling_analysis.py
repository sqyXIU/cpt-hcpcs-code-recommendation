# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Unit tests for the sibling-error diagnostic."""

from __future__ import annotations

import json

import pandas as pd

from cpt_rec.common.evaluation import sibling_analysis as sa


def _write_kb(path):
    rows = [
        # code,  description,                         system, range1,        range1 desc
        ("29881", "Knee arthroscopy w/ meniscectomy", "CPT", "29800-29999", "Knee arthroscopy"),
        ("29882", "Knee arthroscopy w/ repair",       "CPT", "29800-29999", "Knee arthroscopy"),
        ("43235", "EGD diagnostic",                   "CPT", "43200-43499", "Esophagoscopy"),
        ("70000", "Radiologic exam",                  "CPT", "70000-79999", "Radiology"),
        ("99999", "Other service",                    "CPT", "99000-99999", "Other"),
    ]
    pd.DataFrame(
        rows,
        columns=["code", "code_description", "code_system",
                 "code_range_1", "code_range_1_description"],
    ).to_csv(path, index=False)


def test_sibling_analysis_headline(tmp_path):
    kb = tmp_path / "kb.csv"
    _write_kb(kb)

    gold = tmp_path / "gold.csv"
    pd.DataFrame(
        {"note_id": ["n1", "n2"], "proc_codes": ["29881|99999", "43235"]}
    ).to_csv(gold, index=False)

    pred = tmp_path / "pred.csv"
    pd.DataFrame(
        {
            "note_id": ["n1", "n2"],
            "pred_codes": ["29882|99999", "43235|70000"],
            "pred_scores": ["0.9|0.8", "0.7|0.6"],
        }
    ).to_csv(pred, index=False)

    out = tmp_path / "rep.json"
    r = sa.run(
        pred_csv=pred, gold_csv=gold, kb_csv=kb, out_json=out,
        train_stats=None, strata_csv=None, ncci_dir=None, top_families=25,
    )

    # confusion: n1 -> TP{99999} FP{29882} FN{29881}; n2 -> TP{43235} FP{70000}
    assert r["confusion"] == {"tp": 2, "fp": 2, "fn": 1}
    assert r["micro"]["precision"] == 0.5

    h = r["headline"]
    # 29882 shares 29881's family (gold) -> sibling FP; 70000 does not.
    assert h["sibling_fp_rate"] == 0.5
    # 29882 replaces the MISSED 29881 -> substitution.
    assert h["substitution_fp_rate"] == 0.5
    # the one missed gold (29881) had a predicted sibling (29882).
    assert h["nearmiss_fn_rate"] == 1.0

    fams = r["most_confused_families"]
    assert fams and fams[0]["family"] == "29800-29999"
    assert fams[0]["substitution_count"] == 1
    ex = fams[0]["examples"][0]
    assert ex["missed_gold"] == "29881"
    assert ex["predicted_siblings"] == ["29882"]

    assert "WELL-TARGETED" in r["gate_verdict"]
    assert out.exists()
    written = json.loads(out.read_text())
    assert written["headline"]["sibling_fp_rate"] == 0.5


def test_sibling_analysis_per_bin(tmp_path):
    kb = tmp_path / "kb.csv"
    _write_kb(kb)
    gold = tmp_path / "gold.csv"
    pd.DataFrame({"note_id": ["n1"], "proc_codes": ["29881"]}).to_csv(gold, index=False)
    pred = tmp_path / "pred.csv"
    pd.DataFrame({"note_id": ["n1"], "pred_codes": ["29882"]}).to_csv(pred, index=False)

    stats = tmp_path / "freq.csv"
    pd.DataFrame(
        {"code": ["29881", "29882"], "frequency": [10, 8],
         "rank": [1, 2], "bin": ["head", "torso"]}
    ).to_csv(stats, index=False)

    r = sa.run(
        pred_csv=pred, gold_csv=gold, kb_csv=kb, out_json=None,
        train_stats=stats, strata_csv=None, ncci_dir=None, top_families=-1,
    )
    by_bin = r["by_train_bin"]
    # FP 29882 is binned 'torso'; FN 29881 is binned 'head'.
    assert by_bin["torso"]["n_fp"] == 1
    assert by_bin["torso"]["sibling_fp_rate"] == 1.0
    assert by_bin["head"]["n_fn"] == 1
    assert by_bin["head"]["nearmiss_fn_rate"] == 1.0
