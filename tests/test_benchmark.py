#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Tests for the two-corpus benchmark harness (T3).

Everything here runs on tiny synthetic fixtures — no GPU, no MIMIC data, no
network.  The point is to prove three properties that the paper's MIMIC half
depends on:

1. ``build_mimic`` filters to surgery encounters, keeps one row per encounter,
   and never lets a patient cross a split boundary;
2. the label-completeness contract actually deletes precision-family cells on a
   partially recorded corpus and leaves them alone on a complete one;
3. the leak guard drops row-level keys, long strings and long lists, and the
   exporter refuses a destination that ``.gitignore`` would un-ignore.

Run from project root::

    PYTHONPATH=src python3 tests/test_benchmark.py
    PYTHONPATH=src pytest tests/test_benchmark.py -q
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import logging
import tempfile
from pathlib import Path

import pandas as pd

from cpt_rec.benchmark import build_mimic, collate, export
from cpt_rec.benchmark.corpora import MIMIC, VUMC, get_corpus

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_mimic_fixture(root: Path) -> tuple[Path, Path]:
    """Six encounters over six patients.

    ``h3`` is observation-only (no surgical code) so the *encounter* filter drops
    it.  ``h2`` is the code-filter case: a real operation billed alongside an
    observation stay and a cardiac catheterisation, so the three
    ``--code-filter`` modes each give it a different gold set.
    """
    hcpcs = root / "hcpcsevents.csv"
    with hcpcs.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject_id", "hadm_id", "hcpcs_cd", "seq_num"])
        w.writerow(["p1", "h1", "43239", "1"])       # surgery
        w.writerow(["p1", "h1", "43235", "2"])       # surgery, same encounter
        w.writerow(["p2", "h2", "47562", "1"])       # surgery
        w.writerow(["p2", "h2", "99219", "2"])       # E/M observation -> code filter
        w.writerow(["p2", "h2", "G0378", "3"])       # observation per hour -> code filter
        w.writerow(["p2", "h2", "93454", "4"])       # cardiac cath: a procedure, not surgery
        w.writerow(["p3", "h3", "99219", "1"])       # observation only -> dropped
        w.writerow(["p4", "h4", "64415", "1"])       # surgery, but a SHORT note
        w.writerow(["p5", "h5", "31624", "1"])       # surgery
        w.writerow(["p6", "h6", "45380", "1"])       # surgery
    long_note = " ".join(["word"] * 40)
    discharge = root / "discharge.csv.gz"
    with gzip.open(discharge, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["note_id", "subject_id", "hadm_id", "note_type", "text"])
        w.writerow(["n1b", "p1", "h1", "DS", long_note + " addendum"])
        w.writerow(["n1a", "p1", "h1", "DS", long_note])       # duplicate encounter
        w.writerow(["n2", "p2", "h2", "DS", long_note])
        w.writerow(["n3", "p3", "h3", "DS", long_note])        # non-surgical
        w.writerow(["n4", "p4", "h4", "DS", "too short"])      # under the token floor
        w.writerow(["n5", "p5", "h5", "DS", long_note])
        w.writerow(["n6", "p6", "h6", "DS", long_note])
    return hcpcs, discharge


def _run_dir(root: Path, corpus: str, system: str, split: str,
             micro_f1: float, r5: float, r10: float, fam: float) -> Path:
    """Write an exported run the way ``cptrec-bench-export`` would have written it."""
    d = root / corpus / system / split
    d.mkdir(parents=True, exist_ok=True)
    c = get_corpus(corpus)
    raw = {
        "metrics.json": ("metrics", {
            "set": {"micro_f1": micro_f1, "micro_recall": r5, "micro_precision": 0.4,
                    "n_pred_codes": 100, "n_notes": 50},
        }),
        "rank_metrics.json": ("rank", {
            "recall_at": {"5": r5, "10": r10},
            "coverage_at": {"5": r5},
            "precision_at": {"5": 0.3},
            "family": {"family_mrr": fam},
            "shown_at": {"5": 4.5},
            "pool_ceiling": 0.9,
        }),
    }
    files = []
    for fname, (kind, payload) in raw.items():
        clean, dropped, caveats = export.scrub(payload, c, kind)
        (d / fname).write_text(json.dumps(clean), encoding="utf-8")
        files.append({"dropped": dropped, "caveats": caveats})
    (d / "export_summary.json").write_text(json.dumps({
        "corpus": corpus, "system": system, "split": split, "files": files,
    }), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# 1. corpus build
# ---------------------------------------------------------------------------

def test_build_mimic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hcpcs, discharge = _write_mimic_fixture(root)
        out = root / "derived"
        manifest = build_mimic.build(
            hcpcs, discharge, out,
            min_note_tokens=10, val_frac=0.2, test_frac=0.2, chunksize=3,
        )

        # non-surgical encounter never enters the corpus
        assert manifest["codes"]["surgery_encounters"] == 5, manifest["codes"]
        # the short note is dropped by the token floor
        assert manifest["join"]["dropped_short_notes"] == 1, manifest["join"]
        assert manifest["join"]["encounters_kept"] == 4, manifest["join"]

        frames = {}
        subjects = {}
        for name in ("train", "val", "test"):
            df = pd.read_csv(out / f"{name}.csv", dtype=str)
            frames[name] = df
            subjects[name] = set(df.subject_id) if len(df) else set()
            assert list(df.columns) == list(build_mimic.OUT_COLS)

        # patient-disjoint: no subject appears in two splits
        for a in ("train", "val", "test"):
            for b in ("train", "val", "test"):
                if a < b:
                    assert not (subjects[a] & subjects[b]), (a, b, subjects)

        allrows = pd.concat(frames.values())
        assert len(allrows) == 4
        # one row per encounter, deterministic pick = first note_id
        h1 = allrows[allrows.hadm_id == "h1"]
        assert len(h1) == 1 and h1.iloc[0].note_id == "n1a", h1.to_dict("records")
        # codes pipe-joined in sorted order
        assert h1.iloc[0].proc_codes == "43235|43239"
        # p3's observation-only encounter is gone
        assert "h3" not in set(allrows.hadm_id)
        # the default code filter strips the billing noise off a real operation
        h2 = allrows[allrows.hadm_id == "h2"]
        assert h2.iloc[0].proc_codes == "47562|93454", h2.to_dict("records")

        # the completeness number is measured, not asserted
        lc = manifest["label_completeness"]
        assert lc["verdict"] == "partially-recorded"
        assert 0.0 < lc["covered_fraction"] <= 1.0
        assert (out / "code_frequency_stats.csv").exists()
        assert (out / "corpus_stats.json").exists()
        # the corpus card holds counts only
        card = json.loads((out / "corpus_stats.json").read_text())
        assert "note_text" not in json.dumps(card)
    print("PASS test_build_mimic")


def test_build_mimic_code_filter() -> None:
    """The gold set of a surgery encounter depends on ``--code-filter``.

    ``h2`` is one operation (47562) billed with an observation stay (99219,
    G0378) and a cardiac catheterisation (93454).  Only the operation and the
    catheterisation are inferable from a procedure narrative, so the default
    keeps those two and drops the level-of-service pair.
    """
    expected = {
        "procedural": "47562|93454",           # default: procedures only
        "surgery": "47562",                    # strict: CPT-I surgery section only
        "none": "47562|93454|99219|G0378",     # legacy: every charge on the admission
    }
    share: dict[str, float] = {}
    for mode, gold in expected.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hcpcs, discharge = _write_mimic_fixture(root)
            out = root / "derived"
            manifest = build_mimic.build(
                hcpcs, discharge, out, code_filter=mode,
                min_note_tokens=10, val_frac=0.2, test_frac=0.2, chunksize=3,
            )
            allrows = pd.concat(
                pd.read_csv(out / f"{n}.csv", dtype=str) for n in ("train", "val", "test")
            )
            h2 = allrows[allrows.hadm_id == "h2"]
            assert h2.iloc[0].proc_codes == gold, (mode, h2.to_dict("records"))

            cs = manifest["codes"]
            assert cs["code_filter"] == mode
            assert manifest["recipe"]["code_filter"] == mode
            # the encounter rule is untouched by the code rule
            assert cs["surgery_encounters"] == 5, (mode, cs)
            assert cs["encounters_emptied_by_filter"] == 0, (mode, cs)
            # every dropped instance is accounted for, by code, in the manifest
            assert (cs["code_instances_before_filter"] - cs["code_instances_kept"]
                    == cs["dropped_code_instances"] == sum(cs["dropped_codes"].values())), cs
            if mode == "none":
                assert cs["dropped_codes"] == {}
            else:
                # the observation pair is dropped by name, and no surgical code ever is
                assert {"99219", "G0378"} <= set(cs["dropped_codes"]), cs["dropped_codes"]
                assert "47562" not in cs["dropped_codes"], cs["dropped_codes"]
            share[mode] = cs["surgical_share_kept"]

    # the whole point of the filter: it raises the surgical share of the gold
    assert share["surgery"] == 1.0, share
    assert share["none"] < share["procedural"] < share["surgery"], share
    print("PASS test_build_mimic_code_filter")


def test_build_mimic_dry_run_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hcpcs, discharge = _write_mimic_fixture(root)
        out = root / "derived"
        build_mimic.build(hcpcs, discharge, out, min_note_tokens=10,
                          chunksize=3, limit_chunks=1, write=False)
        assert not (out / "train.csv").exists()
    print("PASS test_build_mimic_dry_run_writes_nothing")


# ---------------------------------------------------------------------------
# 2. label-completeness contract
# ---------------------------------------------------------------------------

def test_completeness_contract() -> None:
    payload = {
        "set": {"micro_f1": 0.5, "micro_recall": 0.6, "micro_precision": 0.4,
                "n_tp": 10, "n_fp": 5, "n_pred_codes": 15},
        "per_bin": {"head": {"micro_f1": 0.7, "micro_recall": 0.8, "n_fp": 3}},
        "constraints": {"n_notes": 50, "n_notes_mue": 2},
    }
    clean, dropped, caveats = export.scrub(payload, MIMIC, "metrics")
    assert "micro_f1" not in clean["set"], clean["set"]
    assert "micro_precision" not in clean["set"]
    assert "n_fp" not in clean["set"]
    assert clean["set"]["micro_recall"] == 0.6
    assert clean["set"]["n_pred_codes"] == 15          # prediction-only, always safe
    assert "micro_f1" not in clean["per_bin"]["head"]
    assert clean["constraints"] == payload["constraints"]   # validity is unaffected
    paths = {d["path"] for d in dropped}
    assert {"set.micro_f1", "set.micro_precision", "set.n_fp",
            "per_bin.head.micro_f1"} <= paths, paths
    assert any(c["path"] == "set.micro_recall" and "LOWER BOUND" in c["caveat"]
               for c in caveats), caveats

    # the same payload on a reference-complete corpus is untouched
    clean_v, dropped_v, caveats_v = export.scrub(payload, VUMC, "metrics")
    assert clean_v == payload and not dropped_v and not caveats_v

    # rank family
    rank = {"recall_at": {"5": 0.6}, "precision_at": {"5": 0.3},
            "map_at": {"5": 0.4}, "shown_at": {"5": 4.4},
            "family": {"family_mrr": 0.8}}
    clean_r, dropped_r, _ = export.scrub(rank, MIMIC, "rank")
    assert "precision_at" not in clean_r and "map_at" not in clean_r
    assert clean_r["recall_at"] == {"5": 0.6} and clean_r["shown_at"] == {"5": 4.4}
    assert {"precision_at", "map_at"} <= {d["path"] for d in dropped_r}
    print("PASS test_completeness_contract")


def test_no_precision_derived_cell_survives_partial_gold() -> None:
    """Assert the CONTRACT, not the suppression table against itself.

    test_completeness_contract above enumerates the same key names the
    suppression table does, so it can only ever confirm that the table
    contains what the table contains.  It passed while the 2026-08-27 export
    shipped **83** precision-derived cells for MIMIC:

      * ``set.example_f1`` and ``set.per_category.*.{exact_match,example_f1,
        jaccard_mean}`` -- no matching key name;
      * ``bootstrap.macro_f1.*`` / ``bootstrap.example_f1.*`` -- the pattern
        was ``bootstrap.micro_f1*``;
      * ``ranking.precision_at.*`` / ``ranking.auc_pr`` -- metrics.json nests
        the ranking contract under ``ranking``, and the "rank" patterns are
        rooted at the top level of rank_metrics.json, so they never fired;
      * ``micro.precision`` / ``micro.f1`` in sibling_report.json -- the
        "sibling" suppression table was empty.

    This test walks the scrubbed payload and fails on any surviving key that
    carries a precision-derived token at any depth, so a metric added later is
    caught by shape rather than by someone remembering to extend a list.
    """
    # Token-BOUNDARY matching, not substring.  A substring list is what let the
    # 2026-08-28 re-audit miss six fields per system: "n_fp" happens to appear
    # inside "substitutio(n_fp_)rate" but not inside "sibling_fp_rate" or
    # "confusion.fp", so the list flagged the wrong three and cleared the rest.
    banned_tokens = {"fp", "precision", "f1", "jaccard", "ndcg"}
    banned_phrases = ("exact_match", "auc_pr", "map_at")

    def is_banned(path):
        toks = set(re.split(r"[^a-z0-9]+", path.lower()))
        if toks & banned_tokens:
            return True
        return any(ph in path.lower() for ph in banned_phrases)

    # The only FP-derived quantities allowed through: the T2-5 read-out rates,
    # which ship with an explicit BIASED UPWARD caveat instead of being dropped.
    caveated_ok = ("sibling_fp_rate", "substitution_fp_rate")

    def paths(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from paths(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from paths(v, f"{path}[{i}]")
        else:
            yield path

    # every shape that actually leaked, at the depth it leaked from
    payloads = {
        "metrics": {
            "set": {
                "micro_f1": 0.5, "macro_f1": 0.4, "example_f1": 0.23,
                "micro_precision": 0.4, "exact_match": 0.004,
                "jaccard_mean": 0.14, "n_fp": 5,
                "micro_recall": 0.6, "n_pred_codes": 15,
                "per_category": {"CPT_I": {"example_f1": 0.23,
                                           "exact_match": 0.004,
                                           "jaccard_mean": 0.14,
                                           "micro_recall": 0.6}},
            },
            "per_bin": {"head": {"micro_f1": 0.7, "micro_recall": 0.8, "n_fp": 3}},
            "bootstrap": {
                "micro_f1": {"mean": 0.5}, "macro_f1": {"mean": 0.11},
                "example_f1": {"ci_lo": 0.41, "ci_hi": 0.44},
                "exact_match": {"mean": 0.0}, "jaccard_mean": {"mean": 0.14},
            },
            "ranking": {
                "recall_at": {"5": 0.61}, "precision_at": {"10": 0.123},
                "auc_pr": 0.5047, "map_at": {"5": 0.4}, "ndcg_at": {"5": 0.5},
                "shown_at": {"5": 4.85}, "pool_ceiling": 0.7867,
                "family": {"family_mrr": 0.7634},
            },
            "constraints": {"n_notes": 50},
        },
        "rank": {"recall_at": {"5": 0.6}, "precision_at": {"5": 0.3},
                 "map_at": {"5": 0.4}, "auc_pr": 0.5, "shown_at": {"5": 4.4}},
        # the REAL exported shape, keys copied from
        # outputs/benchmark/top5/mimic/*/test/sibling_report.json (values dummy).
        # Writing this payload from the suppression table instead of from the
        # artifact is exactly how the first pass missed the FP-composition block.
        "sibling": {
            "micro": {"f1": 0.4251, "precision": 0.3889, "recall": 0.4687},
            "confusion": {"tp": 1720, "fp": 2275, "fn": 2978},
            "headline": {
                "n_fn_nearmiss": 589,
                "n_fp_sibling": 1463,
                "n_fp_substitution": 742,
                "nearmiss_fn_rate": 0.1978,
                "sibling_fp_rate": 0.6431,
                "substitution_fp_rate": 0.3262,
            },
            "by_train_bin": {
                "head": {"n_fn": 2018, "nearmiss_fn_rate": 0.2091,
                         "sibling_fp_rate": 0.6598},
                "unseen": {"n_fn": 204, "nearmiss_fn_rate": 0.1275,
                           "sibling_fp_rate": 0.2167},
            },
            "primary_addon": {
                "addon_fp_share": 0.11,
                "addon_fp_without_primary_share": 0.04,
                "n_addon_fp_without_primary": 91,
            },
            "gate_verdict": "FAMILY-AWARE MACHINERY IS WELL-TARGETED",
            "n_gold_notes": 100,
        },
    }

    for kind, payload in payloads.items():
        clean, dropped, _ = export.scrub(payload, MIMIC, kind)
        survivors = [p for p in paths(clean)
                     if is_banned(p) and not p.endswith(caveated_ok)]
        assert not survivors, f"{kind}: precision-derived cells survived: {survivors}"
        assert dropped, f"{kind}: nothing was dropped at all"

    # ...and the quantities the benchmark tables are built on must SURVIVE,
    # or the fix has traded a leak for blank MIMIC rows
    clean, _, caveats = export.scrub(payloads["metrics"], MIMIC, "metrics")
    assert clean["set"]["micro_recall"] == 0.6
    assert clean["set"]["n_pred_codes"] == 15
    assert clean["ranking"]["recall_at"] == {"5": 0.61}
    assert clean["ranking"]["shown_at"] == {"5": 4.85}
    assert clean["ranking"]["pool_ceiling"] == 0.7867
    assert clean["ranking"]["family"]["family_mrr"] == 0.7634
    assert clean["constraints"] == {"n_notes": 50}
    # the nested ranking block must now also pick up its LOWER BOUND caveats,
    # which it never did while the rank table went unconsulted for metrics.json
    assert any(c["path"].startswith("ranking.recall_at") and "LOWER BOUND" in c["caveat"]
               for c in caveats), caveats

    # the sibling FP-composition block must be GONE, and the two rates that are
    # deliberately kept must each carry their BIASED UPWARD caveat -- an
    # uncaveated rate is the same leak wearing a different name
    clean_s, dropped_s, caveats_s = export.scrub(payloads["sibling"], MIMIC, "sibling")
    assert "fp" not in clean_s["confusion"], clean_s["confusion"]
    assert clean_s["confusion"]["tp"] == 1720 and clean_s["confusion"]["fn"] == 2978
    assert "primary_addon" not in clean_s or not clean_s["primary_addon"], clean_s
    assert "gate_verdict" not in clean_s, clean_s
    assert "n_fp_sibling" not in clean_s["headline"], clean_s["headline"]
    assert "n_fp_substitution" not in clean_s["headline"], clean_s["headline"]
    assert clean_s["headline"]["n_fn_nearmiss"] == 589        # FN side is real
    assert clean_s["headline"]["sibling_fp_rate"] == 0.6431   # kept, caveated
    caveated = {c["path"] for c in caveats_s}
    for path in ("headline.sibling_fp_rate", "headline.substitution_fp_rate",
                 "by_train_bin.head.sibling_fp_rate",
                 "by_train_bin.unseen.sibling_fp_rate"):
        assert path in caveated, f"{path} kept without a caveat: {sorted(caveated)}"

    # a reference-complete corpus is still untouched in every family
    for kind, payload in payloads.items():
        clean_v, dropped_v, caveats_v = export.scrub(payload, VUMC, kind)
        assert clean_v == payload and not dropped_v and not caveats_v, kind

    print("PASS test_no_precision_derived_cell_survives_partial_gold")


# ---------------------------------------------------------------------------
# 3. leak guard
# ---------------------------------------------------------------------------

def test_leak_guard() -> None:
    payload = {
        "set": {"micro_recall": 0.6},
        "note_id": "NOTE-123",
        "per_note": [{"note_id": "x"}],
        "blurb": "x" * 500,
        "multiline": "line one\nline two",
        "long_list": list(range(500)),
        "family": {"family_mrr": 0.8, "subject_id": "p1"},
    }
    clean, dropped, _ = export.scrub(payload, MIMIC, "rank", max_str=200, max_list=64)
    for banned in ("note_id", "per_note", "blurb", "multiline", "long_list"):
        assert banned not in clean, banned
    assert "subject_id" not in clean["family"]
    assert clean["family"]["family_mrr"] == 0.8
    reasons = {d["path"]: d["reason"] for d in dropped}
    assert "leak guard" in reasons["note_id"]
    assert "longer than 200 chars" in reasons["blurb"]
    assert "multi-line" in reasons["multiline"]
    assert "longer than 64 items" in reasons["long_list"]

    try:
        export._assert_safe_destination(Path("outputs/benchmark/top5/mimic/predictions/x.json"))
    except SystemExit as exc:
        assert "predictions" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a 'predictions' destination must be refused")
    print("PASS test_leak_guard")


def test_export_run_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "restricted"
        src.mkdir()
        (src / "metrics.json").write_text(json.dumps(
            {"set": {"micro_f1": 0.5, "micro_recall": 0.6, "n_notes": 10}}))
        (src / "rank_metrics.json").write_text(json.dumps(
            {"recall_at": {"5": 0.6}, "precision_at": {"5": 0.3}}))

        corpus = get_corpus("mimic")
        object.__setattr__(corpus, "export_root", root / "outputs" / "benchmark")
        summary = export.export_run(corpus, "m1_bm25_knn", "test", src, [])
        dest = corpus.export_dir("m1_bm25_knn", "test")

        m = json.loads((dest / "metrics.json").read_text())
        assert "micro_f1" not in m["set"] and m["set"]["micro_recall"] == 0.6
        r = json.loads((dest / "rank_metrics.json").read_text())
        assert "precision_at" not in r
        assert summary["n_dropped"] >= 2
        assert (dest / "export_summary.json").exists()

        # an empty/partial metrics dir must fail loudly, not export silence
        empty = root / "empty"
        empty.mkdir()
        try:
            export.export_run(corpus, "b9", "test", empty, [])
        except SystemExit as exc:
            assert "re-score" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("an empty metrics dir must raise")
        object.__setattr__(corpus, "export_root", Path("outputs/benchmark/top5"))
    print("PASS test_export_run_roundtrip")


# ---------------------------------------------------------------------------
# 4. collation
# ---------------------------------------------------------------------------

def test_collate_tables() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "benchmark"
        # same ordering on both corpora except the last two systems swap.
        # m4_rag_local stands in for an optional locally-served secondary:
        # it is outside PORTABLE_CORE, so it exercises the corpus-specific tag.
        _run_dir(root, "vumc", "m1_bm25_knn", "test", 0.48, 0.59, 0.68, 0.84)
        _run_dir(root, "vumc", "m4_rag_local", "test", 0.59, 0.48, 0.55, 0.80)
        _run_dir(root, "vumc", "m6_retrieve_verify", "test", 0.55, 0.67, 0.78, 0.82)
        _run_dir(root, "mimic", "m1_bm25_knn", "test", 0.30, 0.42, 0.51, 0.70)
        _run_dir(root, "mimic", "m4_rag_local", "test", 0.31, 0.38, 0.44, 0.66)
        _run_dir(root, "mimic", "m6_retrieve_verify", "test", 0.33, 0.52, 0.61, 0.72)

        found = collate.discover(root, "test")
        assert set(found) == {"vumc", "mimic"}
        # the contract holds through collation: no MIMIC micro-F1 cell
        assert "micro-F1" not in found["mimic"]["m1_bm25_knn"]["values"]
        assert "micro-F1" in found["mimic"]["m1_bm25_knn"]["suppressed"]
        assert found["vumc"]["m1_bm25_knn"]["values"]["micro-F1"] == 0.48

        main = collate.render_main(found, "test", collate.DEFAULT_FLOORS)
        assert "—<sup>1</sup>" in main and "**Footnotes**" in main

        transfer = collate.render_transfer(found, "test", collate.DEFAULT_FLOORS)
        assert "Ranking metric: **R@5**" in transfer      # micro-F1 unavailable on MIMIC
        assert "Spearman ρ = **1.0**" in transfer         # identical R@5 ordering
        # the headline must be the FLOOR-AWARE coefficient, with the raw
        # ordering demoted to a sensitivity line
        assert "pre-registered R@5 floor" in transfer
        assert "Report this only as a sensitivity" in transfer

        # core rows print as **M<n>** `key`; a secondary prints bare and tagged
        assert "**M6** `m6_retrieve_verify`" in main
        assert "`m4_rag_local` *(corpus-specific)*" in main

        rank_tbl = collate.render_ranking(found, "test")
        assert "R@5" in rank_tbl and "m6_retrieve_verify" in rank_tbl

        out = Path(tmp) / "tables"
        out.mkdir()
        collate.write_long_csv(found, "test", out / "benchmark_summary.csv")
        rows = list(csv.DictReader((out / "benchmark_summary.csv").open()))
        assert any(r["status"] == "suppressed" and r["corpus"] == "mimic" for r in rows)
    print("PASS test_collate_tables")


def test_ranks_with_floor() -> None:
    """The 0.21-point MIMIC gap the manuscript calls a tie.

    render_transfer used raw ranks, so benchmark_transfer.md published
    rho = 0.5 against the manuscript's pre-registered rho = 0.0.
    """
    floor = collate.DEFAULT_FLOORS["R@5"]                 # 0.0099
    mimic = [0.6109, 0.5756, 0.5777]                      # M1, M2, M6
    assert collate._ranks_with_floor(mimic, floor) == [1.0, 2.5, 2.5]
    # ...and with no floor the same values separate cleanly
    assert collate._ranks_with_floor(mimic, 0.0) == [1.0, 3.0, 2.0]

    # grouping is ANCHORED, not chained: a-b and b-c each clear the floor but
    # a-c does not, so a must not be tied to c
    chain = [0.60, 0.591, 0.582]
    assert collate._ranks_with_floor(chain, 0.0099) == [1.5, 1.5, 3.0]

    # half ranks must survive rendering rather than rounding to a clean win
    assert collate._rank_txt(2.5) == "2.5" and collate._rank_txt(2.0) == "2"
    assert collate._delta_txt(-1.5) == "-1.5" and collate._delta_txt(1.0) == "+1"
    print("PASS test_ranks_with_floor")


def test_rank_correlation_math() -> None:
    assert collate.spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert collate.spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert collate.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert collate.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    assert collate.spearman([1, 2], [2, 1]) is None      # too few systems
    print("PASS test_rank_correlation_math")


if __name__ == "__main__":
    # Discover, do not enumerate.  A hand-written call list silently skipped
    # test_no_precision_derived_cell_survives_partial_gold and
    # test_ranks_with_floor for a full day after they were added, which is the
    # same stale-list failure the suppression table itself had.
    _tests = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("test_") and callable(f)]
    for _name, _fn in _tests:
        _fn()
    print(f"\nall {len(_tests)} benchmark tests passed")
