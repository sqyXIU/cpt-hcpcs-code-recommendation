#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Sibling-error diagnostic (``cptrec-sibling-analysis``).

Answers one question before you spend a training run on it: *are the verifier's
false positives mostly near-miss **siblings** of the gold codes — right family,
wrong member?* If they are, the residual is a discrimination problem inside a
family and a family-aware re-ranker or arbiter is well-targeted; if they are
not, that work is misdirected. Measured from files already on disk (a decoded
``predictions.csv`` + gold), no GPU and no model re-run.

What it computes
----------------
Per note: ``TP = P∩G``, ``FP = P\\G``, ``FN = G\\P`` where ``P`` is the decoded
prediction set and ``G`` the gold set. A code's **family** is its deepest KB
range (:meth:`CodeKnowledgeBase.family`), so "same family" reduces to a single
string compare — no per-code member-set scan.

* **sibling_fp_rate** (the headline gate) — share of FPs whose family matches
  *some* gold code's family (``family(p) ∈ {family(g) : g∈G}``). This is the
  advice's "correct-family-wrong-code" rate.
* **substitution_fp_rate** (strict / actionable) — share of FPs whose family
  matches a *missed* gold code's family (``family(p) ∈ {family(g) : g∈FN}``):
  the model emitted a sibling **in place of** a gold member it missed. This is
  the discrimination error a family-aware re-ranker would target.
* **nearmiss_fn_rate** — share of missed gold codes that had a predicted sibling.
* Optional per-train-bin (``--train-stats``) and per-note-tier (``--strata``)
  breakdowns, a primary-vs-add-on axis (``--ncci-dir``), and a ranked list of the
  most-confused families.

Reading the number
------------------
``sibling_fp_rate ≳ 0.25–0.30`` → family-level discrimination is the binding
error and family-aware machinery is well-targeted. ``< ~0.15`` → the errors are
spread across families, and effort is better spent on the ranking objective,
learned cardinality, or the candidate pool.

Run
---
::

    python3 -m cpt_rec.common.evaluation.sibling_analysis \\
      --pred   outputs/verifier/sections192_baseline/predictions/val__default/predictions.csv \\
      --gold   outputs/datasets/vumc/val_eval_sectioned.csv \\
      --kb     data/kb/codes_with_ranges.csv \\
      --train-stats outputs/datasets/vumc/code_frequency_stats.csv \\
      --out    outputs/verifier/sections192_baseline/predictions/val__default/sibling_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from cpt_rec.baselines.common import _ID_CANDIDATES, _pick_col
from cpt_rec.common.evaluation.oracle import load_gold
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase
from cpt_rec.common.preprocess.code_utils import pipe_split_codes

LOGGER = logging.getLogger(__name__)

BINS = ("head", "torso", "tail", "unseen")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _norm(code: str) -> str:
    return str(code).strip().upper()


def load_pred(csv_path: Path, pred_col: str = "pred_codes") -> Dict[str, Set[str]]:
    """``note_id -> decoded predicted code set`` from a predictions CSV.

    Reads the harness-standard ``note_id, pred_codes[, pred_scores]`` schema
    (pipe-separated codes). Uses ``dtype=str`` so 5-digit codes aren't coerced
    to floats (the ``write_predictions`` read-side convention).
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    id_col = _pick_col(df, _ID_CANDIDATES, f"note-id (in {csv_path})")
    if pred_col not in df.columns:
        raise ValueError(
            f"Predictions CSV {csv_path} missing {pred_col!r}; "
            f"have {list(df.columns)}"
        )
    return {
        str(nid).strip(): {_norm(c) for c in pipe_split_codes(codes)}
        for nid, codes in zip(df[id_col], df[pred_col])
    }


def load_note_tier(strata_csv: Path) -> Dict[str, str]:
    """``note_id -> tier`` from a cohort ``cohort_strata.csv`` (strata.py)."""
    df = pd.read_csv(strata_csv, dtype=str)
    if "note_id" not in df.columns or "tier" not in df.columns:
        raise ValueError(
            f"{strata_csv} must have 'note_id' and 'tier' columns; "
            f"have {list(df.columns)}"
        )
    return {str(n).strip(): str(t).strip() for n, t in zip(df["note_id"], df["tier"])}


def load_code_bins(stats_csv: Path) -> Dict[str, str]:
    """``code -> train bin`` (head/torso/tail) from code_frequency_stats.csv."""
    df = pd.read_csv(stats_csv, dtype={"code": str})
    if "bin" not in df.columns or "code" not in df.columns:
        raise ValueError(
            f"{stats_csv} must have 'code' and 'bin' columns; have {list(df.columns)}"
        )
    return {_norm(c): str(b).strip().lower() for c, b in zip(df["code"], df["bin"])}


# ---------------------------------------------------------------------------
# Family helpers (memoized — family() is the only KB call we need)
# ---------------------------------------------------------------------------

class _FamilyResolver:
    """Memoized ``code -> deepest-range (family, family_description)``.

    "Same family" is a string compare on the deepest KB range, so the whole
    diagnostic avoids :meth:`CodeKnowledgeBase.family_codes` (an O(n_codes) df
    scan per call) entirely.
    """

    def __init__(self, kb: CodeKnowledgeBase) -> None:
        self._kb = kb
        self._fam: Dict[str, Optional[str]] = {}
        self._desc: Dict[str, str] = {}

    def family(self, code: str) -> Optional[str]:
        code = _norm(code)
        if code not in self._fam:
            h = self._kb.hierarchy(code)
            if h:
                self._fam[code] = h[-1][0]
                self._desc[code] = h[-1][1]
            else:
                self._fam[code] = None
                self._desc[code] = ""
        return self._fam[code]

    def family_desc(self, code: str) -> str:
        self.family(code)
        return self._desc.get(_norm(code), "")

    def families_of(self, codes: Set[str]) -> Set[str]:
        return {f for f in (self.family(c) for c in codes) if f is not None}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(
    pred: Dict[str, Set[str]],
    gold: Dict[str, Set[str]],
    fam: _FamilyResolver,
    code_bin: Optional[Dict[str, str]],
    note_tier: Optional[Dict[str, str]],
) -> Dict:
    """Walk every note once, tallying global / per-bin / per-tier sibling stats.

    Notes scored = those present in *both* ``pred`` and ``gold`` (an
    ``--pred``/``--gold`` mismatch is logged, not silently dropped).
    """
    common_ids = [nid for nid in gold if nid in pred]
    missing = len(gold) - len(common_ids)
    if missing:
        LOGGER.warning(
            "%d gold notes have no prediction row (scored on the %d in common).",
            missing, len(common_ids),
        )

    # global confusion totals
    tp = fp = fn = 0
    fp_sibling = fp_subst = 0          # FP family ∩ gold / ∩ missed-gold
    fn_nearmiss = 0                    # missed gold with a predicted sibling

    # per-bin (bin the relevant code) and per-tier (bin the note)
    bin_fp = {b: 0 for b in BINS}
    bin_fp_sibling = {b: 0 for b in BINS}
    bin_fn = {b: 0 for b in BINS}
    bin_fn_nearmiss = {b: 0 for b in BINS}
    tier_fp: Dict[str, int] = defaultdict(int)
    tier_fp_sibling: Dict[str, int] = defaultdict(int)

    # most-confused families: family -> {count, desc, examples}
    fam_subst: Dict[str, Dict] = {}

    def _bin_of(code: str) -> str:
        if not code_bin:
            return "unseen"
        b = code_bin.get(_norm(code), "unseen")
        return b if b in bin_fp else "unseen"

    for nid in common_ids:
        G = gold[nid]
        P = pred[nid]
        TP, FP, FN = (P & G), (P - G), (G - P)
        tp += len(TP); fp += len(FP); fn += len(FN)

        gold_fams = fam.families_of(G)
        miss_fams = fam.families_of(FN)
        fp_fams = fam.families_of(FP)
        tier = note_tier.get(nid) if note_tier else None

        for p in FP:
            fpam = fam.family(p)
            is_sib = fpam is not None and fpam in gold_fams
            is_subst = fpam is not None and fpam in miss_fams
            if is_sib:
                fp_sibling += 1
            if is_subst:
                fp_subst += 1
            b = _bin_of(p)
            bin_fp[b] += 1
            if is_sib:
                bin_fp_sibling[b] += 1
            if tier:
                tier_fp[tier] += 1
                if is_sib:
                    tier_fp_sibling[tier] += 1

        for g in FN:
            gfam = fam.family(g)
            b = _bin_of(g)
            bin_fn[b] += 1
            if gfam is not None and gfam in fp_fams:
                fn_nearmiss += 1
                bin_fn_nearmiss[b] += 1
                # record the substitution: gold g missed, sibling(s) predicted
                rec = fam_subst.setdefault(
                    gfam, {"count": 0, "desc": fam.family_desc(g), "examples": []}
                )
                rec["count"] += 1
                sibs = sorted(p for p in FP if fam.family(p) == gfam)
                if len(rec["examples"]) < 5:
                    rec["examples"].append(
                        {"note_id": nid, "missed_gold": g, "predicted_siblings": sibs}
                    )

    def _rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    prec = _rate(tp, tp + fp)
    rec = _rate(tp, tp + fn)
    f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0

    result: Dict = {
        "n_notes_scored": len(common_ids),
        "n_gold_notes": len(gold),
        "confusion": {"tp": tp, "fp": fp, "fn": fn},
        "micro": {"precision": prec, "recall": rec, "f1": f1},
        "headline": {
            "sibling_fp_rate": _rate(fp_sibling, fp),
            "substitution_fp_rate": _rate(fp_subst, fp),
            "nearmiss_fn_rate": _rate(fn_nearmiss, fn),
            "n_fp_sibling": fp_sibling,
            "n_fp_substitution": fp_subst,
            "n_fn_nearmiss": fn_nearmiss,
        },
    }

    if code_bin:
        result["by_train_bin"] = {
            b: {
                "n_fp": bin_fp[b],
                "sibling_fp_rate": _rate(bin_fp_sibling[b], bin_fp[b]),
                "n_fn": bin_fn[b],
                "nearmiss_fn_rate": _rate(bin_fn_nearmiss[b], bin_fn[b]),
            }
            for b in BINS
        }
    if note_tier:
        result["by_note_tier"] = {
            t: {"n_fp": tier_fp[t], "sibling_fp_rate": _rate(tier_fp_sibling[t], tier_fp[t])}
            for t in sorted(tier_fp)
        }

    ranked = sorted(fam_subst.items(), key=lambda kv: -kv[1]["count"])
    result["most_confused_families"] = [
        {"family": f, "description": d["desc"], "substitution_count": d["count"],
         "examples": d["examples"]}
        for f, d in ranked
    ]
    return result


def primary_addon_axis(
    pred: Dict[str, Set[str]],
    gold: Dict[str, Set[str]],
    ncci_dir: Path,
) -> Dict:
    """Primary-vs-add-on axis: are FP add-on codes missing their primary?

    For each FP that is an add-on code (``get_acceptable_primaries`` returns a
    non-empty set), check whether an acceptable primary is present in the note's
    prediction set. An add-on emitted without any acceptable primary is a
    structurally invalid FP the NCCI repair (A9) can catch.
    """
    from cpt_rec.common.ncci.rule_checker import NCCIRuleChecker

    checker = NCCIRuleChecker.from_data_dir(ncci_dir)
    fp_total = addon_fp = addon_fp_no_primary = 0
    for nid, G in gold.items():
        P = pred.get(nid, set())
        for p in (P - G):
            fp_total += 1
            primaries = checker.get_acceptable_primaries(_norm(p))
            if primaries:
                addon_fp += 1
                if not (primaries & P):
                    addon_fp_no_primary += 1
    return {
        "n_fp": fp_total,
        "n_addon_fp": addon_fp,
        "addon_fp_share": round(addon_fp / fp_total, 4) if fp_total else 0.0,
        "n_addon_fp_without_primary": addon_fp_no_primary,
        "addon_fp_without_primary_share": (
            round(addon_fp_no_primary / addon_fp, 4) if addon_fp else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _gate_verdict(sibling_fp_rate: float) -> str:
    if sibling_fp_rate >= 0.25:
        return ("FAMILY-AWARE MACHINERY IS WELL-TARGETED: sibling FPs are a "
                "large share of errors.")
    if sibling_fp_rate < 0.15:
        return ("MISDIRECTED: sibling FPs are a small share — the ranking "
                "objective, learned cardinality or the pool are better bets.")
    return ("BORDERLINE (0.15-0.25): plausible; decide on the stricter "
            "substitution_fp_rate rather than this number.")


def _print_report(r: Dict) -> None:
    h = r["headline"]
    c = r["confusion"]
    print("=" * 70)
    print("SIBLING-ERROR DIAGNOSTIC")
    print("=" * 70)
    print(f"Notes scored : {r['n_notes_scored']:,}   "
          f"TP={c['tp']}  FP={c['fp']}  FN={c['fn']}")
    m = r["micro"]
    print(f"micro        : P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print("-" * 70)
    print(f"sibling_fp_rate        : {h['sibling_fp_rate']:.4f}   "
          f"({h['n_fp_sibling']}/{c['fp']} FPs share a gold family)  <-- GATE")
    print(f"substitution_fp_rate   : {h['substitution_fp_rate']:.4f}   "
          f"({h['n_fp_substitution']}/{c['fp']} FPs replace a MISSED gold sibling)")
    print(f"nearmiss_fn_rate       : {h['nearmiss_fn_rate']:.4f}   "
          f"({h['n_fn_nearmiss']}/{c['fn']} missed gold had a predicted sibling)")
    if "by_train_bin" in r:
        print("-" * 70)
        print("By train bin   (n_fp | sibling_fp_rate || n_fn | nearmiss_fn_rate)")
        for b in BINS:
            bb = r["by_train_bin"][b]
            print(f"   {b:7s} : {bb['n_fp']:>5} | {bb['sibling_fp_rate']:.4f}  ||  "
                  f"{bb['n_fn']:>5} | {bb['nearmiss_fn_rate']:.4f}")
    if "by_note_tier" in r:
        print("-" * 70)
        print("By note tier   (n_fp | sibling_fp_rate)")
        for t, tb in r["by_note_tier"].items():
            print(f"   {t:7s} : {tb['n_fp']:>5} | {tb['sibling_fp_rate']:.4f}")
    if "primary_addon" in r:
        pa = r["primary_addon"]
        print("-" * 70)
        print(f"Add-on FPs    : {pa['n_addon_fp']}/{pa['n_fp']} "
              f"({pa['addon_fp_share']:.4f}); "
              f"{pa['n_addon_fp_without_primary']} lack an acceptable primary "
              f"({pa['addon_fp_without_primary_share']:.4f})")
    fams = r.get("most_confused_families", [])
    if fams:
        print("-" * 70)
        print(f"Top confused families (by substitution count; {len(fams)} total)")
        for frow in fams[:15]:
            print(f"   {frow['family']:12s} x{frow['substitution_count']:<4} "
                  f"{(frow['description'] or '')[:48]}")
    print("-" * 70)
    print("VERDICT: " + _gate_verdict(h["sibling_fp_rate"]))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    pred_csv: Path,
    gold_csv: Path,
    kb_csv: Path,
    out_json: Optional[Path],
    train_stats: Optional[Path],
    strata_csv: Optional[Path],
    ncci_dir: Optional[Path],
    top_families: int,
) -> Dict:
    gold = load_gold(gold_csv)
    pred = load_pred(pred_csv)
    kb = CodeKnowledgeBase.from_csv(kb_csv, build_index=False)
    fam = _FamilyResolver(kb)
    code_bin = load_code_bins(train_stats) if train_stats else None
    note_tier = load_note_tier(strata_csv) if strata_csv else None

    LOGGER.info("Loaded %d gold notes, %d pred notes, %d KB codes.",
                len(gold), len(pred), len(kb))

    result = analyze(pred, gold, fam, code_bin, note_tier)
    if ncci_dir is not None:
        result["primary_addon"] = primary_addon_axis(pred, gold, ncci_dir)
    if top_families >= 0:
        result["most_confused_families"] = result["most_confused_families"][:top_families]
    result["inputs"] = {
        "pred": str(pred_csv),
        "gold": str(gold_csv),
        "kb": str(kb_csv),
        "train_stats": str(train_stats) if train_stats else None,
        "strata": str(strata_csv) if strata_csv else None,
        "ncci_dir": str(ncci_dir) if ncci_dir else None,
    }
    result["gate_verdict"] = _gate_verdict(result["headline"]["sibling_fp_rate"])

    _print_report(result)
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        LOGGER.info("Wrote sibling report -> %s", out_json)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Sibling-error diagnostic."
    )
    p.add_argument("--pred", required=True, type=Path,
                   help="Decoded predictions CSV (note_id, pred_codes).")
    p.add_argument("--gold", required=True, type=Path,
                   help="Gold *_sectioned.csv (proc_codes).")
    p.add_argument("--kb", required=True, type=Path,
                   help="codes_with_ranges.csv (for code families).")
    p.add_argument("--train-stats", type=Path, default=None,
                   help="code_frequency_stats.csv for the per-bin breakdown.")
    p.add_argument("--strata", type=Path, default=None,
                   help="cohort_strata.csv for the per-note-tier breakdown.")
    p.add_argument("--ncci-dir", type=Path, default=None,
                   help="data/ncci for the primary-vs-add-on axis.")
    p.add_argument("--top-families", type=int, default=25,
                   help="Keep the N most-confused families in the report "
                        "(-1 = all).")
    p.add_argument("--out", dest="out_json", type=Path, default=None)
    args = p.parse_args()

    run(
        pred_csv=args.pred,
        gold_csv=args.gold,
        kb_csv=args.kb,
        out_json=args.out_json,
        train_stats=args.train_stats,
        strata_csv=args.strata,
        ncci_dir=args.ncci_dir,
        top_families=args.top_families,
    )


if __name__ == "__main__":
    main()
