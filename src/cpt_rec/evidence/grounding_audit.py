#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Grounding audit (private cohort): does the snippet the verifier saw
actually support the code it predicted?

Three subcommands over the evidence ledger of a scored run
(``cptrec-verifier-predict --emit-ledger``):

``sample``
    Stratified sample of ~150–200 (note, code, best-snippet) triples over
    decision (TP/FP among *kept* predictions) x train bin (head/torso/tail/
    unseen).  Writes ``samples.csv`` (full metadata) plus a **blinded**
    ``rating_sheet.csv`` for the ~50 human co-ratings (no TP/FP or verdict
    columns, so the rater can't anchor).

``judge``
    LLM 3-way judgment per triple on Azure (default deployment
    ``gpt-5.6-sol``): SUPPORTED / PARTIAL / NOT_SUPPORTED + one-line
    rationale.  Writes ``judge_results.csv``.

``agree``
    Joins the judge results with the filled-in human sheet: exact agreement,
    Cohen's kappa, confusion matrix -> ``agreement.json`` + stdout table.

Usage (server; sample+agree are CPU, judge needs Azure creds in ``.env``)::

    cptrec-grounding-audit sample \\
        --ledger outputs/verifier/sections192_baseline/predictions/val__default/ledger.jsonl \\
        --gold outputs/datasets/vumc/val_eval_sectioned.csv \\
        --train-stats outputs/datasets/vumc/code_frequency_stats.csv \\
        --n 200 --n-corate 50 --out-dir outputs/evaluation/grounding_audit_sections192_baseline

    cptrec-grounding-audit judge \\
        --samples outputs/evaluation/grounding_audit_sections192_baseline/samples.csv \\
        --deployment gpt-5.6-sol \\
        --out outputs/evaluation/grounding_audit_sections192_baseline/judge_results.csv

    cptrec-grounding-audit agree \\
        --judge outputs/evaluation/grounding_audit_sections192_baseline/judge_results.csv \\
        --human outputs/evaluation/grounding_audit_sections192_baseline/rating_sheet.csv \\
        --out outputs/evaluation/grounding_audit_sections192_baseline/agreement.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from cpt_rec.common.evaluation.harness import load_gold
from cpt_rec.common.evaluation.sibling_analysis import load_code_bins

LOGGER = logging.getLogger(__name__)

VERDICTS = ("SUPPORTED", "PARTIAL", "NOT_SUPPORTED")
_BINS = ("head", "torso", "tail", "unseen")

JUDGE_SYSTEM = """\
You are an expert surgical-coding auditor. You are given a CPT/HCPCS code
(with its official descriptor) and ONE evidence snippet extracted from an
operative note. Judge whether the snippet supports assigning that code to
this operation.

Answer with STRICT JSON only, no other text:
{"verdict": "<SUPPORTED|PARTIAL|NOT_SUPPORTED>", "rationale": "<one sentence>"}

Verdict definitions:
- SUPPORTED: the snippet documents the specific procedure the code describes
  (technique/anatomy consistent with the descriptor).
- PARTIAL: the snippet describes a closely related or ambiguous procedure —
  right family but the code-discriminating detail (approach, laterality,
  extent, size, add-on condition) is absent or unclear in this snippet.
- NOT_SUPPORTED: the snippet is unrelated to the code, or contradicts it.

Judge ONLY from the snippet. Do not assume unstated facts.\
"""

JUDGE_USER_TMPL = """\
Code: {code}
Descriptor: {description}

Evidence snippet:
\"\"\"{snippet}\"\"\"
"""


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def sample_triples(
    ledger_path: Path,
    gold_csv: Path,
    train_stats: Path,
    out_dir: Path,
    n: int = 200,
    n_corate: int = 50,
    min_per_cell: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    gold = load_gold(gold_csv)
    bins = load_code_bins(train_stats)

    rows: List[Dict] = []
    n_total = n_kept = n_no_snippet = 0
    with open(ledger_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            n_total += 1
            if not e.get("selected_pre_rule"):
                continue  # audit what the system actually predicted
            n_kept += 1
            snippet = e.get("best_snippet")
            if not snippet or not str(snippet).strip():
                n_no_snippet += 1
                continue
            note_id = str(e["note_id"]).strip()
            code = str(e["code"]).strip()
            decision = "TP" if code in gold.get(note_id, set()) else "FP"
            rows.append({
                "note_id": note_id,
                "code": code,
                "code_description": e.get("code_description") or "",
                "snippet": str(snippet).strip(),
                "verifier_score": e.get("verifier_score"),
                "decision": decision,
                "bin": bins.get(code, "unseen"),
            })
    LOGGER.info(
        "Ledger %s: %d entries, %d kept predictions, %d kept WITHOUT a "
        "snippet (%.1f%% — structurally ungroundable, reported not judged)",
        ledger_path, n_total, n_kept, n_no_snippet,
        100.0 * n_no_snippet / max(n_kept, 1),
    )

    # ---- stratified allocation over decision x bin ----
    cells: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        cells[(r["decision"], r["bin"])].append(r)
    rng = random.Random(seed)
    cell_keys = sorted(cells)
    # floor: min_per_cell (or the whole cell if smaller); then fill the
    # remainder proportionally to cell size.
    alloc = {k: min(min_per_cell, len(cells[k])) for k in cell_keys}
    remaining = n - sum(alloc.values())
    if remaining > 0:
        pool_sizes = {k: len(cells[k]) - alloc[k] for k in cell_keys}
        total_extra = sum(pool_sizes.values())
        for k in cell_keys:
            if total_extra <= 0:
                break
            extra = round(remaining * pool_sizes[k] / total_extra)
            alloc[k] += min(extra, pool_sizes[k])
    picked: List[Dict] = []
    for k in cell_keys:
        pool = sorted(cells[k], key=lambda r: (r["note_id"], r["code"]))
        take = min(alloc[k], len(pool))
        picked.extend(rng.sample(pool, take))
    rng.shuffle(picked)
    df = pd.DataFrame(picked)
    df.insert(0, "sample_id", [f"g{idx:04d}" for idx in range(len(df))])
    corate_ids = set(
        rng.sample(df["sample_id"].tolist(), min(n_corate, len(df)))
    )
    df["human_corate"] = df["sample_id"].isin(corate_ids)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "samples.csv", index=False)
    # Blinded human sheet: co-rate subset only, no decision/bin/score columns.
    sheet = df[df["human_corate"]][
        ["sample_id", "code", "code_description", "snippet"]
    ].copy()
    sheet["rating"] = ""  # fill with SUPPORTED / PARTIAL / NOT_SUPPORTED
    sheet.to_csv(out_dir / "rating_sheet.csv", index=False)

    strata = df.groupby(["decision", "bin"]).size().rename("n").reset_index()
    strata.to_csv(out_dir / "sample_strata.csv", index=False)
    with open(out_dir / "sample_stats.json", "w") as f:
        json.dump({
            "ledger": str(ledger_path), "seed": seed,
            "n_ledger_entries": n_total, "n_kept_predictions": n_kept,
            "n_kept_without_snippet": n_no_snippet,
            "pct_kept_without_snippet": round(
                100.0 * n_no_snippet / max(n_kept, 1), 2),
            "n_sampled": int(len(df)), "n_corate": len(corate_ids),
        }, f, indent=2)
    LOGGER.info("Wrote %d samples (%d co-rate) -> %s",
                len(df), len(corate_ids), out_dir)
    print(strata.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------

def _parse_judge_json(text: str) -> Dict[str, str]:
    """Extract the verdict JSON object, tolerating stray prose/fences."""
    s = text.strip()
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            verdict = str(obj.get("verdict", "")).upper().replace(" ", "_")
            if verdict in VERDICTS:
                return {"verdict": verdict,
                        "rationale": str(obj.get("rationale", ""))}
        except json.JSONDecodeError:
            pass
    return {"verdict": "PARSE_ERROR", "rationale": s[:300]}


def judge_samples(
    samples_csv: Path,
    out_csv: Path,
    deployment: str = "gpt-5.6-sol",
    rpm: int = 120,
    limit: Optional[int] = None,
) -> None:
    from cpt_rec.baselines.llm import (
        AzureOpenAIBackend,
        FixedIntervalRateLimiter,
    )

    df = pd.read_csv(samples_csv, dtype=str)
    if limit:
        df = df.head(limit)
    backend = AzureOpenAIBackend(deployment_name=deployment)
    limiter = FixedIntervalRateLimiter(rpm)

    results = []
    for _, row in df.iterrows():
        limiter.acquire()
        desc = row.get("code_description")
        if pd.isna(desc) or not str(desc).strip():
            desc = "(no descriptor)"
        user = JUDGE_USER_TMPL.format(
            code=row["code"], description=desc, snippet=row["snippet"],
        )
        try:
            raw = backend.chat(JUDGE_SYSTEM, user)
            parsed = _parse_judge_json(raw)
        except Exception as exc:  # keep going; one flaky call != lost run
            LOGGER.warning("judge call failed for %s: %s", row["sample_id"], exc)
            parsed = {"verdict": "CALL_ERROR", "rationale": str(exc)[:300]}
        results.append({"sample_id": row["sample_id"],
                        "judge_verdict": parsed["verdict"],
                        "judge_rationale": parsed["rationale"]})
        if len(results) % 25 == 0:
            LOGGER.info("judged %d/%d", len(results), len(df))

    out = df.merge(pd.DataFrame(results), on="sample_id")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    counts = Counter(out["judge_verdict"])
    print("\njudge verdicts:", dict(counts))
    if "decision" in out.columns:
        pivot = out.groupby(["decision", "judge_verdict"]).size().unstack(fill_value=0)
        print("\nby TP/FP decision:")
        print(pivot.to_string())
        supported_rate = {
            d: round(float((sub["judge_verdict"] == "SUPPORTED").mean()), 4)
            for d, sub in out.groupby("decision")
        }
        print("\nSUPPORTED rate:", supported_rate)
    LOGGER.info("Wrote judge results -> %s", out_csv)


# ---------------------------------------------------------------------------
# agree
# ---------------------------------------------------------------------------

_HUMAN_ALIASES = {
    "S": "SUPPORTED", "SUPPORTED": "SUPPORTED",
    "P": "PARTIAL", "PARTIAL": "PARTIAL", "PARTIALLY_SUPPORTED": "PARTIAL",
    "N": "NOT_SUPPORTED", "NOT_SUPPORTED": "NOT_SUPPORTED", "NS": "NOT_SUPPORTED",
}


def _cohen_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(judge_csv: Path, human_csv: Path, out_json: Path) -> None:
    dj = pd.read_csv(judge_csv, dtype=str)
    dh = pd.read_csv(human_csv, dtype=str)
    if "rating" not in dh.columns:
        raise ValueError(f"{human_csv} has no 'rating' column.")
    dh = dh[dh["rating"].notna() & (dh["rating"].str.strip() != "")].copy()
    dh["human_verdict"] = (
        dh["rating"].str.strip().str.upper().map(_HUMAN_ALIASES)
    )
    unknown = dh[dh["human_verdict"].isna()]
    if len(unknown):
        raise ValueError(
            f"Unrecognized human ratings: {sorted(set(unknown['rating']))} "
            f"(accepted: {sorted(set(_HUMAN_ALIASES))})"
        )
    merged = dj.merge(dh[["sample_id", "human_verdict"]], on="sample_id")
    merged = merged[merged["judge_verdict"].isin(VERDICTS)]
    if merged.empty:
        raise ValueError("No overlapping rated samples between judge and human.")

    j = merged["judge_verdict"].tolist()
    h = merged["human_verdict"].tolist()
    exact = sum(1 for x, y in zip(j, h) if x == y) / len(j)
    kappa = _cohen_kappa(j, h)
    conf = (
        merged.groupby(["human_verdict", "judge_verdict"]).size()
        .unstack(fill_value=0)
    )
    result = {
        "n_corated": int(len(merged)),
        "exact_agreement": round(exact, 4),
        "cohen_kappa": round(kappa, 4),
        "confusion_human_rows_judge_cols": {
            str(r): {str(c): int(v) for c, v in row.items()}
            for r, row in conf.iterrows()
        },
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"n={result['n_corated']}  exact agreement={exact:.3f}  "
          f"kappa={kappa:.3f}")
    print(conf.to_string())
    LOGGER.info("Wrote agreement -> %s", out_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Grounding audit.")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("sample", help="Stratified triple sampling from a ledger.")
    p.add_argument("--ledger", required=True, type=Path)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--train-stats", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n-corate", type=int, default=50)
    p.add_argument("--min-per-cell", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p = sp.add_parser("judge", help="LLM 3-way grounding judgment (Azure).")
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--deployment", default="gpt-5.6-sol")
    p.add_argument("--rpm", type=int, default=120)
    p.add_argument("--limit", type=int, default=None,
                   help="Judge only the first N samples (smoke test).")

    p = sp.add_parser("agree", help="Judge vs human agreement stats.")
    p.add_argument("--judge", required=True, type=Path)
    p.add_argument("--human", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.cmd == "sample":
        sample_triples(
            ledger_path=args.ledger, gold_csv=args.gold,
            train_stats=args.train_stats, out_dir=args.out_dir,
            n=args.n, n_corate=args.n_corate,
            min_per_cell=args.min_per_cell, seed=args.seed,
        )
    elif args.cmd == "judge":
        judge_samples(
            samples_csv=args.samples, out_csv=args.out,
            deployment=args.deployment, rpm=args.rpm, limit=args.limit,
        )
    elif args.cmd == "agree":
        agreement(judge_csv=args.judge, human_csv=args.human, out_json=args.out)


if __name__ == "__main__":
    main()
