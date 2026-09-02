# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Strata-aware A/B of kb_index ``diagnose`` audit CSVs.

``cptrec-build-kb-index diagnose --audit-csv <path>`` writes one row per gold
code with the rank at which each retriever surfaced it::

    note_id, gold_code, in_kb, bm25_rank, dense_rank, union_rank

This tool bins those gold codes by *training frequency*
(head / torso / tail / unseen, via ``code_bins.load_code_bins``) and
reports recall@k per bin for one or more audits, so a ``rich`` vs
``default`` KB-index comparison shows whether descriptor enrichment lifts
the recall **ceiling where it matters — the unseen bin**.  Overall recall
washes a small unseen gain out (unseen is only ~4% of gold on the operative-
note corpus), which is why the unseen bin, not overall recall, is the binding
retrieval floor.  Run the rich and default indices through ``diagnose``,
then A/B their audit CSVs here *before* spending a regeneration pass on
descriptor enrichment.

This is an offline diagnostic, alongside ``oracle.py`` and
``candidate_union.py``; it adds no logic to any pipeline stage.

Invocation (no entry point, mirrors oracle.py / candidate_union.py)::

    python3 -m cpt_rec.common.evaluation.audit_strata \\
        --audit default=outputs/baselines/kb_index/diagnose/default_audit.csv \\
        --audit rich=outputs/baselines/kb_index/diagnose/rich_audit.csv \\
        --train-stats outputs/datasets/vumc/code_frequency_stats.csv \\
        --rank-col union_rank --ks 50,100,200,500 \\
        --out-json outputs/baselines/kb_index/diagnose/audit_strata.json

The first ``--audit`` is the baseline; later ones are diffed against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cpt_rec.common.evaluation.code_bins import BINS, load_code_bins


def _parse_audit_arg(spec: str) -> Tuple[str, Path]:
    """Parse a ``LABEL=PATH`` (or bare ``PATH``) ``--audit`` value."""
    if "=" in spec:
        label, _, path = spec.partition("=")
        label = label.strip()
        path = path.strip()
    else:
        path = spec.strip()
        label = Path(path).stem
    if not path:
        raise argparse.ArgumentTypeError(f"--audit {spec!r}: empty path")
    return label, Path(path)


def _coerce_in_kb(series: Optional[pd.Series], n: int) -> pd.Series:
    """Coerce an ``in_kb`` column (bool / 'True' / 1) to boolean."""
    if series is None:
        return pd.Series([False] * n)
    return (
        series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    )


def per_bin_recall(
    audit_csv: Path,
    code_bin: Dict[str, str],
    ks: List[int],
    rank_col: str,
) -> Dict[str, Dict[str, float]]:
    """Per-bin recall@k + ceiling + %in-KB for one audit CSV.

    A gold code counts as recalled@k iff its ``rank_col`` is present
    (non-null ⇒ inside the top-N the diagnose searched) and ``< k``
    (ranks are 0-indexed).  ``ceiling`` is the pool-recall ceiling: the
    fraction with any non-null rank (recalled at the searched depth).
    """
    df = pd.read_csv(audit_csv, dtype={"gold_code": str})
    if "gold_code" not in df.columns:
        raise ValueError(
            f"{audit_csv} missing 'gold_code'; got {list(df.columns)}. "
            "Pass a kb_index 'diagnose --audit-csv' file."
        )
    if rank_col not in df.columns:
        raise ValueError(
            f"{audit_csv} missing rank column {rank_col!r}; "
            f"available: {[c for c in df.columns if c.endswith('_rank')]}."
        )

    rank = pd.to_numeric(df[rank_col], errors="coerce")
    in_kb = _coerce_in_kb(df.get("in_kb"), len(df))
    bin_of = (
        df["gold_code"].fillna("").str.strip().str.upper()
        .map(lambda c: code_bin.get(c, "unseen"))
    )

    out: Dict[str, Dict[str, float]] = {}
    for b in (*BINS, "all"):
        mask = pd.Series(True, index=df.index) if b == "all" else (bin_of == b)
        n = int(mask.sum())
        rec: Dict[str, float] = {"n_gold": n}
        if n == 0:
            rec["pct_in_kb"] = 0.0
            rec["ceiling"] = 0.0
            for k in ks:
                rec[f"recall@{k}"] = 0.0
            out[b] = rec
            continue
        r = rank[mask]
        present = r.notna()
        rec["pct_in_kb"] = round(float(in_kb[mask].mean()), 4)
        rec["ceiling"] = round(float(present.mean()), 4)
        for k in ks:
            rec[f"recall@{k}"] = round(float(((r < k) & present).mean()), 4)
        out[b] = rec
    return out


def _fmt_table(label: str, table: Dict[str, Dict[str, float]],
               ks: List[int]) -> List[str]:
    cols = ["n_gold", "in_kb%", "ceiling", *[f"r@{k}" for k in ks]]
    lines = [f"== {label} ==", "  " + f"{'bin':<8}" +
             "".join(f"{c:>9}" for c in cols)]
    for b in (*BINS, "all"):
        row = table[b]
        cells = [f"{row['n_gold']:>9d}",
                 f"{row['pct_in_kb']*100:>8.1f}",
                 f"{row['ceiling']*100:>8.1f}"]
        cells += [f"{row[f'recall@{k}']*100:>8.1f}" for k in ks]
        lines.append("  " + f"{b:<8}" + "".join(cells))
    return lines


def _fmt_delta(base_label: str, base: Dict[str, Dict[str, float]],
               other_label: str, other: Dict[str, Dict[str, float]],
               ks: List[int]) -> List[str]:
    metrics = ["ceiling", *[f"recall@{k}" for k in ks]]
    cols = ["ceiling", *[f"r@{k}" for k in ks]]
    lines = [f"== Δ ({other_label} − {base_label}), percentage points ==",
             "  " + f"{'bin':<8}" + "".join(f"{c:>9}" for c in cols)]
    for b in (*BINS, "all"):
        cells = []
        for m in metrics:
            d = (other[b][m] - base[b][m]) * 100.0
            cells.append(f"{d:>+8.1f}")
        lines.append("  " + f"{b:<8}" + "".join(cells))
    return lines


def main() -> None:
    p = argparse.ArgumentParser(
        description="Strata-aware A/B of kb_index diagnose audit CSVs."
    )
    p.add_argument(
        "--audit", action="append", required=True, metavar="LABEL=PATH",
        help="Audit CSV as LABEL=PATH (repeatable; first is the baseline).",
    )
    p.add_argument("--train-stats", required=True, type=Path,
                   help="code_frequency_stats.csv (code,bin columns).")
    p.add_argument("--ks", default="50,100,200,500",
                   help="Comma-separated recall@k cutoffs.")
    p.add_argument("--rank-col", default="union_rank",
                   choices=["union_rank", "bm25_rank", "dense_rank"],
                   help="Which retriever's rank to score (default union).")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    ks = [int(x) for x in str(args.ks).split(",") if str(x).strip()]
    code_bin = load_code_bins(str(args.train_stats))
    audits = [_parse_audit_arg(a) for a in args.audit]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    order: List[str] = []
    for label, path in audits:
        results[label] = per_bin_recall(path, code_bin, ks, args.rank_col)
        order.append(label)

    lines: List[str] = [
        f"Strata-aware retrieval A/B  (rank_col={args.rank_col}, "
        f"bins by {args.train_stats})",
    ]
    for label in order:
        lines += _fmt_table(label, results[label], ks)
        lines.append("")
    if len(order) >= 2:
        base = order[0]
        for other in order[1:]:
            lines += _fmt_delta(base, results[base], other, results[other], ks)
            lines.append("")
    report = "\n".join(lines)
    print(report)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rank_col": args.rank_col,
            "ks": ks,
            "train_stats": str(args.train_stats),
            "baseline": order[0] if order else None,
            "audits": {label: results[label] for label in order},
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
