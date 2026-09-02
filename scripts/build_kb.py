#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Build and check a knowledge-base CSV for cpt_rec.

The repository ships ``data/kb/hcpcs_level2_public.csv`` -- HCPCS Level II only,
which is CMS public domain. CPT (Level I) descriptors are copyrighted by the
American Medical Association and are **not** redistributable, so reproducing the
paper's CPT numbers requires your own licensed copy of the CPT descriptor set.

    # check any KB file against the column contract
    python scripts/build_kb.py validate --kb data/kb/hcpcs_level2_public.csv

    # merge your licensed CPT descriptors onto the public HCPCS base
    python scripts/build_kb.py merge \
        --base data/kb/hcpcs_level2_public.csv \
        --cpt  /path/to/your/cpt_descriptors.csv \
        --out  data/kb/codes_with_ranges.csv

`--cpt` needs only two columns, `code` and `code_description`; anything else it
carries that matches the contract (lay terms, range levels) is carried through.
The merge never writes into the repository's tracked demo file, and `data/**`
is gitignored by default, so a licensed build cannot be committed by accident.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCHEMA = Path(__file__).resolve().parent.parent / "data" / "kb" / "kb_schema.json"
RANGE_LEVELS = range(1, 7)


def _contract() -> dict:
    return json.loads(SCHEMA.read_text())


def validate(kb_path: Path) -> int:
    contract = _contract()
    df = pd.read_csv(kb_path, dtype=str, keep_default_na=False)
    problems: list[str] = []

    for col, spec in contract["columns"].items():
        if spec["required"] and col not in df.columns:
            problems.append(f"missing required column: {col}")
    if "code" in df.columns:
        blank = int((df["code"].str.strip() == "").sum())
        if blank:
            problems.append(f"{blank} rows have an empty `code`")
        dupes = df["code"].duplicated().sum()
        if dupes:
            problems.append(f"{dupes} duplicate codes (rows are looked up by code)")
        if not df["code"].equals(df["code"].str.upper()):
            problems.append("`code` contains lowercase characters; codes must be uppercase")
    if "code_description" in df.columns:
        empty = int((df["code_description"].str.strip() == "").sum())
        if empty:
            problems.append(f"{empty} rows have an empty `code_description`")
    if "code_system" in df.columns:
        allowed = set(contract["columns"]["code_system"]["values"])
        seen = set(df["code_system"].unique())
        if not seen <= allowed:
            problems.append(f"unexpected code_system values: {sorted(seen - allowed)}")

    print(f"{kb_path}: {len(df)} rows, {len(df.columns)} columns")
    if "code_system" in df.columns:
        for sysname, n in df["code_system"].value_counts().items():
            lay = 0
            if "code_lay_term" in df.columns:
                sub = df.loc[df["code_system"] == sysname, "code_lay_term"]
                lay = int((sub.str.strip() != "").sum())
            print(f"  {sysname:6s} {n:6d} codes   ({lay} with lay terms)")
    depth = sum(1 for i in RANGE_LEVELS if f"code_range_{i}" in df.columns)
    print(f"  hierarchy levels present: {depth}/6")

    if problems:
        print("\nFAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: file satisfies the column contract in data/kb/kb_schema.json")
    return 0


def merge(base: Path, cpt: Path, out: Path) -> int:
    b = pd.read_csv(base, dtype=str, keep_default_na=False)
    c = pd.read_csv(cpt, dtype=str, keep_default_na=False)
    need = {"code", "code_description"}
    missing = need - set(c.columns)
    if missing:
        print(f"ERROR: --cpt is missing {sorted(missing)}", file=sys.stderr)
        return 1

    c = c.copy()
    c["code"] = c["code"].str.strip().str.upper()
    if "code_system" not in c.columns:
        c["code_system"] = "CPT"
    for col in b.columns:
        if col not in c.columns:
            c[col] = ""
    c = c[list(b.columns)]

    overlap = set(b["code"]) & set(c["code"])
    if overlap:
        print(f"note: {len(overlap)} codes appear in both files; the licensed file wins")
        b = b[~b["code"].isin(overlap)]

    merged = pd.concat([b, c], ignore_index=True).sort_values("code", kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"wrote {out}: {len(merged)} codes "
          f"({len(b)} from {base.name} + {len(c)} from {cpt.name})")
    return validate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check a KB CSV against the column contract")
    v.add_argument("--kb", type=Path, required=True)

    m = sub.add_parser("merge", help="merge licensed CPT descriptors onto the public base")
    m.add_argument("--base", type=Path, default=Path("data/kb/hcpcs_level2_public.csv"))
    m.add_argument("--cpt", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)

    a = ap.parse_args()
    sys.exit(validate(a.kb) if a.cmd == "validate" else merge(a.base, a.cpt, a.out))


if __name__ == "__main__":
    main()
