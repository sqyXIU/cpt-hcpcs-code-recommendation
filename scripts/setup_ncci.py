#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Lay out the CMS NCCI edit tables in the directory this project expects.

The NCCI tables are US-government public domain, but they are ~250 MB and CMS
reissues them quarterly, so they are downloaded rather than committed here.
This script turns the ZIPs CMS publishes into the layout the loaders want, and
then verifies that the result actually loads.

Target layout::

    <out>/
      PTP_edits/ccipra-v<ver>-f{1,2,3,4}/ccipra-v<ver>-f{1,2,3,4}.TXT
      AOC_edits/AOC_V<quarter>-F-MCR.txt
      MUE/MCR_MUE_PractitionerServices_Eff_<mm-dd-yyyy>.csv

Usage
-----
Download the four PTP files, the AOC flat file, and the practitioner MUE table
from the CMS NCCI pages (see SOURCES below), put the archives in one directory,
then::

    python scripts/setup_ncci.py --from-zips ~/Downloads/ncci --out data/ncci

Already have a directory laid out?  Check it::

    python scripts/setup_ncci.py --check --out data/ncci

Provenance
----------
The results reported in the paper used:

    PTP   ccipra-v320r0 (version 32.0, release 0)
    AOC   AOC_V2026Q2-F-MCR.txt
    MUE   MCR_MUE_PractitionerServices_Eff_04-01-2026.csv

A different quarter will shift the constraint-violation counts slightly.  The
edit tables are held fixed rather than matched to each note's service date;
that limitation is stated in the paper.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

SOURCES = """\
CMS NCCI download pages (public domain, no account required):

  PTP  Procedure-to-Procedure edits, practitioner services
       https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-procedure-procedure-ptp-edits
       -> the four practitioner files, ccipra-v<ver>-f1 .. -f4

  AOC  Add-on Code edits
       https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-add-code-edits
       -> the Medicare flat file, AOC_V<quarter>-F-MCR.txt

  MUE  Medically Unlikely Edits
       https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-medically-unlikely-edits
       -> Practitioner Services MUE table (CSV)

CMS moves these URLs when the quarter rolls over.  If a link 404s, start from
https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits
"""

PTP_RE = re.compile(r"ccipra[-_]?v\d+r\d+[-_]?f[1-4]", re.I)
AOC_RE = re.compile(r"AOC_V\d{4}Q\d-F-MCR\.txt$", re.I)
MUE_RE = re.compile(r"MCR_MUE_PractitionerServices_Eff_.*\.csv$", re.I)


def _place(src_name: str, data: bytes, out: Path) -> Path | None:
    """Write one extracted member to its slot in *out*, or return None."""
    name = Path(src_name).name
    if not name or name.startswith("."):
        return None

    if name.upper().endswith(".TXT") and PTP_RE.search(name):
        stem = PTP_RE.search(name).group(0).lower().replace("_", "-")
        dest = out / "PTP_edits" / stem / f"{stem}.TXT"
    elif AOC_RE.search(name):
        dest = out / "AOC_edits" / name
    elif MUE_RE.search(name):
        dest = out / "MUE" / name
    else:
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def unpack(zips_dir: Path, out: Path) -> int:
    """Extract every recognized NCCI member under *zips_dir* into *out*."""
    placed = 0
    candidates = sorted(zips_dir.rglob("*.zip")) + sorted(zips_dir.rglob("*.ZIP"))
    for zpath in candidates:
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                dest = _place(member, zf.read(member), out)
                if dest is not None:
                    print(f"  {zpath.name}:{Path(member).name} -> {dest.relative_to(out)}")
                    placed += 1

    # loose (already-extracted) files are fine too
    for loose in sorted(zips_dir.rglob("*")):
        if loose.is_file() and loose.suffix.lower() not in (".zip",):
            dest = _place(loose.name, loose.read_bytes(), out)
            if dest is not None:
                print(f"  {loose.name} -> {dest.relative_to(out)}")
                placed += 1
    return placed


def check(out: Path) -> int:
    """Load the directory through the real checker and report what it found."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from cpt_rec.common.ncci import NCCIRuleChecker
    except ImportError as exc:  # pragma: no cover - environment problem
        print(f"cannot import the package ({exc}); install it first:\n"
              "    pip install -e .", file=sys.stderr)
        return 2

    try:
        checker = NCCIRuleChecker.from_data_dir(out)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print(f"\n{SOURCES}", file=sys.stderr)
        return 1

    print(f"OK: {out} loads.")
    print(f"  active PTP pairs          : {len(checker._pair_to_ccmi):,}")
    print(f"  add-on codes w/ primaries : {len(checker._addon_to_primaries):,}")
    print(f"  contractor-defined add-ons: {len(checker._contractor_defined_addons):,}")
    print(f"  codes with an MUE limit   : {len(checker._code_to_max_units):,}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SOURCES,
    )
    ap.add_argument("--from-zips", type=Path, metavar="DIR",
                    help="directory holding the CMS archives (or extracted files)")
    ap.add_argument("--out", type=Path, default=Path("data/ncci"),
                    help="target directory (default: data/ncci)")
    ap.add_argument("--check", action="store_true",
                    help="only verify that --out loads")
    args = ap.parse_args()

    if not args.check and args.from_zips is None:
        ap.print_help()
        print(f"\n{SOURCES}")
        return 2

    if args.from_zips is not None:
        if not args.from_zips.is_dir():
            print(f"not a directory: {args.from_zips}", file=sys.stderr)
            return 2
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"unpacking {args.from_zips} -> {args.out}")
        placed = unpack(args.from_zips, args.out)
        if placed == 0:
            print("\nNothing recognized. Expected file names like:\n"
                  "  ccipra-v320r0-f1.TXT   (four of these)\n"
                  "  AOC_V2026Q2-F-MCR.txt\n"
                  "  MCR_MUE_PractitionerServices_Eff_04-01-2026.csv\n"
                  f"\n{SOURCES}", file=sys.stderr)
            return 1
        print(f"placed {placed} file(s)\n")

    return check(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
