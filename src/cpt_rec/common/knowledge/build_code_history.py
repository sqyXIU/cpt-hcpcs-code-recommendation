#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
CLI: build a cached ``CodeHistory`` artifact from the three KB CSVs.

Reads:

* ``codes_with_ranges.csv``  (active 2026 set)
* ``code_changes.csv``       (2018–2026 new/revised/deleted events)
* ``deleted_codes.csv``      (1990–2026 precise deletion ledger)

Writes into ``--out`` (default ``outputs/indices/code_history/``):

* ``code_events.parquet``     — flattened event table (audit-friendly)
* ``code_crosswalk.json``     — ``{deleted_code: substitute}`` resolved
  as of ``--as-of`` (default: today).  Useful as a drop-in lookup in
  baselines before :mod:`cpt_rec.common.knowledge.code_history`
  is wired into the prediction path.
* ``code_history_stats.json`` — summary counts for reporting.

CLI
---

::

    cptrec-build-code-history \\
        --kb data/kb/codes_with_ranges.csv \\
        --changes data/kb/code_changes.csv \\
        --deleted data/kb/deleted_codes.csv \\
        --out outputs/indices/code_history/
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from cpt_rec.common.knowledge.code_history import CodeHistory
from cpt_rec.common.knowledge.code_kb import CodeKnowledgeBase

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build outputs/indices/code_history/code_events.parquet + crosswalk JSON "
                    "from the three knowledge-base CSVs.",
    )
    p.add_argument(
        "--kb",
        type=Path,
        default=Path("data/kb/codes_with_ranges.csv"),
        help="Active 2026 codes_with_ranges.csv (for the active-codes set).",
    )
    p.add_argument(
        "--changes",
        type=Path,
        default=Path("data/kb/code_changes.csv"),
        help="Annual new/revised/deleted events (2018–2026).",
    )
    p.add_argument(
        "--deleted",
        type=Path,
        default=Path("data/kb/deleted_codes.csv"),
        help="Precise deletion ledger (1990–2026).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/indices/code_history/"),
        help="Output directory; will be created if missing.",
    )
    p.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Reference date (YYYY-MM-DD) for the crosswalk JSON. "
             "Defaults to today.",
    )
    p.add_argument(
        "--parquet-engine",
        default="pyarrow",
        choices=["pyarrow", "fastparquet"],
        help="Which parquet backend to use (install the matching pkg).",
    )
    return p.parse_args()


def _parse_iso_date(s: str) -> date:
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    as_of = _parse_iso_date(args.as_of) if args.as_of else date.today()
    LOGGER.info("Reference date for crosswalk JSON: %s", as_of.isoformat())

    # Load active 2026 KB to seed the active-codes set.
    kb = CodeKnowledgeBase.from_csv(args.kb, build_index=False)
    LOGGER.info("Active 2026 KB: %d codes", len(kb))

    history = CodeHistory.from_csvs(
        changes_csv=args.changes,
        deleted_csv=args.deleted,
        active_kb_codes=kb.codes,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    # (1) Flattened event table
    events_df = history.to_events_df()
    events_path = args.out / "code_events.parquet"
    try:
        events_df.to_parquet(events_path, engine=args.parquet_engine, index=False)
    except ImportError:
        # Fallback to CSV if no parquet engine is installed.
        events_path = events_path.with_suffix(".csv")
        events_df.to_csv(events_path, index=False)
        LOGGER.warning(
            "Parquet engine %r unavailable; wrote CSV fallback to %s",
            args.parquet_engine, events_path,
        )
    LOGGER.info("Wrote %d event rows to %s", len(events_df), events_path)

    # (2) Crosswalk JSON (resolved as of ``as_of``)
    xwalk = history.crosswalk_map(as_of)
    xwalk_path = args.out / "code_crosswalk.json"
    with xwalk_path.open("w") as f:
        json.dump(
            {"as_of": as_of.isoformat(), "crosswalk": xwalk},
            f,
            indent=2,
            sort_keys=True,
        )
    LOGGER.info(
        "Wrote %d deleted→substitute mappings to %s",
        len(xwalk), xwalk_path,
    )

    # (3) Stats JSON
    stats = history.summary_stats()
    stats["n_codes_with_resolved_crosswalk_as_of"] = len(xwalk)
    stats["as_of"] = as_of.isoformat()
    stats_path = args.out / "code_history_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    LOGGER.info("Wrote stats to %s: %s", stats_path, stats)


if __name__ == "__main__":
    main()
