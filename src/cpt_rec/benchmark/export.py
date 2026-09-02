#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
DUA-safe export of benchmark metrics (``cptrec-bench-export``).

MIMIC-derived artifacts live under ``outputs/datasets/mimic_iv/``, which
``.gitignore`` excludes wholesale.  That protects the row-level files but also
hides the aggregate numbers the paper needs.  This tool moves **only** aggregate
JSON out of the restricted tree and into ``outputs/benchmark/top5/``, and it does two
independent jobs on the way:

**1. Leak guard (unconditional).**  The payload is walked leaf by leaf.  Anything
that could carry a row — a denied key name (``note_id``, ``subject_id``,
``hadm_id``, ``text``, ``snippet``, …), a long or multi-line string, or a list
longer than ``--max-list`` — is dropped and recorded.  The tool also refuses to
write into any directory named ``predictions``, because ``.gitignore`` rule 3b
re-includes those wholesale.

**2. Label-completeness contract (corpus-dependent).**  On a partially recorded
corpus, precision-family cells are *deleted*, and recall-family cells are kept
with the bias direction attached.  The rules and their reasons live in
:mod:`cpt_rec.benchmark.corpora`; this tool just applies them, so
a MIMIC micro-F1 cannot reach a paper table by being copied by hand.

The result of every run is an ``export_summary.json`` next to the exported
files, listing each dropped path and why.  That file is the audit trail: if a
reviewer asks why MIMIC has no F1 column, it is answerable from the repo.

Run
---
::

    uv run --no-sync cptrec-bench-export \\
      --corpus mimic --system m1_bm25_knn --split test \\
      --metrics-dir outputs/datasets/mimic_iv/metrics/m1_bm25_knn/test \\
      --also outputs/datasets/mimic_iv/metrics/m1_bm25_knn/test/sibling_report.json

    # the corpus card (aggregate counts only) — run once per corpus
    uv run --no-sync cptrec-bench-export --corpus mimic --manifest \\
      outputs/datasets/mimic_iv/corpus_stats.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cpt_rec.benchmark.corpora import Corpus, get_corpus

LOGGER = logging.getLogger("bench.export")

#: Key names that may never appear in an exported aggregate, at any depth.
DENIED_KEYS = frozenset(
    {
        "note_id", "note_ids", "note_index", "note_text", "notes_text", "text",
        "subject_id", "subject_ids", "hadm_id", "hadm_ids", "encounter_csn_id",
        "pat_mrn_id", "mrn", "patient_id", "snippet", "snippets", "evidence",
        "raw_text", "full_text", "report_text", "note", "rows", "records",
        "per_note", "examples_text",
    }
)

#: ``metrics.json`` / ``rank_metrics.json`` / ``sibling_report.json`` → contract family.
KIND_BY_FILE = {
    "metrics.json": "metrics",
    "rank_metrics.json": "rank",
    "sibling_report.json": "sibling",
}

#: Aux exports get their own contract family.  They are cardinality/coverage
#: aggregates over PREDICTIONS (how many codes a system offered, how often it
#: returned nothing) or over the corpus's own label distribution -- neither is
#: derived from gold matching, so partial gold does not bias them and no
#: suppression applies.  They still go through the leak guard unchanged.
AUX_KIND = "aux"

#: Aux files are written with this suffix because that is what carries them
#: across the `.gitignore` allowlist: rule 3a re-includes `outputs/**/*stats.json`
#: by NAME, wherever it sits.  An aux file called `threshold.json` would be
#: written successfully and then silently never sync.
AUX_SUFFIX = "_stats.json"


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def _leak_reason(key: str, value: Any, max_str: int, max_list: int) -> Optional[str]:
    if key.lower() in DENIED_KEYS:
        return f"denied key name '{key}'"
    if isinstance(value, str):
        if "\n" in value or "\r" in value:
            return "multi-line string"
        if len(value) > max_str:
            return f"string longer than {max_str} chars"
    if isinstance(value, list) and len(value) > max_list:
        return f"list longer than {max_list} items"
    return None


def scrub(
    payload: Any,
    corpus: Corpus,
    kind: str,
    *,
    max_str: int = 200,
    max_list: int = 64,
) -> Tuple[Any, List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Return ``(clean_payload, dropped, caveats)``.

    ``dropped`` and ``caveats`` are lists of ``{path, reason}`` records.
    """
    dropped: List[Dict[str, str]] = []
    caveats: List[Dict[str, str]] = []

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for k, v in node.items():
                child = f"{path}.{k}" if path else str(k)
                leak = _leak_reason(str(k), v, max_str, max_list)
                if leak:
                    dropped.append({"path": child, "reason": f"leak guard: {leak}"})
                    continue
                ok, note = corpus.quotable(kind, child)
                if not ok:
                    dropped.append({"path": child, "reason": f"label completeness: {note}"})
                    continue
                if note:
                    caveats.append({"path": child, "caveat": note})
                out[k] = walk(v, child)
            return out
        if isinstance(node, list):
            return [walk(v, f"{path}[]") for v in node]
        return node

    return walk(payload, ""), dropped, caveats


def _assert_safe_destination(dest: Path) -> None:
    parts = {p.lower() for p in dest.parts}
    if "predictions" in parts:
        raise SystemExit(
            f"refusing to export into {dest}: '.gitignore' rule 3b re-includes "
            "outputs/**/predictions/** wholesale, so a per-note file placed there "
            "would be committed. Pick a directory that is not named 'predictions'."
        )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_file(
    src: Path,
    dest_dir: Path,
    corpus: Corpus,
    kind: Optional[str] = None,
    *,
    max_str: int = 200,
    max_list: int = 64,
) -> Dict[str, Any]:
    kind = kind or KIND_BY_FILE.get(src.name, "metrics")
    payload = json.loads(src.read_text(encoding="utf-8"))
    clean, dropped, caveats = scrub(
        payload, corpus, kind, max_str=max_str, max_list=max_list
    )
    dest = dest_dir / src.name
    _assert_safe_destination(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info(
        "%s -> %s (%d dropped, %d caveats)", src, dest, len(dropped), len(caveats)
    )
    return {
        "source": str(src),
        "exported": str(dest),
        "contract_family": kind,
        "dropped": dropped,
        "caveats": caveats,
    }


def export_run(
    corpus: Corpus,
    system: str,
    split: str,
    metrics_dir: Optional[Path],
    extra: List[Path],
    *,
    max_str: int = 200,
    max_list: int = 64,
) -> Dict[str, Any]:
    dest_dir = corpus.export_dir(system, split)
    sources: List[Path] = []
    if metrics_dir is not None:
        for name in KIND_BY_FILE:
            p = metrics_dir / name
            if p.exists():
                sources.append(p)
        if not sources:
            raise SystemExit(
                f"{metrics_dir}: none of {sorted(KIND_BY_FILE)} found. On a partial "
                "or interrupted scoring run this directory can exist but hold "
                "nothing readable — re-score before exporting."
            )
    sources.extend(extra)

    files = [export_file(s, dest_dir, corpus, max_str=max_str, max_list=max_list)
             for s in sources]
    summary = {
        "corpus": corpus.key,
        "corpus_access": corpus.access,
        "label_completeness": corpus.label_completeness,
        "completeness_note": corpus.completeness_note,
        "system": system,
        "split": split,
        "files": files,
        "n_dropped": sum(len(f["dropped"]) for f in files),
        "n_caveats": sum(len(f["caveats"]) for f in files),
    }
    out = dest_dir / "export_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", out)
    return summary


def _aux_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise SystemExit(f"--aux {name!r}: no usable characters in the name")
    return slug + AUX_SUFFIX


def bin_code_frequency(csv_path: Path) -> Dict[str, Any]:
    """
    Reduce a ``code_frequency_stats.csv`` to a binned aggregate.

    The per-code table is NOT exported.  A full code->count table over a
    restricted corpus is the label column of that corpus re-serialised, and
    while code counts are routinely published, a complete one is closer to
    redistribution than to a statistic.  What the local repo actually needs is
    the bin definition (which codes count as head / torso / tail) and the shape
    of the distribution, so that is what crosses.
    """
    import csv as _csv

    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise SystemExit(f"{csv_path}: no rows")
    missing = {"code", "frequency", "bin"} - set(rows[0])
    if missing:
        raise SystemExit(
            f"{csv_path}: missing column(s) {sorted(missing)}; expected the "
            "code,frequency,cumulative_freq,rank,bin schema"
        )

    freqs = sorted(int(r["frequency"]) for r in rows)
    total = sum(freqs)
    bins: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        b = bins.setdefault(r["bin"], {"n_codes": 0, "occurrences": 0})
        b["n_codes"] += 1
        b["occurrences"] += int(r["frequency"])
    for b in bins.values():
        b["share_of_occurrences"] = round(b["occurrences"] / total, 6) if total else 0.0

    def pct(q: float) -> int:
        return freqs[min(len(freqs) - 1, int(q * (len(freqs) - 1)))]

    return {
        "source": csv_path.name,
        "n_codes": len(rows),
        "total_occurrences": total,
        "bins": bins,
        "frequency_percentiles": {
            "p50": pct(0.50), "p90": pct(0.90), "p95": pct(0.95), "p99": pct(0.99),
        },
        "min_frequency": freqs[0],
        "max_frequency": freqs[-1],
        "aggregation": (
            "binned aggregate of the per-code frequency table; the per-code "
            "rows are deliberately not exported"
        ),
    }


def export_aux(
    corpus: Corpus,
    name: str,
    payload: Any,
    source: str,
    *,
    max_str: int = 200,
    max_list: int = 64,
) -> Dict[str, Any]:
    """Export one aggregate that belongs to the corpus rather than to a run."""
    dest_dir = corpus.export_root / corpus.key / "aux"
    dest = dest_dir / _aux_name(name)
    _assert_safe_destination(dest)
    clean, dropped, caveats = scrub(
        payload, corpus, AUX_KIND, max_str=max_str, max_list=max_list
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rec = {
        "source": source,
        "exported": str(dest),
        "contract_family": AUX_KIND,
        "dropped": dropped,
        "caveats": caveats,
    }
    LOGGER.info("%s -> %s (%d dropped)", source, dest, len(dropped))

    # An aux export is its own audit trail; merge into a single per-corpus
    # summary so repeated runs accumulate rather than overwrite each other.
    summary_path = dest_dir / "export_summary.json"
    summary: Dict[str, Any] = {"corpus": corpus.key, "files": []}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["files"] = [f for f in summary["files"] if f["exported"] != rec["exported"]]
    summary["files"].append(rec)
    summary["corpus_access"] = corpus.access
    summary["label_completeness"] = corpus.label_completeness
    summary["n_dropped"] = sum(len(f["dropped"]) for f in summary["files"])
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rec


def export_manifest(corpus: Corpus, manifest: Path) -> Dict[str, Any]:
    dest_dir = corpus.export_root / corpus.key
    rec = export_file(manifest, dest_dir, corpus, kind="manifest",
                      max_str=400, max_list=64)
    LOGGER.info("corpus card exported to %s", dest_dir / manifest.name)
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Copy note-free aggregate metrics out of a restricted corpus tree."
    )
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--system", help="system key, e.g. m1_bm25_knn (omit with --manifest)")
    ap.add_argument("--split", help="split name, e.g. test (omit with --manifest)")
    ap.add_argument("--metrics-dir", type=Path,
                    help="directory holding metrics.json / rank_metrics.json.")
    ap.add_argument("--also", type=Path, nargs="*", default=[],
                    help="extra JSON files to export (e.g. sibling_report.json).")
    ap.add_argument("--manifest", type=Path,
                    help="export a corpus card (corpus_stats.json) instead of a run.")
    ap.add_argument("--aux", metavar="NAME",
                    help="export ONE corpus-level aggregate under "
                         "<root>/<corpus>/aux/<NAME>_stats.json instead of a "
                         "run. Pair with --file (a JSON aggregate) or with "
                         "--from-code-stats (a code_frequency_stats.csv, which "
                         "is reduced to bin counts and percentiles -- the "
                         "per-code rows never leave the restricted tree). The "
                         "_stats.json suffix is forced because that is what "
                         "carries the file across the .gitignore allowlist.")
    ap.add_argument("--file", type=Path,
                    help="the JSON aggregate to export with --aux, e.g. a "
                         "*_stats.json sidecar written beside a predictions CSV.")
    ap.add_argument("--from-code-stats", type=Path, metavar="CSV",
                    help="with --aux: read a code_frequency_stats.csv and "
                         "export its BINNED aggregate only.")
    ap.add_argument("--max-str", type=int, default=200)
    ap.add_argument("--max-list", type=int, default=64)
    ap.add_argument("--export-root", type=Path, default=None,
                    help="Write under this root instead of the corpus's own "
                         "export_root (outputs/benchmark/top5). For a PARALLEL arm — "
                         "e.g. the matched-budget B=10 tree — whose systems share "
                         "names with the default tree and must not join its "
                         "transfer correlation. Omit for default behaviour. The "
                         "same DUA suppression contract applies to any root.")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    corpus = get_corpus(args.corpus)
    if args.export_root is not None:
        # dataclasses.replace, not attribute assignment: Corpus is frozen on
        # purpose so a corpus's contract cannot be edited in place mid-run.
        # Only the destination moves; partial_gold, the suppression table and
        # every caveat travel with the object unchanged.
        corpus = dataclasses.replace(corpus, export_root=args.export_root)
        LOGGER.warning("export root overridden -> %s (parallel arm; this tree "
                       "is NOT the default-budget benchmark)", args.export_root)

    if args.manifest is not None:
        print(json.dumps(export_manifest(corpus, args.manifest), indent=2, sort_keys=True))
        return

    if args.aux is not None:
        if (args.file is None) == (args.from_code_stats is None):
            raise SystemExit(
                "--aux takes exactly one of --file (a JSON aggregate) or "
                "--from-code-stats (a code_frequency_stats.csv)"
            )
        if args.from_code_stats is not None:
            payload = bin_code_frequency(args.from_code_stats)
            source = str(args.from_code_stats)
        else:
            payload = json.loads(args.file.read_text(encoding="utf-8"))
            source = str(args.file)
        rec = export_aux(corpus, args.aux, payload, source,
                         max_str=args.max_str, max_list=args.max_list)
        print(json.dumps(rec, indent=2, sort_keys=True))
        return

    if not (args.system and args.split):
        raise SystemExit("--system and --split are required unless --manifest is given")
    summary = export_run(
        corpus, args.system, args.split, args.metrics_dir, list(args.also),
        max_str=args.max_str, max_list=args.max_list,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
