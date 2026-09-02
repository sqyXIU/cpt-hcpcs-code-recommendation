#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Collate the two-corpus benchmark into paper tables (``cptrec-bench-collate``).

Reads the exported, contract-filtered aggregates under ``outputs/benchmark/top5/``
(written by :mod:`cpt_rec.benchmark.export`) and emits the three
objects the benchmark paper needs:

``benchmark_main.md``
    One row per system, columns grouped by corpus.  A cell the corpus cannot
    support prints ``—`` with a footnote marker, and the footnote is the
    machine-recorded suppression reason — the table explains its own holes.

``benchmark_ranking.md``
    The recall-at-budget family (R@B, Coverage@B and codes *actually shown* at
    each B), which is the metric family both corpora support.

``benchmark_transfer.md``
    The headline of a two-corpus benchmark: **does the system ordering survive
    the change of corpus?**  Spearman ρ and Kendall τ between the two corpora's
    system rankings, plus the per-system rank table that produced them.  A high
    ρ says method conclusions drawn on the open corpus transfer to the private
    one; a low ρ is the more interesting finding and must be reported as such.

Every delta is read against the **measured** nondeterminism floors rather than
against zero: two systems whose gap is smaller than the floor are marked ``≈``
and must not be described as different.

Run
---
::

    uv run --no-sync cptrec-bench-collate \\
      --root outputs/benchmark/top5 --split test \\
      --out-dir outputs/benchmark/top5/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cpt_rec.benchmark.corpora import (
    PORTABLE_CORE, REGISTRY, get_corpus, system_label,
)

LOGGER = logging.getLogger("bench.collate")

#: ``(column label, artifact file, dotted path, decimals, higher_is_better)``
METRIC_SPECS: Tuple[Tuple[str, str, str, int, bool], ...] = (
    ("micro-F1",     "metrics.json",      "set.micro_f1",           4, True),
    ("micro-R",      "metrics.json",      "set.micro_recall",       4, True),
    ("R@5",          "rank_metrics.json", "recall_at.5",            4, True),
    ("R@10",         "rank_metrics.json", "recall_at.10",           4, True),
    ("Cov@5",        "rank_metrics.json", "coverage_at.5",          4, True),
    ("FamMRR",       "rank_metrics.json", "family.family_mrr",      4, True),
    ("shown@5",      "rank_metrics.json", "shown_at.5",             2, False),
    ("set size",     "metrics.json",      "set.n_pred_codes",       0, False),
    ("pool ceiling", "rank_metrics.json", "pool_ceiling",           4, True),
    ("sibling FP",   "sibling_report.json", "sibling_fp_rate",      4, False),
)

#: Measured on a byte-identical seed-42 twin pair: two runs of one recipe that
#: differ only in nondeterministic kernel scheduling.  A gap below the floor is
#: noise, not a result.  Re-measure these on your own corpus and hardware.
DEFAULT_FLOORS: Dict[str, float] = {"micro-F1": 0.0070, "R@5": 0.0099}

#: No twin-pair measurement exists for R@10, so a matched-budget transfer table
#: ranked on it has NO measured floor of its own.  Reusing the R@5 floor is a
#: deliberate, stated approximation -- pass ``--floor-transfer`` to override.
#: Never let it default silently to 0.0: that would rank noise as signal.
R10_FLOOR_NOTE = "R@10 has no measured floor; the R@5 floor is reused as a stated approximation"

#: The metric the transfer table ranks systems by, in preference order: the
#: first one available on BOTH corpora wins (micro-F1 is unavailable on a
#: partially-recorded corpus by construction).
TRANSFER_METRIC_PREFERENCE: Tuple[str, ...] = ("R@5", "R@10", "FamMRR", "micro-R")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _dig(payload: Any, dotted: str) -> Optional[Any]:
    node = payload
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def load_cell_values(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    """Return ``(values, suppressions, caveats)`` keyed by column label."""
    payloads: Dict[str, Any] = {}
    for fname in {spec[1] for spec in METRIC_SPECS}:
        p = run_dir / fname
        if p.exists():
            payloads[fname] = json.loads(p.read_text(encoding="utf-8"))

    dropped: Dict[str, str] = {}
    caveats_by_path: Dict[str, str] = {}
    summary_path = run_dir / "export_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for f in summary.get("files", []):
            for d in f.get("dropped", []):
                dropped[d["path"]] = d["reason"]
            for c in f.get("caveats", []):
                caveats_by_path[c["path"]] = c["caveat"]

    values: Dict[str, Any] = {}
    suppressed: Dict[str, str] = {}
    caveats: Dict[str, str] = {}
    for label, fname, dotted, _dec, _hib in METRIC_SPECS:
        val = _dig(payloads.get(fname, {}), dotted)
        if val is not None:
            values[label] = val
            for path, note in caveats_by_path.items():
                if dotted == path or dotted.startswith(path + "."):
                    caveats[label] = note
                    break
            continue
        for path, reason in dropped.items():
            if dotted == path or dotted.startswith(path + "."):
                suppressed[label] = reason
                break
    return values, suppressed, caveats


def discover(root: Path, split: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{corpus: {system: {values/suppressed/caveats}}}`` for one split."""
    found: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not root.exists():
        raise SystemExit(f"{root} does not exist — run cptrec-bench-export first.")
    # Registered corpora only. A budget root also holds `tables/` (the rendered
    # export), which used to sort out on its leading underscore and no longer
    # does. It yields no systems today, so nothing broke -- but a directory
    # dropped in there would become a phantom corpus, so filter on the registry.
    for corpus_dir in sorted(p for p in root.iterdir()
                             if p.is_dir() and p.name in REGISTRY):
        for system_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
            run_dir = system_dir / split
            if not run_dir.is_dir():
                continue
            values, suppressed, caveats = load_cell_values(run_dir)
            if not values and not suppressed:
                continue
            found.setdefault(corpus_dir.name, {})[system_dir.name] = {
                "values": values, "suppressed": suppressed, "caveats": caveats,
            }
    if not found:
        raise SystemExit(f"no exported runs for split '{split}' under {root}")
    return found


# ---------------------------------------------------------------------------
# rank correlation (no scipy dependency)
# ---------------------------------------------------------------------------

def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _rank_txt(r: float) -> str:
    """Average ranks from floor ties are halves; ``.0f`` rounded 2.5 to "2",
    which printed a tie as a clean win and left the delta column short by 0.5.
    """
    return f"{r:.0f}" if float(r).is_integer() else f"{r:.1f}"


def _delta_txt(d: float) -> str:
    return f"{d:+.0f}" if float(d).is_integer() else f"{d:+.1f}"


def _ranks_with_floor(xs: Sequence[float], floor: float) -> List[float]:
    """Descending ranks in which gaps at or below ``floor`` are ties.

    The paper pre-registers this rule -- "ties within the measured Recall@5
    floor are assigned average rank and described as indistinguishable" -- but
    ``render_transfer`` used raw ``_ranks``, so the exported headline was the
    NO-FLOOR number.  On 2026-08-27 that put rho = 0.5 in benchmark_transfer.md
    against the manuscript's rho = 0.0, off a 0.21-point MIMIC gap that the
    floor calls a tie.

    Grouping is anchored, not chained: each group is opened by its highest
    remaining value and admits only values within ``floor`` OF THAT ANCHOR.
    Chaining would let a > b > c collapse into one tie whenever a - b and
    b - c each clear the floor even though a - c does not.
    """
    order = sorted(range(len(xs)), key=lambda i: -xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        anchor = xs[order[i]]
        while j + 1 < len(order) and anchor - xs[order[j + 1]] <= floor:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return round(num / den, 4) if den else None


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    if n < 3:
        return None
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    total = con + dis
    return round((con - dis) / total, 4) if total else None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _fmt(label: str, val: Any, dec: int) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.{dec}f}" if dec else f"{int(round(val)):,}"
    return str(val)


def render_main(found: Dict[str, Dict[str, Dict[str, Any]]], split: str,
                floors: Dict[str, float]) -> str:
    corpora = sorted(found)
    systems = sorted({s for c in found.values() for s in c},
                     key=lambda s: (s not in PORTABLE_CORE, s))
    labels = [spec[0] for spec in METRIC_SPECS]
    dec = {spec[0]: spec[3] for spec in METRIC_SPECS}
    hib = {spec[0]: spec[4] for spec in METRIC_SPECS}

    # per (corpus, label) best value, for the ≈ / bold marking
    best: Dict[Tuple[str, str], float] = {}
    for c in corpora:
        for label in labels:
            vals = [d["values"][label] for d in found[c].values()
                    if isinstance(d["values"].get(label), (int, float))]
            if vals and hib[label]:
                best[(c, label)] = max(vals)

    notes: Dict[str, str] = {}
    lines: List[str] = []
    lines.append(f"# Benchmark — main table (split: `{split}`)")
    lines.append("")
    lines.append("`—ⁿ` = the corpus's label completeness does not support that cell; "
                 "footnote *n* gives the recorded reason. "
                 "`≈` = within the measured nondeterminism floor of the best value "
                 "in that column, i.e. not distinguishable. "
                 "On a partially-recorded corpus **`micro-R` is never bolded**: it is "
                 "unbudgeted, so it rises with set size, and precision — the term that "
                 "would charge a system for that volume — is suppressed. Read it beside "
                 "`set size`, and rank on the budgeted `R@B` columns.")
    lines.append("")
    for c in corpora:
        corpus = get_corpus(c) if c in ("vumc", "mimic") else None
        present = [l for l in labels
                   if any(l in d["values"] or l in d["suppressed"] for d in found[c].values())]
        head = f"## {corpus.display_name if corpus else c}"
        if corpus:
            head += f" — {corpus.label_completeness}, {corpus.split_policy.split('(')[0].strip()}"
        lines.append(head)
        lines.append("")
        lines.append("| system | " + " | ".join(present) + " |")
        lines.append("|---|" + "---|" * len(present))
        for s in sorted(found[c]):
            d = found[c][s]
            cells = []
            for label in present:
                if label in d["values"]:
                    txt = _fmt(label, d["values"][label], dec[label])
                    b = best.get((c, label))
                    v = d["values"][label]
                    if b is not None and isinstance(v, (int, float)):
                        floor = floors.get(label, 0.0)
                        # micro-R is UNBUDGETED: it rises with set size, and on
                        # a partially-recorded corpus precision is suppressed,
                        # so nothing in the row charges a system for emitting
                        # more.  Bolding it rewarded dumping -- the 2026-08-27
                        # MIMIC panel bolded m2_label_attention's 0.6758 over
                        # m1_bm25_knn's 0.4687 off 24,834 predicted codes
                        # against 5,662, while M1 led every budgeted metric.
                        # Print the value, never crown it.
                        unbudgeted = (label == "micro-R"
                                      and corpus is not None
                                      and corpus.partial_gold)
                        if unbudgeted:
                            pass        # print the value, never crown it
                        elif v == b:
                            txt = f"**{txt}**"
                        elif b - v <= floor:
                            txt = f"{txt} ≈"
                elif label in d["suppressed"]:
                    reason = d["suppressed"][label]
                    if reason not in notes:
                        notes[reason] = str(len(notes) + 1)
                    txt = f"—<sup>{notes[reason]}</sup>"
                else:
                    txt = ""
                cells.append(txt)
            tag = "" if s in PORTABLE_CORE else " *(corpus-specific)*"
            lab = system_label(s)
            name = f"**{lab}** `{s}`" if lab != s else f"`{s}`"
            lines.append(f"| {name}{tag} | " + " | ".join(cells) + " |")
        lines.append("")
    if notes:
        lines.append("**Footnotes**")
        lines.append("")
        for reason, n in sorted(notes.items(), key=lambda kv: int(kv[1])):
            lines.append(f"{n}. {reason}")
        lines.append("")
    return "\n".join(lines)


def render_transfer(found: Dict[str, Dict[str, Dict[str, Any]]], split: str,
                    floors: Dict[str, float],
                    force_metric: Optional[str] = None) -> str:
    corpora = sorted(found)
    lines = [f"# Benchmark — cross-corpus transfer (split: `{split}`)", ""]
    if len(corpora) < 2:
        lines += ["Only one corpus exported; the transfer table needs two.", ""]
        return "\n".join(lines)

    a, b = corpora[0], corpora[1]
    shared = sorted(set(found[a]) & set(found[b]))
    if len(shared) < 3:
        lines += [
            f"Systems on both corpora: {len(shared)} "
            f"({', '.join(shared) if shared else 'none'}). "
            "A rank correlation needs at least 3 — run more of the portable core.",
            "",
        ]
        return "\n".join(lines)

    # ``force_metric`` exists for the matched-budget arm: at B=5 the
    # LLM rows show 1.3-2.1 codes, so R@5 collapses to their plain recall and
    # the ranking is partly a budget-compliance ranking.  Ranking at B=10 is a
    # DIFFERENT question, not a better answer to the same one -- the caller is
    # required to say which, and the header below records the choice so a
    # reader can never mistake one table for the other.
    if force_metric is not None:
        if not all(force_metric in found[c][s]["values"]
                   for c in (a, b) for s in shared):
            missing = sorted(s for s in shared
                             if any(force_metric not in found[c][s]["values"]
                                    for c in (a, b)))
            lines += [f"`{force_metric}` was requested but is missing for: "
                      f"{', '.join(missing)}. No transfer table.", ""]
            return "\n".join(lines)
        metric = force_metric
    else:
        metric = next(
            (m for m in TRANSFER_METRIC_PREFERENCE
             if all(m in found[c][s]["values"] for c in (a, b) for s in shared)),
            None,
        )
    if metric is None:
        lines += ["No metric is available on both corpora for every shared system.", ""]
        return "\n".join(lines)

    xa = [float(found[a][s]["values"][metric]) for s in shared]
    xb = [float(found[b][s]["values"][metric]) for s in shared]
    # the pre-registered rule ranks with the measured nondeterminism floor;
    # the raw ordering is reported below it as a sensitivity, never as the
    # headline (see _ranks_with_floor)
    floor = floors.get(metric, 0.0)
    ra, rb = _ranks_with_floor(xa, floor), _ranks_with_floor(xb, floor)

    # Systems that exist on exactly one corpus are dropped from the correlation
    # silently by construction (`shared` is an intersection).  That silence is
    # the problem: the shared-set size is the single largest lever on rho we
    # have measured, larger than the budget change this table exists to report.
    # Name what was excluded, in the table itself, so a reader cannot compare
    # an n=4 rho against an n=5 rho without seeing why they differ.
    _one_sided = sorted(
        (set(found[a]) | set(found[b])) - set(shared)
    )

    lines += [
        (f"Ranking metric: **{metric}** — set explicitly by `--transfer-metric`, "
         f"NOT the `TRANSFER_METRIC_PREFERENCE` default. Report this beside the "
         f"default-budget table, never instead of it. "
         f"**Ranked over {len(shared)} shared system(s)"
         + (f"; {', '.join('`'+s+'`' for s in sorted(_one_sided))} "
            f"appear(s) on one corpus only and is EXCLUDED.**"
            if _one_sided else ".**")
         + (" The shared-set size is not a footnote: on the 2026-08-27 return, "
            "dropping one system moved \u03c1 from +0.50 to 0.00 on identical data. "
            "The budget is a real lever too -- over the same five shared systems on "
            "that return, matching the request at ten codes moved \u03c1 from 0.3591 "
            "to 0.5263 at R@5 and from 0.5000 to 0.6156 at R@10 -- so a \u03c1 is "
            "comparable to another only when BOTH the shared set and the budget match."
            if _one_sided else "")
         if force_metric is not None else
         f"Ranking metric: **{metric}** — the first metric in "
         f"`TRANSFER_METRIC_PREFERENCE` available on both corpora for all "
         f"{len(shared)} shared systems."),
        "",
        f"| system | {a} {metric} | rank | {b} {metric} | rank | Δrank |",
        "|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(shared):
        lines.append(
            f"| `{s}` | {xa[i]:.4f} | {_rank_txt(ra[i])} | {xb[i]:.4f} "
            f"| {_rank_txt(rb[i])} | {_delta_txt(rb[i] - ra[i])} |"
        )
    rho, tau = spearman(ra, rb), kendall_tau([-r for r in ra], [-r for r in rb])
    rho_raw = spearman([-v for v in xa], [-v for v in xb])
    tau_raw = kendall_tau(xa, xb)
    lines += [
        "",
        (f"- Spearman ρ = **{rho}**  (ranks with the {metric} floor of "
         f"{floor:.4f}, REUSED from R@5 — no twin-pair measurement of {metric} "
         f"nondeterminism exists, so this floor is a stated approximation, not "
         f"a pre-registered value)"
         if force_metric is not None and metric != "R@5" else
         f"- Spearman ρ = **{rho}**  (ranks with the pre-registered {metric} "
         f"floor of {floor:.4f}; equal ranks above are ties under that floor)"),
        f"- Kendall τ = **{tau}**  (same floor)",
        f"- Sensitivity, ranking the unrounded point estimates with NO floor: "
        f"ρ = {rho_raw}, τ = {tau_raw}. Report this only as a sensitivity — "
        f"the floor-aware pair above is the pre-registered headline.",
        "",
        "Read this as the benchmark's headline: a high ρ means a method "
        "conclusion drawn on the open corpus transfers to the private one, so "
        "MIMIC-IV is a usable proxy for method selection. A low ρ is the more "
        "consequential finding — it says the field's open-corpus results do not "
        "predict behaviour on real billing data, and it must be reported as "
        "prominently as a high one.",
        "",
    ]
    return "\n".join(lines)


def render_ranking(found: Dict[str, Dict[str, Dict[str, Any]]], split: str) -> str:
    lines = [f"# Benchmark — recall at budget (split: `{split}`)", "",
             "The one metric family both corpora support. On a partially "
             "recorded corpus these are **lower bounds**: a correct code the "
             "source table never recorded still occupies a top-*B* slot.", ""]
    for c in sorted(found):
        corpus = get_corpus(c) if c in ("vumc", "mimic") else None
        lines.append(f"## {corpus.display_name if corpus else c}")
        lines.append("")
        lines.append("| system | R@5 | R@10 | Cov@5 | FamMRR | shown@5 |")
        lines.append("|---|---|---|---|---|---|")
        for s in sorted(found[c]):
            v = found[c][s]["values"]
            lines.append(
                f"| `{s}` | {_fmt('R@5', v.get('R@5'), 4)} | {_fmt('R@10', v.get('R@10'), 4)} "
                f"| {_fmt('Cov@5', v.get('Cov@5'), 4)} | {_fmt('FamMRR', v.get('FamMRR'), 4)} "
                f"| {_fmt('shown@5', v.get('shown@5'), 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_long_csv(found: Dict[str, Dict[str, Dict[str, Any]]], split: str, out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "system", "split", "metric", "value", "status", "note"])
        for c in sorted(found):
            for s in sorted(found[c]):
                d = found[c][s]
                for label, val in sorted(d["values"].items()):
                    w.writerow([c, s, split, label, val,
                                "caveat" if label in d["caveats"] else "ok",
                                d["caveats"].get(label, "")])
                for label, reason in sorted(d["suppressed"].items()):
                    w.writerow([c, s, split, label, "", "suppressed", reason])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Collate exported benchmark metrics.")
    ap.add_argument("--root", type=Path, default=Path("outputs/benchmark/top5"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark/top5/tables"))
    ap.add_argument("--floor-micro-f1", type=float, default=DEFAULT_FLOORS["micro-F1"])
    ap.add_argument("--floor-recall-at-5", type=float, default=DEFAULT_FLOORS["R@5"])
    ap.add_argument("--transfer-metric", default=None,
                    choices=("R@5", "R@10", "FamMRR", "micro-R"),
                    help="Rank the transfer table on this metric instead of the "
                         "first available in TRANSFER_METRIC_PREFERENCE. For the "
                         "matched-budget arm. Omit for byte-identical default "
                         "behaviour.")
    ap.add_argument("--transfer-out", type=Path, default=None,
                    help="Write the transfer table here instead of "
                         "<out-dir>/benchmark_transfer.md, so a matched-budget "
                         "arm never overwrites the default-budget table.")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    floors = {"micro-F1": args.floor_micro_f1, "R@5": args.floor_recall_at_5}
    if args.transfer_metric == "R@10":
        floors["R@10"] = args.floor_recall_at_5
        LOGGER.warning("%s (using %.4f)", R10_FLOOR_NOTE, args.floor_recall_at_5)

    found = discover(args.root, args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "benchmark_main.md": render_main(found, args.split, floors),
        "benchmark_ranking.md": render_ranking(found, args.split),
        "benchmark_transfer.md": render_transfer(found, args.split, floors,
                                                 args.transfer_metric),
    }
    if args.transfer_out is not None:
        outputs.pop("benchmark_transfer.md")
        args.transfer_out.parent.mkdir(parents=True, exist_ok=True)
        args.transfer_out.write_text(
            render_transfer(found, args.split, floors, args.transfer_metric) + "\n",
            encoding="utf-8")
        LOGGER.info("wrote %s", args.transfer_out)
    for name, text in outputs.items():
        (args.out_dir / name).write_text(text + "\n", encoding="utf-8")
        LOGGER.info("wrote %s", args.out_dir / name)
    write_long_csv(found, args.split, args.out_dir / "benchmark_summary.csv")
    LOGGER.info("wrote %s", args.out_dir / "benchmark_summary.csv")

    n = sum(len(v) for v in found.values())
    print(f"collated {n} run(s) over {len(found)} corpus/corpora for split '{args.split}'")
    print(outputs.get("benchmark_transfer.md",
                      args.transfer_out.read_text() if args.transfer_out else ""))


if __name__ == "__main__":  # pragma: no cover
    main()
