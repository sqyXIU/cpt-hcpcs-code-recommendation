#!/usr/bin/env python3
# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Build the MIMIC-IV half of the benchmark (``cptrec-bench-build-mimic``).

Turns the two PhysioNet tables into the repo's own split schema so that every
method in ``baselines/`` and ``pipeline/`` runs on MIMIC-IV with no
code change — only different ``--notes`` / ``--train-stats`` paths.

What it does
------------
1. **Two filters, at two levels.**  *Encounter* level: keep admissions with at
   least one CPT-I surgical code (``10004``–``69990`` by default).  *Code*
   level: drop the non-procedural codes from the gold set of the encounters
   that survive (``--code-filter``, default ``procedural``).

   The second filter is the one that makes MIMIC the same task as VUMC.
   ``hcpcsevents`` is an admission-level *billing* table, not a procedure log:
   on the 21,554 surgery-coded admissions only 61.6% of the (deduplicated)
   code instances are surgical, and 20,702 of the remaining 20,876 are six
   codes — ``G0378`` observation-per-hour, ``99218``/``99219``/``99220``
   observation E/M, ``G0379``, ``99291``.  None of them is inferable from a
   procedure narrative, so scoring against them measures the billing table,
   not the model.  Dropping them lifts the surgical share to 99.5% and takes
   codes/encounter from 2.52 to 1.56 — against VUMC's own 93.8% and 1.82.
   The 174 non-surgical instances that survive (cardiac catheterisation, PCI,
   arthroscopy, screening colonoscopy) are procedures, and they are the same
   kind of tail VUMC's gold carries.
2. **One row per encounter.** ``hcpcsevents`` is joined to ``discharge`` on
   ``hadm_id``; codes are pipe-joined in sorted order, exactly like
   ``proc_codes`` elsewhere in the repo.
3. **Patient-disjoint split.** MIMIC shifts dates *per patient*, so a temporal
   split carries no meaning and a random split leaks a patient's other
   admissions across the boundary.  The split is drawn on ``subject_id``.
4. **MIMIC's own frequency bins.** head/torso/tail are recomputed from MIMIC's
   train split — the VUMC bins describe a different code distribution and must
   never be reused here.
5. **A manifest of aggregates.** Including the *measured* label-completeness
   ratio, which is the number the paper's MIMIC caveat rests on.

Safety
------
Everything is written under ``outputs/datasets/mimic_iv/``, which ``.gitignore``
excludes wholesale (``/data/mimic_iv/``).  Nothing in this module prints or
logs note text, and the manifest holds counts only — it is the one artifact
that may be copied out (via ``cptrec-bench-export --manifest``).

Run
---
::

    uv run --no-sync cptrec-bench-build-mimic \\
      --hcpcs    data/mimic_iv/hcpcsevents.csv \\
      --discharge data/mimic_iv/discharge.csv.gz \\
      --out-dir  outputs/datasets/mimic_iv

    # schema-only check, reads the first chunk of each file:
    uv run --no-sync cptrec-bench-build-mimic ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from cpt_rec.common.preprocess.label_normalizer import (
    compute_code_frequency_stats,
)

LOGGER = logging.getLogger("bench.build_mimic")

OUT_COLS: Tuple[str, ...] = ("note_id", "hadm_id", "subject_id", "note_text", "proc_codes")
_CODE_RE = re.compile(r"^[0-9]{5}$")

#: CPT-I Evaluation & Management.  Office/hospital visits, observation stays,
#: critical care, consults — a level of service, never a procedure.  MIMIC's
#: observation block (``99218``–``99220``) lives here.
EM_RANGE: Tuple[int, int] = (99202, 99499)

#: HCPCS-II codes that are E/M-equivalent facility services rather than
#: procedures.  ``G0378`` alone is 9,236 instances on MIMIC's surgery-coded
#: admissions; the rest are listed because they are the same class of code and
#: a future MIMIC release may surface them.
NONPROCEDURAL_HCPCS2: frozenset = frozenset({
    "G0378",   # hospital observation service, per hour
    "G0379",   # direct admission of patient for hospital observation care
    "G0380", "G0381", "G0382", "G0383", "G0384",   # type B ED visits
    "G0425", "G0426", "G0427",                     # ED/inpatient telehealth consults
    "G0463",   # hospital outpatient clinic visit for assessment of a patient
    "G0316", "G0317", "G0318", "G2212",            # prolonged E/M add-ons
})

CODE_FILTERS: Tuple[str, ...] = ("procedural", "surgery", "none")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def is_surgical(code: str, lo: int, hi: int) -> bool:
    """CPT-I surgery section: five digits inside ``[lo, hi]``."""
    return bool(_CODE_RE.match(code)) and lo <= int(code) <= hi


def is_nonprocedural(code: str) -> bool:
    """True for a code that describes a *level of service*, not a procedure.

    Two classes, both un-inferable from a procedure narrative: CPT-I E/M
    (:data:`EM_RANGE`, which contains MIMIC's ``99218``–``99220`` observation
    block) and the E/M-equivalent HCPCS-II facility codes in
    :data:`NONPROCEDURAL_HCPCS2`.
    """
    if code in NONPROCEDURAL_HCPCS2:
        return True
    return bool(_CODE_RE.match(code)) and EM_RANGE[0] <= int(code) <= EM_RANGE[1]


def keeps_code(code: str, code_filter: str, surg_lo: int, surg_hi: int) -> bool:
    """Does ``code`` belong in the gold set under ``code_filter``?

    ``procedural`` (default) drops E/M and E/M-equivalent facility codes and
    keeps every procedure, which is what VUMC's billed gold looks like.
    ``surgery`` keeps the CPT-I surgery section only — stricter than VUMC,
    useful as a robustness arm.  ``none`` is the pre-2026-08-24 behaviour:
    every code on a surgery-coded admission, billing noise included.
    """
    if code_filter == "none":
        return True
    if code_filter == "surgery":
        return is_surgical(code, surg_lo, surg_hi)
    if code_filter == "procedural":
        return not is_nonprocedural(code)
    raise ValueError(f"unknown --code-filter {code_filter!r}; choose from {CODE_FILTERS}")


def _code_section(code: str) -> str:
    """Coarse HCPCS bucket, for the corpus card only."""
    if not _CODE_RE.match(code):
        return "hcpcs2"
    n = int(code)
    if 10004 <= n <= 69990:
        return "surgery"
    if 99202 <= n <= 99499:
        return "em"
    if 70010 <= n <= 79999:
        return "radiology"
    if 80047 <= n <= 89398:
        return "pathology"
    return "medicine"


def _mix(codes: Sequence[str]) -> Dict[str, int]:
    """Coarse section histogram, for the corpus card."""
    return {k: int(v) for k, v in sorted(Counter(_code_section(c) for c in codes).items())}


def load_codes(
    hcpcs_csv: Path,
    surg_lo: int,
    surg_hi: int,
    code_col: str = "hcpcs_cd",
    code_filter: str = "procedural",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return ``(per-encounter code frame, aggregate stats)`` for surgery encounters.

    Two filters run here and they are *not* the same filter.  The surgery range
    picks the **encounters**: an admission is in the corpus if it carries at
    least one surgical code.  ``code_filter`` then picks the **codes** that
    become that encounter's gold set — without it the gold keeps every charge
    on the admission, and on MIMIC 38% of those are observation and E/M codes
    that no procedure narrative can support.
    """
    hc = pd.read_csv(hcpcs_csv, dtype=str).fillna("")
    for required in (code_col, "hadm_id", "subject_id"):
        if required not in hc.columns:
            raise SystemExit(
                f"{hcpcs_csv}: missing column '{required}' (found: {list(hc.columns)})"
            )
    hc[code_col] = hc[code_col].str.strip().str.upper()
    hc = hc[hc[code_col] != ""]

    # --- encounter level: an admission needs one surgical code to qualify ---
    surg_mask = hc[code_col].map(lambda c: is_surgical(c, surg_lo, surg_hi))
    surg_enc: Set[str] = set(hc.loc[surg_mask, "hadm_id"])
    on_surg = hc[hc.hadm_id.isin(surg_enc)].drop_duplicates(["hadm_id", code_col])

    # --- code level: what actually lands in proc_codes ----------------------
    keep_mask = on_surg[code_col].map(
        lambda c: keeps_code(c, code_filter, surg_lo, surg_hi)
    )
    kept = on_surg[keep_mask]
    dropped = Counter(on_surg.loc[~keep_mask, code_col])

    codes = (
        kept.groupby("hadm_id")
        .agg(
            proc_codes=(code_col, lambda x: "|".join(sorted(set(x)))),
            subject_id=("subject_id", "first"),
        )
        .reset_index()
    )
    # `surgery` can in principle empty an encounter only if surg_lo/surg_hi were
    # widened for the encounter rule and narrowed for the code rule; guard anyway.
    emptied = len(surg_enc) - len(codes)

    kept_mix = _mix(kept[code_col])
    n_kept = int(len(kept))
    stats = {
        "hcpcs_rows": int(len(hc)),
        "hcpcs_admissions": int(hc.hadm_id.nunique()),
        "code_filter": code_filter,
        "code_instance_mix": _mix(hc[code_col]),
        "code_instance_mix_on_surgery_encounters": _mix(on_surg[code_col]),
        "code_instance_mix_kept": kept_mix,
        "code_instances_before_filter": int(len(on_surg)),
        "code_instances_kept": n_kept,
        "dropped_code_instances": int(len(on_surg) - n_kept),
        # small and auditable by construction — `procedural` drops six codes on
        # MIMIC-IV v3.1; if this list ever grows, read it before trusting the run.
        "dropped_codes": {c: int(n) for c, n in dropped.most_common(50)},
        "surgical_share_kept": round(kept_mix.get("surgery", 0) / n_kept, 4) if n_kept else None,
        "codes_per_encounter_before_filter": (
            round(len(on_surg) / len(surg_enc), 4) if surg_enc else None
        ),
        "surgery_encounters": int(len(surg_enc)),
        "encounters_emptied_by_filter": int(emptied),
        "distinct_surgery_codes": int(
            on_surg.loc[
                on_surg[code_col].map(lambda c: is_surgical(c, surg_lo, surg_hi)), code_col
            ].nunique()
        ),
        "surgery_code_range": [surg_lo, surg_hi],
    }
    return codes, stats


def load_notes(
    discharge_csv: Path,
    keep_hadm: Set[str],
    chunksize: int,
    note_select: str,
    limit_chunks: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Stream ``discharge.csv.gz`` and keep one note per wanted ``hadm_id``."""
    frames: List[pd.DataFrame] = []
    n_rows = 0
    n_admissions_seen: Set[str] = set()
    for i, chunk in enumerate(
        pd.read_csv(
            discharge_csv,
            usecols=["note_id", "hadm_id", "text"],
            dtype=str,
            chunksize=chunksize,
        )
    ):
        n_rows += len(chunk)
        n_admissions_seen.update(chunk.hadm_id.dropna())
        hit = chunk[chunk.hadm_id.isin(keep_hadm)]
        if len(hit):
            frames.append(hit)
        if limit_chunks is not None and i + 1 >= limit_chunks:
            break

    if not frames:
        raise SystemExit(
            "no discharge note matched a surgery-coded hadm_id — check that "
            "discharge.csv.gz and hcpcsevents.csv come from the same MIMIC release"
        )
    notes = pd.concat(frames, ignore_index=True)
    notes["text"] = notes["text"].fillna("")

    if note_select == "longest":
        notes["_len"] = notes.text.str.len()
        notes = notes.sort_values(["hadm_id", "_len", "note_id"],
                                  ascending=[True, False, True]).drop(columns="_len")
    else:                                     # "first" — deterministic by note_id
        notes = notes.sort_values(["hadm_id", "note_id"])
    n_before = len(notes)
    notes = notes.drop_duplicates("hadm_id", keep="first")

    stats = {
        "discharge_rows": int(n_rows),
        "discharge_admissions": int(len(n_admissions_seen)),
        "notes_matched": int(n_before),
        "notes_after_one_per_encounter": int(len(notes)),
        "note_select": note_select,
    }
    return notes, stats


def patient_disjoint_split(
    df: pd.DataFrame,
    patient_col: str,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> pd.Series:
    """Assign every row to train/val/test with no patient crossing a boundary."""
    rng = np.random.default_rng(seed)
    subjects = np.asarray(df[patient_col].astype(str).unique(), dtype=object)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test: n_test + n_val])
    return df[patient_col].astype(str).map(
        lambda s: "test" if s in test_s else ("val" if s in val_s else "train")
    )


def split_summary(sub: pd.DataFrame, patient_col: str) -> Dict[str, object]:
    """Aggregate counts for one split.  Empty splits are legal (tiny fixtures)."""
    if not len(sub):
        return {k: 0 for k in (
            "n_notes", "n_patients", "n_code_instances", "n_distinct_codes",
            "codes_per_note_mean", "codes_per_note_max",
            "note_tokens_mean", "note_tokens_p95",
        )}
    per_note = sub.proc_codes.fillna("").map(lambda s: len([c for c in s.split("|") if c]))
    tok = sub.note_text.fillna("").str.split().str.len()
    return {
        "n_notes": int(len(sub)),
        "n_patients": int(sub[patient_col].nunique()),
        "n_code_instances": int(per_note.sum()),
        "n_distinct_codes": int(len({c for s in sub.proc_codes.fillna("")
                                     for c in s.split("|") if c})),
        "codes_per_note_mean": round(float(per_note.mean()), 4),
        "codes_per_note_max": int(per_note.max()),
        "note_tokens_mean": round(float(tok.mean()), 1),
        "note_tokens_p95": int(tok.quantile(0.95)),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(
    hcpcs_csv: Path,
    discharge_csv: Path,
    out_dir: Path,
    *,
    surg_lo: int = 10004,
    surg_hi: int = 69990,
    code_filter: str = "procedural",
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    min_note_tokens: int = 100,
    chunksize: int = 100_000,
    note_select: str = "first",
    limit_chunks: Optional[int] = None,
    write: bool = True,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, code_stats = load_codes(hcpcs_csv, surg_lo, surg_hi, code_filter=code_filter)
    LOGGER.info(
        "surgery-coded encounters: %d | --code-filter %s dropped %d of %d code "
        "instances (%d distinct); surgical share of the gold now %s",
        code_stats["surgery_encounters"], code_filter,
        code_stats["dropped_code_instances"], code_stats["code_instances_before_filter"],
        len(code_stats["dropped_codes"]), code_stats["surgical_share_kept"],
    )

    notes, note_stats = load_notes(
        discharge_csv, set(codes.hadm_id), chunksize, note_select, limit_chunks
    )
    LOGGER.info("discharge notes matched: %d", note_stats["notes_matched"])

    df = notes.merge(codes, on="hadm_id", how="inner").rename(columns={"text": "note_text"})
    n_joined = len(df)

    short = df.note_text.str.split().str.len() < min_note_tokens
    n_short = int(short.sum())
    df = df[~short]
    df = df[list(OUT_COLS)].reset_index(drop=True)

    part = patient_disjoint_split(df, "subject_id", seed, val_frac, test_frac)

    manifest: Dict[str, object] = {
        "corpus": "mimic",
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "hcpcs": str(hcpcs_csv),
            "discharge": str(discharge_csv),
            "note": "PhysioNet MIMIC-IV under a DUA; no row-level artifact leaves this tree",
        },
        "recipe": {
            "surgery_range": [surg_lo, surg_hi],
            "code_filter": code_filter,
            "code_filter_note": (
                "surgery_range selects ENCOUNTERS; code_filter selects the CODES that "
                "become the gold set.  'procedural' drops CPT-I E/M 99202-99499 and the "
                "E/M-equivalent HCPCS-II facility codes (G0378 observation-per-hour and "
                "kin) — a level of service is not inferable from a procedure narrative."
            ),
            "note_select": note_select,
            "min_note_tokens": min_note_tokens,
            "split_policy": "patient-disjoint on subject_id",
            "split_fracs": {"train": round(1 - val_frac - test_frac, 4),
                            "val": val_frac, "test": test_frac},
            "seed": seed,
        },
        "codes": code_stats,
        "notes": note_stats,
        "join": {
            "encounters_joined": int(n_joined),
            "dropped_short_notes": n_short,
            "encounters_kept": int(len(df)),
        },
        "splits": {},
    }

    # the number the MIMIC caveat rests on — measured, not quoted.
    #
    # Under --dry-run the two sides come from DIFFERENT populations: the
    # denominator counts admissions seen in the first `limit_chunks` discharge
    # chunks, while the numerator is a full pass over hcpcsevents (which is not
    # chunked). Their ratio is not a partial estimate of the real value, it is a
    # category error -- on the 2026-08-26 build it read 0.7268 against a true
    # 0.4381, i.e. 66% high, on the one number the corpus's "no precision/F1"
    # caveat rests on. So refuse to compute it rather than emit a plausible
    # wrong one; a truncated read gets a null and an explicit reason.
    d_adm = note_stats["discharge_admissions"]
    h_adm = code_stats["hcpcs_admissions"]
    partial = limit_chunks is not None
    manifest["label_completeness"] = {
        "noted_admissions": d_adm,
        "admissions_with_any_hcpcs_row": h_adm,
        "covered_fraction": None if partial else (round(h_adm / d_adm, 4) if d_adm else None),
        "verdict": "not-measured (truncated read)" if partial else "partially-recorded",
        "consequence": (
            "--dry-run read only the first {} discharge chunk(s), so noted_admissions is "
            "truncated while admissions_with_any_hcpcs_row is a full pass; the ratio would "
            "compare different populations and is withheld. Re-run without --dry-run to "
            "measure it.".format(limit_chunks)
            if partial else
            "an absent code does not mean an absent procedure; precision-family "
            "metrics are not comparable across systems on this corpus"
        ),
    }
    if partial:
        manifest["label_completeness"]["noted_admissions_is_truncated"] = True
        # Every note-derived field below (join counts, split sizes, token stats)
        # is computed on the truncated read too. Say so once, at the top level,
        # so no field of a --dry-run manifest is quotable by accident.
        manifest["truncated_read"] = {
            "dry_run": True,
            "discharge_chunks_read": limit_chunks,
            "affects": ["notes", "join", "splits", "label_completeness"],
            "note": "smoke test only — schema and plumbing, not corpus statistics",
        }

    for name in ("train", "val", "test"):
        sub = df[part == name]
        manifest["splits"][name] = split_summary(sub, "subject_id")
        if write:
            sub.to_csv(out_dir / f"{name}.csv", index=False)
            LOGGER.info("wrote %s (%d notes)", out_dir / f"{name}.csv", len(sub))

    if write:
        stats_path = out_dir / "code_frequency_stats.csv"
        compute_code_frequency_stats(
            train_csv=out_dir / "train.csv", output_csv=stats_path, code_col="proc_codes"
        )
        bins = pd.read_csv(stats_path, dtype=str)
        manifest["frequency_bins"] = {
            b: int((bins["bin"] == b).sum()) for b in ("head", "torso", "tail")
        }
        (out_dir / "corpus_stats.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOGGER.info("wrote %s", out_dir / "corpus_stats.json")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build repo-schema MIMIC-IV splits for the two-corpus benchmark."
    )
    ap.add_argument("--hcpcs", type=Path, default=Path("data/mimic_iv/hcpcsevents.csv"))
    ap.add_argument("--discharge", type=Path, default=Path("data/mimic_iv/discharge.csv.gz"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/datasets/mimic_iv"))
    ap.add_argument("--surgery-range", nargs=2, type=int, default=(10004, 69990),
                    metavar=("LO", "HI"),
                    help="CPT-I surgery section; selects which ENCOUNTERS enter the corpus.")
    ap.add_argument("--code-filter", choices=CODE_FILTERS, default="procedural",
                    help="which CODES enter an encounter's gold set. 'procedural' (default) "
                         "drops E/M 99202-99499 and E/M-equivalent HCPCS-II facility codes "
                         "such as G0378; 'surgery' keeps the surgery section only; 'none' "
                         "keeps every charge on the admission (pre-2026-08-24 behaviour).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--min-note-tokens", type=int, default=100,
                    help="mirrors the VUMC normalizer's --min-tokens (default 100).")
    ap.add_argument("--note-select", choices=("first", "longest"), default="first",
                    help="which discharge row to keep when an encounter has several.")
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--dry-run", action="store_true",
                    help="read only --dry-run-chunks of the note file and write nothing.")
    ap.add_argument("--dry-run-chunks", type=int, default=2)
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    manifest = build(
        hcpcs_csv=args.hcpcs,
        discharge_csv=args.discharge,
        out_dir=args.out_dir,
        surg_lo=args.surgery_range[0],
        surg_hi=args.surgery_range[1],
        code_filter=args.code_filter,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_note_tokens=args.min_note_tokens,
        chunksize=args.chunksize,
        note_select=args.note_select,
        limit_chunks=args.dry_run_chunks if args.dry_run else None,
        write=not args.dry_run,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
