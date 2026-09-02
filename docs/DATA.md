# Data

Nothing in this repository is clinical data, and nothing in it is a licensed
code descriptor. This page says what you need, where it comes from, and what
you can substitute.

| what | shipped here? | how you get it |
|---|---|---|
| HCPCS Level II descriptors | **yes** — `data/kb/hcpcs_level2_public.csv`, 7,379 rows | CMS public domain, already in the repo |
| CPT (Level I) descriptors | no | licensed from the AMA; merged in with `scripts/build_kb.py merge` |
| NCCI PTP / MUE / AOC edits | no | free download from CMS; unpacked with `scripts/setup_ncci.py` |
| MIMIC-IV discharge summaries + `hcpcsevents` | no | PhysioNet, credentialed access under a DUA |
| VUMC operative notes | no, and never | private institutional data, not redistributable |
| Section pattern configs | starter only — `data/sectionizer/discharge_sections.json` | induce your own; see [`../data/sectionizer/README.md`](../data/sectionizer/README.md) |

`data/**` is gitignored by default, and `.gitignore` re-admits public files one
at a time by exact name. A licensed CPT build or a corpus derivative therefore
cannot be committed by accident.

---

## The knowledge base

Every system in this repo reads one KB CSV. The column contract is
`data/kb/kb_schema.json`; validate any candidate file against it:

```bash
python scripts/build_kb.py validate --kb data/kb/hcpcs_level2_public.csv
```

**What ships.** `hcpcs_level2_public.csv` holds the 7,379 HCPCS Level II codes
(the `A`–`V` letter codes) with their CMS long descriptors. These are public
domain, so the file is tracked. It is enough to exercise every command end to
end, and it is a real KB, not a mock.

**What it is not.** It contains no CPT Level I codes, so it will not reproduce
the paper's numbers — the VUMC label space is 93.78% CPT-I surgery. CPT
descriptors are copyrighted by the American Medical Association and cannot be
redistributed. `code_lay_term`, a column the KB schema allows, is a separate
commercial product and is blank in every shipped row.

**Adding your licensed CPT descriptors.** Supply a CSV with at least `code` and
`code_description`:

```bash
python scripts/build_kb.py merge \
  --base data/kb/hcpcs_level2_public.csv \
  --cpt  /path/to/your/cpt_descriptors.csv \
  --out  data/kb/codes_with_ranges.csv
```

Anything else your file carries that matches the contract — lay terms, range
levels — is carried through. The merge never writes into the tracked demo file.
Point every command at the result with `--kb data/kb/codes_with_ranges.csv`.

A test guards the shipped file against contamination: it asserts every code
matches `^[A-V]`, every `code_system` is `HCPCS`, and every `code_lay_term` is
blank. If you merge in place, that test will fail — which is the intended
alarm, not a bug.

---

## NCCI edits

The National Correct Coding Initiative tables are CMS public domain, but they
are large (~254 MB unpacked) and versioned quarterly, so they are downloaded
rather than vendored. Get the Procedure-to-Procedure (PTP), Medically Unlikely
Edit (MUE), and Add-On Code (AOC) archives from the CMS NCCI page, then:

```bash
python scripts/setup_ncci.py --from-zips ~/Downloads/ncci --out data/ncci
python scripts/setup_ncci.py --check --out data/ncci
```

`--check` loads the tree through `NCCIRuleChecker` and prints what it found.
The versions behind the paper's numbers are recorded in the script's docstring;
`--check` on them prints 1,728,194 active PTP pairs, 681 add-on codes with
primaries, 50 contractor-defined add-ons, and 15,095 MUE codes.

Without NCCI data, seven tests skip with a message naming this script, and the
constrained-decode stage has no rules to apply. Everything else runs.

---

## Corpora

### MIMIC-IV

Requires credentialed PhysioNet access and a signed DUA. You need
`hosp/hcpcsevents.csv.gz` and `note/discharge.csv.gz` from MIMIC-IV v3.1 (notes
are a separate PhysioNet project from the hosp module — both are needed).

```bash
cptrec-bench-build-mimic \
  --hcpcs     /path/to/mimic-iv/hosp/hcpcsevents.csv.gz \
  --discharge /path/to/mimic-iv-note/note/discharge.csv.gz \
  --out-dir   outputs/datasets/mimic_iv
```

Three things about this corpus are structural, not incidental, and the code
enforces all three:

**The gold standard is partially recorded.** `hcpcsevents` covers 145,364 of
331,793 noted admissions. An absent code is not evidence of an absent
procedure. Precision, F1, MAP, and nDCG are therefore *suppressed* on this
corpus — `cptrec-bench-export` refuses to emit them — and every recall figure
is a lower bound. This is encoded as `PARTIAL_GOLD_SUPPRESSIONS` in
`benchmark/corpora.py` so a later table cannot quietly contradict it.

**The split is patient-disjoint, not temporal.** MIMIC shifts dates per
patient, so a temporal split would be meaningless. There is no drift split on
this corpus and there cannot be one.

**`hcpcsevents` is a billing table.** As recorded it is 61.6% surgical at 2.523
codes per encounter, because it bills a level of service alongside the
operation. The default `--code-filter procedural` drops E/M `99202`–`99499` and
E/M-equivalent HCPCS-II facility codes (`G0378` observation-per-hour and kin),
giving 99.48% surgical at 1.562 codes per encounter — a label space comparable
to a coder's procedure set. A level of service is not inferable from a
procedure narrative. The dropped codes are listed in the generated
`corpus_stats.json` under `codes.dropped_codes`.

Derived per-note files stay under `outputs/datasets/mimic_iv/`. The only path
out is `cptrec-bench-export`, which strips row-level keys and refuses any
destination named `predictions`.

### VUMC

Not obtainable. It is in the registry because the paper reports it and because
its entry documents the contract the other corpus is compared against —
reference-complete gold, a temporal split with a drift window, and 19
sectionizer headers. Read `benchmark/corpora.py` for the full contract.

### Your own

See [`OWN_CORPUS.md`](OWN_CORPUS.md).
