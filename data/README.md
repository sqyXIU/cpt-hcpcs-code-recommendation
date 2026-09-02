# `data/`

Everything this repository can legally distribute, plus the empty slots where
the things it cannot go.

`.gitignore` is default-deny for this whole tree: `data/**` is ignored, and the
handful of public-domain files below are re-admitted one line at a time. That
is deliberate — a corpus dropped in here is ignored by default, not tracked by
accident.

## What ships

| Path | What it is | Provenance |
|---|---|---|
| `kb/hcpcs_level2_public.csv` | 7,379 HCPCS Level II codes with CMS descriptors and range hierarchy | CMS, public domain |
| `kb/kb_schema.json` | The column contract every knowledge base must satisfy | this repo |
| `sectionizer/discharge_sections.json` | Starter header patterns for discharge summaries, 41 patterns over 26 sections | hand-authored, not corpus-derived |

## What does not, and how to supply it

| Slot | Why it is absent | How to fill it |
|---|---|---|
| `kb/codes_with_ranges.csv` | CPT® Level I descriptors are AMA copyright | `scripts/build_kb.py merge` — see [kb/README.md](kb/README.md) |
| `ncci/` | CMS public domain but ~250 MB and reissued quarterly | `python scripts/setup_ncci.py --from-zips <dir>` |
| `notes/` | Clinical text. VUMC operative notes are institutional; MIMIC-IV needs a PhysioNet DUA | see [../docs/DATA.md](../docs/DATA.md) |
| `mimic_iv/` | PhysioNet credentialed access | `hcpcsevents.csv` + `discharge.csv.gz` from MIMIC-IV v3.1 |

## The rule

No clinical text, no patient- or encounter-level record, and no licensed code
descriptor may be committed to this repository. If git refuses to stage a file
here, that is the guard working. The fix is to add an explicit allowlist line
to `.gitignore` §1 *after* confirming the file contains none of those three
things — never `git add -f`.
