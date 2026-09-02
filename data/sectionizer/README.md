# Section patterns

A clinical note is split into named sections before anything reads it. The
splitter takes two things: a **taxonomy** (which sections exist, which one
holds the text before the first header, which one holds a note with no headers
at all) and a **pattern config** (regexes that mark where each section starts).

Taxonomies are code — `cpt_rec.common.sectionizer.split_op_notes.TAXONOMIES`
holds `vumc_op_note` (19 operative-note sections) and `mimic_discharge` (26
discharge-summary sections). Pattern configs are data, and live here.

## `discharge_sections.json`

41 patterns over all 26 sections of the `mimic_discharge` taxonomy,
hand-authored from the C-CDA Discharge Summary section set (LOINC 18842-5) and
the headings US discharge summaries conventionally use. It is not induced from
any corpus, so it ships.

```bash
cptrec-split-op-notes \
  --input  your_notes.csv --output your_notes_sectioned.csv \
  --pattern-config data/sectionizer/discharge_sections.json \
  --taxonomy mimic_discharge
```

The output is a *superset* of the input: every original column survives
verbatim and one column per section is appended. Pointing a baseline's
`--notes` at it therefore still reads the whole raw note — reading sections is
a separate, explicit flag.

A config is checked against its taxonomy at load time: naming a section the
taxonomy does not have is an error, not a silent drop. That is why this file
must be loaded with `--taxonomy mimic_discharge` — under the default
operative-note taxonomy it is correctly rejected.

## Inducing your own

The starter config covers conventional headings. Your corpus will have its own.
Induction is two stages, then the splitter.

**1. Extract headers.** An LLM reads a sample of notes and maps each
header-shaped line onto one of the taxonomy's sections. Write the taxonomy you
are targeting to a file, one section per line, and pass it as
`--sections-file` — that file is what keeps all three stages agreeing.

```bash
python -c "from cpt_rec.common.constants import MIMIC_DISCHARGE_SECTIONS as S; \
           print('\n'.join(S))" > sections_discharge.txt

cptrec-extract-headers \
  --input your_notes.csv --output headers.jsonl \
  --n-sample 2000 --sections-file sections_discharge.txt
```

This stage calls an LLM endpoint — see [../../docs/DATA.md](../../docs/DATA.md)
before pointing it at restricted note text. It is the only place in the
sectionizer that leaves the machine.

**2. Build the config.** Turn the observed headers into regexes.

```bash
cptrec-build-pattern-config \
  --input-jsonl headers.jsonl \
  --output-config data/sectionizer/your_corpus.json
```

**3. Split**, passing the same taxonomy as in step 1.

```bash
cptrec-split-op-notes \
  --input your_notes.csv --output your_notes_sectioned.csv \
  --pattern-config data/sectionizer/your_corpus.json \
  --taxonomy mimic_discharge
```

One warning from doing this on discharge summaries: header *frequency* alone
over-segments. An unfiltered induction produced 74 buckets, promoting physical-
exam subfields (`Vitals:`, `Neuro:`, `HEENT:`) to top-level sections — and
because every pattern is a split point, those subfields chopped the hospital
course into fragments, which is the single most code-relevant section in a
discharge summary. The taxonomy is curated by hand for exactly this reason.
Review an induced config before using it, and check that your narrative section
survives intact. `tests/test_sectionizer_portability.py` pins this property.

The configs used for the paper's runs were induced from their corpora and are
therefore corpus derivatives; they are not distributed here.
