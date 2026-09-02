# Running on your own corpus

Every system in this repo takes `--notes`, `--kb`, and `--train-stats` as
paths, so pointing a method at a third corpus is a matter of pointing it
somewhere else. What is *not* interchangeable is what the resulting numbers
mean, and that is what the corpus registry exists to pin down.

Two levels, again:

- **Just run the models.** Produce CSVs in the expected shape and every command
  in [`REPRODUCE.md`](REPRODUCE.md) works. Nothing else is required.
- **Enter the benchmark.** Register the corpus in
  `src/cpt_rec/benchmark/corpora.py` so `cptrec-bench-export` knows which
  metrics are honest on your data and `cptrec-bench-collate` can put you in the
  cross-corpus table.

---

## 1. Shape your notes

One CSV per split, each row one document.

| column | required | notes |
|---|---|---|
| note id | yes | any of `NOTE_ID`, `note_id`, `note_index` — autodetected, or name it with `--note-id-col` |
| note text | yes | any of `NOTE_TEXT`, `note_text`, `text` — or `--note-text-col` |
| `proc_codes` | for train/val/test | the gold set, pipe-separated: `44970\|49320` |
| section columns | optional | one column per section; enables the section-restricted arms |

Everything else you carry is preserved untouched, so patient ids, dates, and
service lines survive for your own analysis.

Codes are uppercased and stripped on load, and every prediction is validated
against the KB, so out-of-vocabulary strings are dropped rather than scored.

### The split is yours to design, and it is a claim

Pick the split axis your claim needs, and record which one you picked — the two
shipped corpora differ here and the difference is load-bearing. VUMC is split
temporally (train ≤2023, val 2024, test 2025, plus a 2026 drift window), which
is what licenses its generalization claim. MIMIC-IV cannot be: it shifts dates
per patient, so it is split patient-disjoint on `subject_id` and has no drift
split at all. A patient-disjoint split cannot answer a temporal-drift question,
and saying so in `split_policy` is how the registry stops a later table from
implying otherwise.

### Training statistics

The head/torso/tail/unseen binning reads a `code_frequency_stats.csv` computed
over your **training** split — head is the top 80% of cumulative frequency,
torso 80–95%, tail the remaining 5%, and unseen is anything absent from train.

```bash
cptrec-normalize-labels stats --train-csv train.csv \
                              --output code_frequency_stats.csv
```

Columns: `code, frequency, cumulative_freq, rank, bin`.
`cptrec-bench-build-mimic` writes this file for you; pass it as `--train-stats`
to `cptrec-evaluate`. `cptrec-split-train-test` will build the splits
themselves if you do not already have them.

---

## 2. Optional: sections

If your documents have headers and you want the section-restricted arms, induce
a pattern config from your own corpus rather than reusing either shipped
taxonomy:

```bash
# 1. sample train notes and have an LLM map their headers onto your taxonomy
cptrec-extract-headers --input train.csv --output headers.jsonl \
                       --sections-file sections.txt --n-sample 2000

# 2. summarize those headers into a regex pattern config
cptrec-build-pattern-config --input-jsonl headers.jsonl \
                            --output-config data/sectionizer/your_corpus.json

# 3. split every note with the frozen config
cptrec-split-op-notes --input notes.csv --output notes_sectioned.csv \
                      --pattern-config data/sectionizer/your_corpus.json \
                      --taxonomy mimic_discharge
```

Step 1 sends note text to an LLM endpoint — the only stage in the sectionizer
that does. Check your data agreement before pointing it at restricted text.

Induce on **train only** — inducing patterns on the evaluation split leaks.
`data/sectionizer/discharge_sections.json` is a starter taxonomy for discharge
summaries, hand-authored from the C-CDA LOINC 18842-5 section list rather than
learned from any corpus. See
[`../data/sectionizer/README.md`](../data/sectionizer/README.md), which also
explains the over-segmentation failure mode (an uncurated induction on MIMIC
discharge summaries produces 74 buckets, most of them noise).

---

## 3. Optional: an LLM candidate prior

`cptrec-verifier-train` and `cptrec-verifier-predict` accept `--llm-sources`, a
CSV of candidate codes proposed by a generator of your choosing. The generator
itself is not shipped — the reported M6 result does not use one — but the
reader is, so you can plug yours in. Columns:

```
note_id, code, source, llm_rank, llm_confidence, supporting_phrase
```

`--keep-llm-sources` selects which `source` tags are honoured (default
`llm_concept`); `llm_confidence` may be blank; on a duplicate code the best
rank wins. Feature version `v1b` reads the `llm_*` columns as scorer features,
`v1a` uses the prior for candidate generation only.

---

## 4. Register the corpus

Add a `Corpus` to `src/cpt_rec/benchmark/corpora.py` and put it in `REGISTRY`.
The dataclass is the contract; the fields that carry real consequences are:

| field | why it matters |
|---|---|
| `label_completeness` | `"reference-complete"` or `"partially-recorded"` — this single field decides whether precision, F1, MAP, nDCG, and exact-match are emitted at all |
| `completeness_note` | the sentence a reader needs to interpret your recall numbers; it is copied into every export |
| `split_policy` | free text, but write the real one; it is what stops a patient-disjoint split from being read as a temporal one |
| `label_space_note` | what survived your construction filter — "as billed" and "procedures only" are different tasks |
| `external_api_allowed` | whether note text may leave for a third-party endpoint |
| `restricted_tree` | if true, derived per-note files may not leave `root` by any route except the export |
| `section_cols` / `reader_input_note` | what the corpus offers vs. what each system actually reads |

Then:

```bash
cptrec-bench-corpora <your-key>     # prints the contract back at you
```

### The completeness field is the one to get right

If your gold standard is partially recorded — a billing extract, a registry, a
chart review that stopped at the top three — set
`label_completeness="partially-recorded"`. A correct-but-unrecorded prediction
is then scored as a false positive, and the penalty *scales with how many codes
a system emits*, which makes precision and F1 not merely noisy but
**not comparable between systems**. Setting the field makes
`cptrec-bench-export` refuse to emit those metrics, via the `PARTIAL_GOLD_SUPPRESSIONS`
globs, and attach lower-bound caveats to the recall family that survives. The
globs match at any depth on purpose: a metric added later is suppressed by
default rather than shipped by default.

Recall, Coverage@B, and FamilyMRR survive as lower bounds — they are
gold-anchored, and the only distortion is displacement in the ranking, which
every system suffers equally. Prediction-only quantities — NCCI validity, set
size, latency — are unaffected.

---

## 5. Enter the cross-corpus table

`PORTABLE_CORE` in `corpora.py` lists the systems meant to run on every corpus
under one recipe. Anything outside it is printed as `*(corpus-specific)*` and
is excluded from the rank correlation, because a system present on one corpus
cannot be ranked against one it never ran on.

Give your system a paper label in `SYSTEM_LABELS` if you want it printed as
`M<n>`; leave it out and the table prints the bare key. The key is also the
metrics-directory name that `cptrec-bench-export` reads, so pick it once and
keep it: renaming a key after a run orphans that run's directory.

Then export and collate exactly as in
[`REPRODUCE.md`](REPRODUCE.md#step-6--export-and-collate). The transfer table
ranks on the first metric in `TRANSFER_METRIC_PREFERENCE` that every shared
system supports on every corpus, which for a partially recorded corpus means
R@B rather than F1.
