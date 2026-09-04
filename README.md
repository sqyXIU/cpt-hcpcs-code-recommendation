# CPT/HCPCS procedure-code recommendation

Reference implementation for ***CPT/HCPCS Code Recommendation from Clinical
Notes: A Comparative Evaluation of AI Methods***
([medRxiv 2026.08.29.26361731](https://www.medrxiv.org/content/10.64898/2026.08.29.26361731v1)).

Six systems — lexical retrieval, a long-document neural classifier, two
prompted frontier-LLM configurations, a supervised fine-tune, and a staged
retrieve-and-verify pipeline — evaluated on two clinical corpora with
deliberately different properties: VUMC operative notes (private,
reference-complete gold, temporally split) and MIMIC-IV discharge summaries
(credentialed-open, partially recorded gold, patient-disjoint split).

**This repository ships code and instructions, not results.** It holds every
system the paper reports, the harness that scored them, and enough public data
to run the whole benchmark on MIMIC-IV or on a corpus of your own. It ships no
clinical text, no metrics files, and no licensed code descriptor. The numbers
are in the paper; what is here is what produces them.

---

## Quick start

```bash
uv sync --extra gpu --extra serve --extra dev
```

> **Never run a bare `uv sync`** — torch lives in the `gpu` extra, so a bare
> sync uninstalls your GPU stack. Always pass the extras.

`uv.lock` is committed, and it is the environment the paper's numbers were
produced in — 172 packages, resolved once and not re-resolved since. The pins
that matter most are **numpy 2.2.6, torch 2.7.1, transformers 4.57.6 and
vllm 0.10.1.1**; the numpy-2 boundary in particular is load-bearing, because the
cached artifacts the pipeline writes are pickles. `uv sync` honours the lock by
default, so the command above reproduces that environment rather than resolving
a fresh one. Add `--locked` to make a drifted lock an error instead of a silent
re-resolution.

Changing a pin is a deliberate act: edit `pyproject.toml`, then `uv lock` and
commit the result. Installing into the environment with `pip` leaves the lock
describing something that no longer exists.

```bash
pytest -q
```

On a clean CPU-only clone: **201 passed, 21 skipped**. The skips name the data
or the GPU they need — they are real skips, not silent no-ops.

Then pick your path:

| you have | read |
|---|---|
| MIMIC-IV credentials, and want the paper's MIMIC numbers | [`docs/REPRODUCE.md`](docs/REPRODUCE.md) |
| your own coded notes | [`docs/OWN_CORPUS.md`](docs/OWN_CORPUS.md) |
| neither yet, and want to know what data is needed | [`docs/DATA.md`](docs/DATA.md) |

---

## The systems

`M<n>` is the paper's table label. The key beside it is the on-disk
metrics-directory name a run writes and `cptrec-bench-export` reads — spelled
the same way, so a directory listing and a table read alike.

| paper | key | command | code |
|---|---|---|---|
| M1 | `m1_bm25_knn` | `cptrec-m1-bm25-knn` | BM25 over training notes, weighted code vote |
| M2 | `m2_label_attention` | `cptrec-m2-longformer` | Clinical-Longformer + PLM-ICD label attention |
| M3 | `m3_zeroshot_frontier` | `cptrec-m3-zeroshot` | frontier LLM, whole note, JSON out |
| M4 | `m4_rag_frontier` | `cptrec-m4-rag` | BM25 exemplars + candidate descriptions → LLM |
| M5 | `m5_sft_local` | `cptrec-m5-sft` | LoRA SFT of an open-weights model, scored through M3's unmodified CLI |
| M6 | `m6_retrieve_verify` | `cptrec-verifier-*` | retrieve-and-verify pipeline |

M6 is a **staged pipeline of pluggable specialists**, not a multi-agent system.
Four stages, each replaceable at its interface:

1. **Evidence selection** — sectionize the note, or pass it whole. Which
   sections the verifier reads is a flag (`--section-cols`), not a separate
   system: the sectioned arm and the whole-note arm are one binary.
2. **Candidate pooling** — union of BM25 over KB descriptions, a SapBERT dense
   retriever, and training-note neighbours. This stage sets the recall ceiling
   available to everything downstream; `python -m cpt_rec.common.evaluation.oracle`
   measures it.
3. **Verification** — a cross-encoder scores each `(evidence, candidate)` pair.
   This slot previously held a bi-encoder, and swapping it is the exercised
   proof that the stages really are pluggable (the paper reports the gain).
4. **Decoding** — threshold calibration plus NCCI constraint repair. Constrained
   decode enforces validity; it is not an F1 lever.

`kb_index` is the KB retrieval index M6 reuses, not a compared system. Anything
you run on one corpus only — a locally-served variant of an API baseline, say —
gets its own key, prints tagged `*(corpus-specific)*`, and is excluded from the
rank correlation, since a system cannot be ranked against a corpus it never ran
on.

---

## Repository map

```
src/cpt_rec/
├── baselines/     M1–M5, plus the KB retrieval index and shared LLM plumbing
├── pipeline/      M6: pool/ → crossencoder/ → decode/
├── common/        sectionizer, preprocessing, knowledge base, NCCI, evaluation
├── evidence/      per-candidate evidence ledger and the grounding audit
└── benchmark/     corpus registry, MIMIC builder, export gate, table collation
data/              public KB + starter sectionizer config (everything else gitignored)
scripts/           KB builder/validator, NCCI setup
tests/             201 tests, no corpus required
docs/              REPRODUCE · DATA · OWN_CORPUS
```

Runs write under `outputs/`, which is gitignored in full.

---

## Data

Nothing in this repository is clinical data.

| | shipped | how to get it |
|---|---|---|
| HCPCS Level II KB (7,379 codes) | **yes**, CMS public domain | already here |
| CPT Level I descriptors | no — AMA copyright | license, then `scripts/build_kb.py merge` |
| NCCI PTP / MUE / AOC edits | no — large, quarterly | `scripts/setup_ncci.py --from-zips` |
| MIMIC-IV | no — PhysioNet DUA | credentialed access, then `cptrec-bench-build-mimic` |
| VUMC notes | no, and never | private institutional data |

The shipped KB is real and runs everything, but it holds no CPT-I codes, so it
will not reproduce the paper's label space (VUMC is 93.78% CPT-I surgery).
`.gitignore` is default-deny over `data/` and denies run outputs outright:
public files are re-admitted one at a time by exact name, so a licensed KB build
or a corpus derivative cannot be committed by accident.

See [`docs/DATA.md`](docs/DATA.md).

---

## Reading your numbers

Three rules decide whether a difference you measure is real. All three are
enforced in code rather than left to the reader, and all three will bite you on
your own corpus too.

**MIMIC-IV reports recall, never precision.** Its `hcpcsevents` table covers
145,364 of 331,793 noted admissions, so a code the source never recorded is not
evidence of an absent procedure. Precision, F1, MAP, and nDCG are suppressed at
export time on that corpus, and every recall figure there is a lower bound. If
your own gold is partially recorded, set `label_completeness` accordingly in the
corpus registry and the same suppression applies to you.

**Recall at a fixed budget rewards filling the budget.** A system that emits
five codes at B=5 is not comparable to one that emits two. Read `shown@B` beside
every `R@B`, and use `burden_at_recall` in `rank_metrics.json` for the
matched-burden comparison.

**Differences are read against measured floors.** `collate.DEFAULT_FLOORS`
carries the nondeterminism measured for this study (0.0070 micro-F1, 0.0099
R@5); gaps inside the floor are marked `≈` in the collated tables and treated as
ties when ranking. Those two constants are *our* hardware and *our* corpus —
re-measure them from a byte-identical twin pair before trusting them on yours.

A fourth point is a finding rather than a rule, and it is why the benchmark has
two corpora at all: the system ranking does **not** transfer between them. A
method ordering established on one corpus is not evidence about another.

---

## Scope

This repository contains the systems the paper reports and the harness that
produced its numbers. Three investigated tracks are deliberately absent: the
bi-encoder scorer the cross-encoder replaced, the ranking-loss objective
variants, and the sibling arbiter. They did not ship, and carrying dead
configurations would make the reproducible surface harder to trust.

---

## Citation

> Song Q, Ni C, Liu W, Li Y, Malin BA, Yin Z. *CPT/HCPCS Code Recommendation
> from Clinical Notes: A Comparative Evaluation of AI Methods.* medRxiv
> 2026.08.29.26361731; doi: https://doi.org/10.64898/2026.08.29.26361731

```bibtex
@article{song2026cptrec,
  title   = {{CPT/HCPCS} Code Recommendation from Clinical Notes:
             A Comparative Evaluation of {AI} Methods},
  author  = {Song, Qingyuan and Ni, Congning and Liu, Weixin and
             Li, Yike and Malin, Bradley A. and Yin, Zhijun},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.29.26361731},
  url     = {https://www.medrxiv.org/content/10.64898/2026.08.29.26361731v1}
}
```

[`CITATION.cff`](CITATION.cff) carries the same entry in machine-readable
form, so GitHub's "Cite this repository" button resolves to the preprint.

## License

[MIT](LICENSE) for the code in this repository.

The data it reads is licensed separately and is not covered by that grant:
HCPCS Level II descriptors and the NCCI tables are CMS public domain, CPT
descriptors are AMA copyright and must be licensed by you, and MIMIC-IV is
governed by the PhysioNet data use agreement you sign to obtain it.
