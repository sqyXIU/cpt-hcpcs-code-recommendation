# Reproducing the paper

This repository ships code, not results, so "reproduce" here means *rerun*, not
*re-read*. What that costs depends on which corpus you have.

| corpus | needs | time | what you get |
|---|---|---|---|
| **MIMIC-IV** | PhysioNet DUA, 1 GPU | hours–days | the MIMIC column of the paper's tables, end to end |
| **VUMC operative notes** | — | — | not reproducible by anyone outside VUMC |
| **your own coded notes** | your data, 1 GPU | hours–days | the same benchmark on your corpus — see [`OWN_CORPUS.md`](OWN_CORPUS.md) |

The VUMC row is stated plainly because the alternative is implying otherwise:
those notes are private institutional data and cannot be released, so that
column can be read in the paper but not recomputed. MIMIC-IV is the reproducible
half of the benchmark, and it is why this code is public.

---

## Install

```bash
uv sync --extra gpu --extra serve --extra dev
```

The base install is CPU-only on purpose — corpus preparation, BM25 retrieval,
evaluation, and the benchmark export need no GPU. `--extra gpu` adds torch for
the neural systems (M2, M5, M6); `--extra serve` adds vLLM so M3/M4/M5 can run
against a local open-weights model instead of an external API.

**Never run a bare `uv sync`.** torch lives in the `gpu` extra, so a bare sync
uninstalls it. Pass the extras every time, and use `uv run --no-sync` for
long-running commands so a stale lock cannot re-resolve mid-experiment.

Without uv:

```bash
pip install -e ".[gpu,serve,dev]"
```

Then check the install:

```bash
pytest -q
```

On a clean CPU-only clone this is **201 passed, 21 skipped**. The skips are
honest: they name the data or the GPU they need. Adding NCCI data (below)
converts seven of them to passes.

---

## Run the benchmark on MIMIC-IV

### Prerequisites

1. A knowledge base. The shipped HCPCS-only demo KB runs everything but will
   not reproduce the paper's label space; merge your licensed CPT descriptors
   first. See [`DATA.md`](DATA.md).
2. NCCI edits, if you want the constrained-decode stage and the validity
   metrics: `python scripts/setup_ncci.py --from-zips … --out data/ncci`.
3. MIMIC-IV v3.1 `hosp/hcpcsevents.csv.gz` and `note/discharge.csv.gz`.

### Step 1 — build the corpus

```bash
cptrec-bench-build-mimic \
  --hcpcs     /path/to/mimic-iv/hosp/hcpcsevents.csv.gz \
  --discharge /path/to/mimic-iv-note/note/discharge.csv.gz \
  --out-dir   outputs/datasets/mimic_iv
```

Writes `train.csv` / `val.csv` / `test.csv` (patient-disjoint 70/10/20 on
`subject_id`, seed 42), `code_frequency_stats.csv`, and `corpus_stats.json`.
Add `--dry-run` first to sanity-check the join without writing.

Confirm the registry agrees with what landed on disk:

```bash
cptrec-bench-corpora mimic
```

### Step 2 — build the indexes

```bash
# BM25 over training notes — M1's index, and M4's exemplar source
cptrec-m1-bm25-knn build-index \
  --train outputs/datasets/mimic_iv/train.csv \
  --index-out outputs/indices/note_bm25/mimic/index.pkl

# BM25 + dense over KB code descriptions — M6's candidate generator
cptrec-build-kb-index build-index \
  --kb data/kb/codes_with_ranges.csv \
  --out-dir outputs/indices/code_kb/mimic/ \
  --biencoder cambridgeltl/SapBERT-from-PubMedBERT-fulltext
```

### Step 3 — run the systems

Each writes a predictions CSV. Full flag lists are in each module's docstring
(`python -m cpt_rec.baselines.m1_bm25_knn --help`).

```bash
# M1 — BM25 neighbor voting                                        (CPU)
cptrec-m1-bm25-knn tune-threshold --notes outputs/datasets/mimic_iv/val.csv \
  --index outputs/indices/note_bm25/mimic/index.pkl \
  --out-json outputs/baselines/m1_bm25_knn/threshold.json
cptrec-m1-bm25-knn predict --notes outputs/datasets/mimic_iv/test.csv \
  --index outputs/indices/note_bm25/mimic/index.pkl \
  --threshold-json outputs/baselines/m1_bm25_knn/threshold.json \
  --out outputs/baselines/m1_bm25_knn/predictions/test.csv --dump-scores-npz

# M2 — Clinical-Longformer + label attention                       (GPU)
cptrec-m2-longformer train --train outputs/datasets/mimic_iv/train.csv \
  --val outputs/datasets/mimic_iv/val.csv \
  --model-out outputs/baselines/m2_label_attention/model/ \
  --head label-attention --max-length 4096 --epochs 3 --min-code-freq 5

# M3 — zero-shot frontier LLM                                      (API or vLLM)
cptrec-m3-zeroshot --notes outputs/datasets/mimic_iv/test.csv \
  --kb data/kb/codes_with_ranges.csv \
  --out outputs/baselines/m3_zeroshot_frontier/predictions/test.csv \
  --max-note-tokens 4096

# M4 — exemplar RAG                                                (API or vLLM)
cptrec-m4-rag --notes outputs/datasets/mimic_iv/test.csv \
  --index outputs/indices/note_bm25/mimic/index.pkl \
  --kb data/kb/codes_with_ranges.csv \
  --out outputs/baselines/m4_rag_frontier/predictions/test.csv --top-k 5

# M5 — LoRA SFT of a local open-weights model                      (GPU)
cptrec-m5-sft train --base-model <hf-model-or-path> \
  --notes outputs/datasets/mimic_iv/train.csv \
  --out-dir outputs/baselines/m5_sft_local/sft --max-note-tokens 4096
cptrec-m5-sft merge --base-model <hf-model-or-path> \
  --adapter outputs/baselines/m5_sft_local/sft/adapter-final \
  --out outputs/baselines/m5_sft_local/merged
```

M5 is scored by serving the merged model with `vllm serve` and running the
**unmodified** M3 CLI against it (`--backend local`). That is deliberate: it
holds the prompt and the parser fixed, so the M3 → M5 difference is the
supervision and nothing else.

### Step 4 — M6, the retrieve-and-verify pipeline

```bash
cptrec-verifier-train \
  --train-csv outputs/datasets/mimic_iv/train.csv \
  --val-csv   outputs/datasets/mimic_iv/val.csv \
  --kb        data/kb/codes_with_ranges.csv \
  --kb-index-dir outputs/indices/code_kb/mimic/ \
  --bm25-index   outputs/indices/note_bm25/mimic/index.pkl \
  --model-out    outputs/verifier/mimic_wholenote/ \
  --epochs 5 --neg-per-pos 10 --max-neg-per-note 20 \
  --max-pair-len 320 --pair-truncation only_second --num-workers 8

cptrec-verifier-predict \
  --notes outputs/datasets/mimic_iv/test.csv \
  --model outputs/verifier/mimic_wholenote/ \
  --kb    data/kb/codes_with_ranges.csv \
  --kb-index-dir outputs/indices/code_kb/mimic/ \
  --bm25-index   outputs/indices/note_bm25/mimic/index.pkl \
  --split test --save-npz
```

`--save-npz` is required for the recall-at-budget metrics: the NPZ stores every
candidate's score regardless of the decision threshold, so any operating point
can be re-scored later without re-running the model.

The evidence-selection ablation is a flag on these same two commands, not a
second system: pass `--section-cols Procedures "Hospital Course" Results` to
both `cptrec-verifier-train` and `cptrec-verifier-predict` to feed M6 the
sectioned view instead of the whole note. Build the section columns first with
the sectionizer (see [`OWN_CORPUS.md`](OWN_CORPUS.md)); the shipped MIMIC corpus
is whole-note, and the arm writes under the same `m6_retrieve_verify` key.

Optionally calibrate the threshold and apply the NCCI edits:

```bash
cptrec-calibrate --val-npz outputs/verifier/mimic_wholenote/predictions/val/predictions.npz \
                 --out     outputs/verifier/mimic_wholenote/calibration.json
cptrec-constrained-decode --scores outputs/verifier/mimic_wholenote/predictions/test/predictions.npz \
                          --ncci-dir data/ncci
```

Constrained decode is a validity repair, not an F1 lever — it enforces NCCI
cardinality and pairing rules on an already-scored candidate set.

### Step 5 — evaluate

```bash
cptrec-evaluate \
  --predictions outputs/baselines/m1_bm25_knn/predictions/test.csv \
  --gold outputs/datasets/mimic_iv/test.csv \
  --kb   data/kb/codes_with_ranges.csv \
  --train-stats outputs/datasets/mimic_iv/code_frequency_stats.csv \
  --scores-npz  outputs/baselines/m1_bm25_knn/predictions/test_scores.npz \
  --out-dir outputs/baselines/m1_bm25_knn/metrics/test
```

`--scores-npz` enables the shortlist-review suite (R@B, Coverage@B, FamilyMRR,
pool ceiling, burden-at-recall). Without it you get set metrics only.

### Step 6 — export and collate

```bash
for sys in m1_bm25_knn m2_label_attention m3_zeroshot_frontier \
           m4_rag_frontier m5_sft_local m6_retrieve_verify; do
  cptrec-bench-export --corpus mimic --system "$sys" --split test \
    --metrics-dir outputs/baselines/$sys/metrics/test \
    --export-root outputs/benchmark/top5
done

cptrec-bench-collate --root outputs/benchmark/top5 --out-dir outputs/benchmark/top5/tables
```

The export is the boundary. It strips row-level keys, applies the
partial-gold suppressions (so no precision or F1 escapes from MIMIC), refuses
any destination named `predictions`, and writes an `export_summary.json`
recording the source of every file and every caveat attached to it. Nothing
should reach a shareable directory by any other route.

---

## What is not in this repo

Under the paper's scope this repository ships the reported systems and the
harness that produced the reported numbers. Three research tracks that appear
in the paper's narrative but not in its result tables were left out: the
bi-encoder scorer that the cross-encoder replaced, the ranking-loss objective
variants, and the sibling arbiter. They were investigated and did not ship, and
carrying dead configurations would make the reproducible surface harder to
trust, not easier.

The LLM candidate-prior generator is also absent. The `--llm-sources` flag that
would consume its output remains, default-off, with the file format documented
in [`OWN_CORPUS.md`](OWN_CORPUS.md) — the reported M6 result does not use one.
