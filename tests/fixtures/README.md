# Test fixtures

## `mini_kb.csv`

A 24-row knowledge base that satisfies the full 16-column contract in
`data/kb/kb_schema.json`. It exists so the KB tests can assert on a hierarchy
small enough to reason about, and so they keep passing on a clone that has no
licensed CPT data.

**16 synthetic codes** (`code_system = "DEMO"`) in a fictional two-level
taxonomy:

| range | family |
|---|---|
| `DEMO001`–`DEMO099` | surgical |
| `DEMO010`–`DEMO019` | · upper endoscopy |
| `DEMO020`–`DEMO029` | · lower endoscopy |
| `DEMO030`–`DEMO039` | · skin |
| `DEMO100`–`DEMO199` | medical |
| `DEMO110`–`DEMO119` | · imaging |

The descriptors are invented. Nothing here is a CPT descriptor, a lay term, or
a paraphrase of one, and no test asserts a real code's real meaning against
this file.

**8 real HCPCS Level II rows** — `J0585`, `J1885`, `A4550`, `E0114`, `L3908`,
`G0378`, `Q4101`, `K0001` — copied verbatim from the CMS public-domain
descriptor set. They are here so the tests exercise real code shapes
(`^[A-V]` letter codes, mixed families) alongside the synthetic taxonomy.

The fixture is loaded through the real `CodeKnowledgeBase`, not a stub, so a
schema change breaks these tests rather than silently passing.

## Pointing tests at your own data

Tests that need a corpus read environment variables and skip when the path is
absent, so a clean clone reports honest skips rather than silent no-ops:

| variable | default |
|---|---|
| `CPT_REC_KB_CSV` | `data/kb/hcpcs_level2_public.csv` |
| `CPT_REC_SAMPLE_CSV` | `data/notes/sample_notes.csv` |
| `CPT_REC_SAMPLE_NORM_CSV` | `outputs/notes/sample_notes_normalized.csv` |
| `CPT_REC_KB_INDEX_DIR` | `outputs/indices/code_kb/default` |
| `CPT_REC_NCCI_DIR` | `data/ncci` |
| `CPT_REC_TRAIN_STATS_CSV` | `outputs/splits/code_frequency_stats.csv` |

`tests/_skip.py` is what makes the skips honest: under pytest it calls
`pytest.skip`, and outside pytest it prints a `SKIP:` line. The previous
`print(...); return` idiom counted as a pass.
