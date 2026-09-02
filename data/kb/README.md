# Knowledge base

A single CSV keyed by procedure code: one row per code, carrying its
description and its position in the code hierarchy. Everything downstream —
retrieval, negative sampling, family grouping, the sibling analysis — reads
this one file through `cpt_rec.common.knowledge.code_kb.CodeKnowledgeBase`.

`kb_schema.json` is the column contract. `scripts/build_kb.py validate` checks
a file against it.

## What ships: `hcpcs_level2_public.csv`

7,379 HCPCS Level II codes (letter-prefixed: A–V), with CMS long descriptors
and the CMS letter-range groupings as hierarchy. Public domain, works out of
the box, and enough to run the whole pipeline end to end.

The `code_lay_term` column is present but empty. Lay-term text is a commercial
product; the column is kept so a public build and a licensed build have the
same shape and the same code path.

## What does not ship: CPT® Level I

CPT® is copyright of the American Medical Association. The numeric five-digit
codes and their descriptors are licensed, so no CPT descriptor appears in this
repository and none may ever be committed to it. `tests/test_codes_and_kb.py`
has a test that fails if one is.

The paper's results use a licensed 2026 knowledge base of ~19,300 codes —
CPT Level I plus HCPCS Level II. To reproduce them you need your own licensed
source.

## Building a licensed KB

Put your licensed export in the schema's columns, then merge it onto the public
base:

```bash
python scripts/build_kb.py merge \
  --base data/kb/hcpcs_level2_public.csv \
  --licensed /path/to/your_cpt_export.csv \
  --out  data/kb/codes_with_ranges.csv
```

`merge` normalizes code case, defaults `code_system` to `CPT`, pads missing
range columns, resolves overlap in favour of the licensed file, and re-runs
`validate` on the result. The output path is gitignored.

Point the tools at it with `--kb data/kb/codes_with_ranges.csv`, or export
`CPT_REC_KB_CSV` so the test suite picks it up too.

## Optional: code history

`build_code_history` reads two more files, `code_changes.csv` and
`deleted_codes.csv`, to answer "was this code active on this date?". They are
optional; without them the `--history-*` flags are unavailable and everything
else works unchanged.
