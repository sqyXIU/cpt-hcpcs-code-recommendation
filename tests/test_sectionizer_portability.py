# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
The sectionizer's two corpus-portability flags.

Both are additive and default-off, so the first thing every test here asserts is
that the default path is unchanged -- the champion-recipe discipline the rest of
the repo runs under ("byte-identical when off, one variable per experiment").

What the flags exist for:

``--backend {azure,local}``
    ``extract_headers`` constructed ``AzureOpenAIBackend`` directly, so the only
    way to run header discovery was to send note text to a third-party endpoint.
    That is fine for VUMC under the institutional agreement and not fine for a
    corpus whose registry entry says ``external_api_allowed=False``.

``--sections-file``
    both stages had ``STANDARD_SECTIONS`` hard-wired -- ``build_pattern_config``
    allocates one Counter per section name up front, so a header that is not one
    of the 19 VUMC operative-note buckets has no key to land in and is dropped
    without a warning.  ``test_hardwired_taxonomy_silently_drops_mimic_headers``
    pins that failure mode, because it is the one that would have produced a
    plausible-looking empty config rather than an error.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpt_rec.common.constants import STANDARD_SECTIONS
from cpt_rec.common.sectionizer import extract_headers as eh
from cpt_rec.common.sectionizer.build_pattern_config import (
    build_pattern_config,
)

# A discharge-summary taxonomy: none of these are VUMC operative-note sections.
MIMIC_SECTIONS = [
    "Major Surgical or Invasive Procedure",
    "Brief Hospital Course",
    "Pertinent Results",
]


def _write_jsonl(path, mapping, n=3):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps(
                {"row_index": i, "standard_section_headers": mapping}
            ) + "\n")
    return str(path)


# --------------------------------------------------------------------------
# default path unchanged
# --------------------------------------------------------------------------

def test_default_system_prompt_is_the_vumc_prompt():
    assert eh.build_system_prompt() == eh.SYSTEM_PROMPT
    assert eh.build_system_prompt(STANDARD_SECTIONS) == eh.SYSTEM_PROMPT
    for section in STANDARD_SECTIONS:
        assert f"- {section}" in eh.SYSTEM_PROMPT


def test_default_payload_and_normalizer_use_standard_sections():
    assert list(eh.empty_headers_payload()) == list(STANDARD_SECTIONS)
    out = eh.validate_and_normalize_output(
        {"standard_section_headers": {"Findings": ["Findings:", "Findings:"]}}
    )
    assert list(out) == list(STANDARD_SECTIONS)
    assert out["Findings"] == ["Findings:"]          # deduplicated, order kept


def test_pattern_config_default_matches_explicit_standard_sections(tmp_path):
    src = _write_jsonl(
        tmp_path / "vumc.jsonl",
        {"Findings": ["Findings:"], "Procedure(s) Performed": ["Procedure Performed:"]},
    )
    default, _ = build_pattern_config(src, 2, 100, False)
    explicit, _ = build_pattern_config(src, 2, 100, False, sections=list(STANDARD_SECTIONS))
    assert default["patterns"] == explicit["patterns"]
    assert default["patterns"], "VUMC headers should survive the default path"


# --------------------------------------------------------------------------
# --sections-file
# --------------------------------------------------------------------------

def test_custom_taxonomy_replaces_the_vumc_buckets():
    prompt = eh.build_system_prompt(MIMIC_SECTIONS)
    for section in MIMIC_SECTIONS:
        assert f"- {section}" in prompt
    assert "- Specimens Removed" not in prompt
    assert list(eh.empty_headers_payload(MIMIC_SECTIONS)) == MIMIC_SECTIONS


def test_custom_taxonomy_normalizer_keeps_only_its_own_sections():
    out = eh.validate_and_normalize_output(
        {"standard_section_headers": {
            "Brief Hospital Course": ["Brief Hospital Course:"],
            "Specimens Removed": ["Specimens:"],
        }},
        MIMIC_SECTIONS,
    )
    assert out["Brief Hospital Course"] == ["Brief Hospital Course:"]
    assert "Specimens Removed" not in out


def test_pattern_config_represents_a_mimic_taxonomy(tmp_path):
    src = _write_jsonl(
        tmp_path / "mimic.jsonl",
        {"Brief Hospital Course": ["Brief Hospital Course:"],
         "Major Surgical or Invasive Procedure": ["Major Surgical or Invasive Procedure:"]},
    )
    cfg, _ = build_pattern_config(src, 2, 100, False, sections=MIMIC_SECTIONS)
    blob = json.dumps(cfg)
    assert "Brief Hospital Course" in blob
    assert "Major Surgical or Invasive Procedure" in blob
    assert "Specimens Removed" not in blob


def test_hardwired_taxonomy_silently_drops_mimic_headers(tmp_path):
    """The failure mode the flag exists to prevent: no error, just nothing."""
    src = _write_jsonl(
        tmp_path / "mimic.jsonl",
        {"Brief Hospital Course": ["Brief Hospital Course:"],
         "Major Surgical or Invasive Procedure": ["Major Surgical or Invasive Procedure:"]},
    )
    cfg, _ = build_pattern_config(src, 2, 100, False)   # no sections= -> the 19
    assert cfg["patterns"] == [], (
        "discharge-summary headers have no matching key among STANDARD_SECTIONS, "
        "so the hard-wired builder returns an empty config instead of failing"
    )


# --------------------------------------------------------------------------
# --backend / the external-API guard
# --------------------------------------------------------------------------

# Every corpus in the live REGISTRY is now api-allowed -- VUMC institutionally,
# MIMIC-IV by the data owner on 2026-08-26 -- so nothing real trips the guard and
# these two tests would pass vacuously if they still pointed at MIMIC.  The guard
# is still what stands between a mistyped --input and a DUA breach on the NEXT
# corpus that lands, so it is exercised against a synthetic restricted entry
# instead, and test_mimic_authorization_is_deliberate below pins the policy fact
# that made the old fixture stale.
LOCKED_ROOT = "data/locked_corpus/derived"


@pytest.fixture
def restricted_corpus(monkeypatch):
    """A synthetic corpus that still forbids the external API."""
    from dataclasses import replace

    from cpt_rec.benchmark import corpora

    locked = replace(
        corpora.MIMIC,
        key="locked",
        display_name="Locked Corpus (test fixture)",
        root=Path(LOCKED_ROOT),
        external_api_allowed=False,
        external_api_note="synthetic fixture: its DUA forbids third-party endpoints",
    )
    monkeypatch.setitem(corpora.REGISTRY, "locked", locked)
    return locked


def test_guard_refuses_a_restricted_corpus_by_default(restricted_corpus):
    with pytest.raises(SystemExit) as exc:
        eh.assert_external_api_allowed(f"{LOCKED_ROOT}/train.csv", False)
    msg = str(exc.value)
    assert "REFUSING" in msg
    assert "--backend local" in msg          # points at the compliant route
    assert "--allow-external-api" in msg     # and at the override


def test_guard_allows_an_explicit_override(restricted_corpus):
    eh.assert_external_api_allowed(f"{LOCKED_ROOT}/train.csv", True)


def test_mimic_authorization_is_deliberate():
    """MIMIC-IV is api-allowed on purpose; a silent flip-back should fail here.

    The data owner authorized Azure OpenAI on MIMIC-IV note text on 2026-08-26,
    which is what lets the frontier rows run on both corpora.  The permission is
    conditional on the deployment (PhysioNet requires human review of prompts and
    completions to be disabled), so it is recorded in the corpus entry rather
    than assumed -- and if someone reverts it, the benchmark's cross-corpus
    comparability goes with it, so this asserts rather than infers.
    """
    from cpt_rec.benchmark import corpora

    assert corpora.MIMIC.external_api_allowed is True
    assert "2026-08-26" in corpora.MIMIC.external_api_note
    assert corpora.VUMC.external_api_allowed is True


def test_guard_does_not_fire_on_an_api_allowed_corpus():
    eh.assert_external_api_allowed(
        "outputs/datasets/vumc/train_eval_sectioned.csv", False
    )


def test_guard_does_not_fire_on_an_unregistered_path():
    eh.assert_external_api_allowed("/tmp/some_scratch_notes.csv", False)


def test_local_backend_is_importable_and_distinct():
    """--backend local resolves to a real class with the same chat contract."""
    assert eh.LocalOpenAIBackend is not eh.AzureOpenAIBackend
    assert hasattr(eh.LocalOpenAIBackend, "chat")


# --------------------------------------------------------------------------
# the "messages must contain the word json" 400
# --------------------------------------------------------------------------
#
# Azure only accepts response_format={"type":"json_object"} when the messages
# contain the literal LOWERCASE "json".  Every prompt in this repo writes "JSON"
# in caps, so deployments that enforce the rule 400 every call -- and because
# _maybe_adapt_params did not recognise this particular 400, the retry loop
# burned all four attempts on identical errors and then raised.  Observed on
# gpt-5.6-terra, 2026-08-26.

REAL_JSON_400 = (
    "Error code: 400 - {'error': {'message': \"'messages' must contain the word "
    "'json' in some form, to use 'response_format' of type 'json_object'.\", "
    "'type': 'invalid_request_error', 'param': 'messages', 'code': None}}"
)


@pytest.fixture()
def azure(monkeypatch):
    """An AzureOpenAIBackend whose client construction is stubbed out."""
    import cpt_rec.baselines.llm as llm

    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")

    def _make(deployment_name="gpt-5.6-terra"):
        backend = llm.AzureOpenAIBackend.__new__(llm.AzureOpenAIBackend)
        backend.deployment_name = deployment_name
        backend.temperature = 0.0
        backend.max_tokens = 800
        backend._use_completion_tokens = True
        backend._send_temperature = False
        backend._send_response_format = True
        return backend

    return _make


def test_json_mode_is_requested_by_default(azure):
    assert "response_format" in azure()._build_kwargs("sys", "user")


def test_the_json_word_400_drops_response_format(azure):
    b = azure()
    assert b._maybe_adapt_params(Exception(REAL_JSON_400)) is True
    assert b._send_response_format is False
    assert "response_format" not in b._build_kwargs("sys", "user")


def test_the_correction_is_one_way(azure):
    """Second time through it must report no change, or the loop never ends."""
    b = azure()
    b._maybe_adapt_params(Exception(REAL_JSON_400))
    assert b._maybe_adapt_params(Exception(REAL_JSON_400)) is False


def test_unrelated_400s_leave_json_mode_alone(azure):
    b = azure()
    assert b._maybe_adapt_params(Exception("Error 400: rate limit exceeded")) is False
    assert b._send_response_format is True


def test_existing_corrections_still_fire(azure):
    b = azure("gpt-4.1")
    b._use_completion_tokens = False
    b._send_temperature = True
    assert b._maybe_adapt_params(
        Exception("use max_completion_tokens instead of max_tokens")
    ) is True
    assert b._use_completion_tokens is True
    assert b._maybe_adapt_params(Exception("temperature is not supported")) is True
    assert b._send_temperature is False


# --------------------------------------------------------------------------
# loads_lenient
# --------------------------------------------------------------------------
#
# Only reachable once json mode has been dropped, which is exactly when it
# matters: call_llm_extract_headers turns any parse failure into an EMPTY
# payload, so a bare json.loads would have scored zero headers on every one of
# 2000 notes while logging one line each -- a silent null result, not a crash.

PAYLOAD = '{"standard_section_headers": {"Findings": ["Findings:"]}}'


@pytest.mark.parametrize("raw", [
    PAYLOAD,
    f"```json\n{PAYLOAD}\n```",
    f"```\n{PAYLOAD}\n```",
    f"Here is the result:\n{PAYLOAD}",
    f"Sure!\n{PAYLOAD}\nHope that helps.",
])
def test_loads_lenient_recovers_wrapped_json(raw):
    assert eh.loads_lenient(raw)["standard_section_headers"]["Findings"] == ["Findings:"]


def test_loads_lenient_is_not_confused_by_braces_inside_strings():
    raw = '{"standard_section_headers": {"Findings": ["Findings {x}:"]}}'
    assert eh.loads_lenient(raw)["standard_section_headers"]["Findings"] == ["Findings {x}:"]


def test_loads_lenient_empty_is_an_empty_object():
    assert eh.loads_lenient("") == {}


def test_loads_lenient_raises_when_there_is_no_json():
    with pytest.raises(json.JSONDecodeError):
        eh.loads_lenient("no object here at all")


def test_prompt_carries_the_lowercase_token_azure_requires():
    """The whole point of the added rule line -- keep json mode ON."""
    assert "json" in eh.SYSTEM_PROMPT, (
        "Azure rejects response_format=json_object unless the messages contain "
        "the literal lowercase 'json'; without it every call 400s"
    )


# --------------------------------------------------------------------------
# the MIMIC taxonomy, and the third stage that was never ported
# --------------------------------------------------------------------------
#
# Measured 2026-08-26: deriving the taxonomy from header frequencies produced 74
# buckets on 2000 MIMIC notes -- exam subfields promoted to top level,
# abbreviation duplicates of buckets already present, medication-sig fragments,
# patient-instruction prose. Because `find_non_overlapping_markers` treats every
# pattern as a split point, the subfields chopped Brief Hospital Course, the most
# CPT-relevant section in a discharge summary. The fix is a hand-curated
# constant, matching VUMC's own procedure. These tests pin the properties that
# make it correct, not merely the length of the list.

from cpt_rec.common.constants import MIMIC_DISCHARGE_SECTIONS
from cpt_rec.common.sectionizer import split_op_notes as sp

# The operative-note pattern config used for the paper's VUMC runs was induced
# from that corpus and is not distributed here; ``cptrec-extract-headers``
# induces an equivalent one from any corpus (see docs/OWN_CORPUS.md).  These
# tests exercise the *taxonomy contract* rather than that particular file, so a
# small synthetic config in the default taxonomy stands in for it.
VUMC_CFG_PATTERNS = {"version": 1, "patterns": [
    {"section": "Pre-operative Diagnosis",
     "regex": r"\bPRE[- ]?OPERATIVE\s+DIAGNOSIS\s*:\s*"},
    {"section": "Post-operative Diagnosis",
     "regex": r"\bPOST[- ]?OPERATIVE\s+DIAGNOSIS\s*:\s*"},
    {"section": "Procedure(s) Performed",
     "regex": r"\bPROCEDURES?\s+PERFORMED\s*:\s*"},
    {"section": "Anesthesia Type", "regex": r"\bANESTHESIA\s*:\s*"},
    {"section": "Findings", "regex": r"\bFINDINGS\s*:\s*"},
    {"section": "Estimated Blood Loss (EBL)",
     "regex": r"\bESTIMATED\s+BLOOD\s+LOSS\s*:\s*"},
]}

MIMIC_CFG = {"version": 1, "patterns": [
    {"section": "Procedures",
     "regex": r"\bMajor\s+Surgical\s+or\s+Invasive\s+Procedure\s*:\s*"},
    {"section": "Hospital Course", "regex": r"\bBrief\s+Hospital\s+Course\s*:\s*"},
    {"section": "Discharge Diagnosis", "regex": r"\bDischarge\s+Diagnosis\s*:\s*"},
    {"section": "Allergies and Intolerances", "regex": r"\bAllergies\s*:\s*"},
]}

MIMIC_NOTE = (
    "Name: ___   Unit No: ___\n"
    "Allergies: Penicillin\n"
    "Major Surgical or Invasive Procedure: laparoscopic cholecystectomy\n"
    "Brief Hospital Course: Pt did well post-op. Neuro: intact. Vitals: stable. "
    "HEENT: normal. Discharged home on POD2.\n"
    "Discharge Diagnosis: acute cholecystitis\n"
)


@pytest.fixture()
def mimic_cfg(tmp_path):
    p = tmp_path / "mimic_patterns.json"
    p.write_text(json.dumps(MIMIC_CFG))
    return str(p)


@pytest.fixture()
def vumc_cfg(tmp_path):
    p = tmp_path / "vumc_patterns.json"
    p.write_text(json.dumps(VUMC_CFG_PATTERNS))
    return str(p)


def test_mimic_taxonomy_has_no_duplicates():
    assert len(set(MIMIC_DISCHARGE_SECTIONS)) == len(MIMIC_DISCHARGE_SECTIONS)


def test_mimic_taxonomy_covers_the_ccda_required_sections():
    """C-CDA Discharge Summary (LOINC 18842-5) required sections must exist."""
    for required in ("Allergies and Intolerances", "Hospital Course",
                     "Discharge Diagnosis", "Plan of Treatment",
                     "Hospital Consultations", "Procedures", "Results"):
        assert required in MIMIC_DISCHARGE_SECTIONS, required


def test_mimic_taxonomy_excludes_the_buckets_that_caused_the_74(): 
    """The omissions are the load-bearing part; a regression here re-breaks it."""
    for banned in (
        # physical-exam subfields -- these recur inside Hospital Course
        "Vitals", "Neuro", "Heent", "Abd", "Ext", "Pulm", "Lungs", "Resp",
        "Cardiac", "Chest", "Heart", "Eyes", "Neck", "Skin", "Psych",
        "Extremities", "General", "Gen", "Abdomen", "Mental Status",
        # abbreviation duplicates of buckets already present
        "Psh", "Pmh", "Ros",
        # medication-sig fragments off "Disp:*30 Tablet Refills:*0"
        "Disp", "Refills", "Tablet Refills",
        # prose inside the discharge instructions
        "Your Bowels", "Your Incision", "How You May Feel", "Wound Care",
        "Pain Management", "Incision Care",
        # order-entry artifact
        "Reason For Prn Duplicate Override",
    ):
        assert banned not in MIMIC_DISCHARGE_SECTIONS, banned


def test_vumc_and_mimic_taxonomies_are_registered():
    assert sorted(sp.TAXONOMIES) == ["mimic_discharge", "vumc_op_note"]
    assert sp.VUMC_OP_NOTE.sections == tuple(STANDARD_SECTIONS)
    assert sp.MIMIC_DISCHARGE.sections == tuple(MIMIC_DISCHARGE_SECTIONS)


def test_splitter_default_is_still_the_operative_note_path(vumc_cfg):
    """Default-off discipline: adding MIMIC must not move the operative-note path.

    ``load_pattern_config`` with no ``taxonomy=`` argument still means the
    operative-note taxonomy, a config in it still loads, and a note with no
    recognizable header still lands whole in the fallback section.
    """
    rules = sp.load_pattern_config(vumc_cfg)          # no taxonomy= argument
    assert len(rules) == len(VUMC_CFG_PATTERNS["patterns"])
    out = sp.extract_sections("no headers at all here", sp.compile_rules(rules))
    assert set(out) == set(STANDARD_SECTIONS)
    assert out["Detailed Description"] == "no headers at all here"


def test_splitter_prefix_and_fallback_roles_differ_per_corpus():
    assert sp.VUMC_OP_NOTE.prefix_section == "Facility Information"
    assert sp.VUMC_OP_NOTE.fallback_section == "Detailed Description"
    assert sp.MIMIC_DISCHARGE.prefix_section == "Patient Identification"
    assert sp.MIMIC_DISCHARGE.fallback_section == "Hospital Course"


def test_mimic_note_splits_into_the_mimic_taxonomy(mimic_cfg):
    rules = sp.load_pattern_config(mimic_cfg, taxonomy=sp.MIMIC_DISCHARGE)
    out = sp.extract_sections(MIMIC_NOTE, sp.compile_rules(rules),
                              taxonomy=sp.MIMIC_DISCHARGE)
    assert set(out) == set(MIMIC_DISCHARGE_SECTIONS)
    assert "cholecystectomy" in out["Procedures"]
    assert "cholecystitis" in out["Discharge Diagnosis"]
    # the pre-header admin banner lands in the prefix role, not a facility bucket
    assert "Unit No" in out["Patient Identification"]


def test_exam_subfields_stay_inside_hospital_course(mimic_cfg):
    """THE regression test: the 74-bucket taxonomy chopped this section."""
    rules = sp.load_pattern_config(mimic_cfg, taxonomy=sp.MIMIC_DISCHARGE)
    out = sp.extract_sections(MIMIC_NOTE, sp.compile_rules(rules),
                              taxonomy=sp.MIMIC_DISCHARGE)
    course = out["Hospital Course"]
    for subfield in ("Neuro:", "Vitals:", "HEENT:"):
        assert subfield in course, f"{subfield} split the narrative instead of staying in it"
    assert "POD2" in course, "the tail of the narrative was truncated at a subfield"


def test_mimic_config_under_the_vumc_taxonomy_raises_and_names_the_flag(mimic_cfg):
    """Exactly the failure a port to a new genre hits; the message must be actionable."""
    with pytest.raises(ValueError) as exc:
        sp.load_pattern_config(mimic_cfg)             # defaults to vumc
    assert "mimic_discharge" in str(exc.value)


def test_vumc_config_under_the_mimic_taxonomy_raises_too(vumc_cfg):
    with pytest.raises(ValueError) as exc:
        sp.load_pattern_config(vumc_cfg, taxonomy=sp.MIMIC_DISCHARGE)
    assert "vumc_op_note" in str(exc.value)


SHIPPED_DISCHARGE_CFG = (
    Path(__file__).resolve().parents[1] / "data" / "sectionizer" / "discharge_sections.json"
)


def test_shipped_discharge_config_loads_and_covers_the_taxonomy():
    """The starter config in data/sectionizer/ must stay usable.

    It is hand-authored from conventional discharge-summary headings, not
    induced from any corpus, and it is the file a new user points
    ``--pattern-config`` at before inducing their own with
    ``cptrec-extract-headers``.  If the taxonomy gains a section and the config
    does not, a discharge summary silently loses that section.
    """
    assert SHIPPED_DISCHARGE_CFG.exists(), SHIPPED_DISCHARGE_CFG
    rules = sp.load_pattern_config(str(SHIPPED_DISCHARGE_CFG),
                                   taxonomy=sp.MIMIC_DISCHARGE)
    assert rules, "starter config is empty"
    covered = {r.section_name for r in rules}
    missing = set(MIMIC_DISCHARGE_SECTIONS) - covered
    assert not missing, f"no header pattern for: {sorted(missing)}"


def test_shipped_discharge_config_is_rejected_under_the_default_taxonomy():
    """It names discharge sections, so the operative-note taxonomy must refuse it."""
    with pytest.raises(ValueError) as exc:
        sp.load_pattern_config(str(SHIPPED_DISCHARGE_CFG))
    assert "vumc_op_note" in str(exc.value) or "not " in str(exc.value)


def test_shipped_discharge_config_keeps_exam_subfields_in_the_narrative():
    """Same regression the curated taxonomy exists to prevent, on the real config."""
    rules = sp.load_pattern_config(str(SHIPPED_DISCHARGE_CFG),
                                   taxonomy=sp.MIMIC_DISCHARGE)
    out = sp.extract_sections(MIMIC_NOTE, sp.compile_rules(rules),
                              taxonomy=sp.MIMIC_DISCHARGE)
    assert "cholecystectomy" in out["Procedures"]
    assert "cholecystitis" in out["Discharge Diagnosis"]
    for subfield in ("Neuro:", "Vitals:", "HEENT:"):
        assert subfield in out["Hospital Course"], subfield
    assert "POD2" in out["Hospital Course"]


def test_a_taxonomy_cannot_name_a_role_outside_its_own_sections():
    with pytest.raises(ValueError):
        sp.SectionTaxonomy(key="bad", sections=("A", "B"),
                           prefix_section="Nope", fallback_section="A")
