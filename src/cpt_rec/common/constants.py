# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Shared constants used across modules."""

from __future__ import annotations

from typing import List

STANDARD_SECTIONS: List[str] = [
    "Patient Identification",
    "Facility Information",
    "Date & Time",
    "Surgical Team",
    "Pre-operative Diagnosis",
    "Post-operative Diagnosis",
    "Procedure(s) Performed",
    "Indications for Surgery",
    "Anesthesia Type",
    "Positioning",
    "Detailed Description",
    "Findings",
    "Estimated Blood Loss (EBL)",
    "Specimens Removed",
    "Complications",
    "Implants & Equipment",
    "Counts",
    "Disposition",
    "Signature",
]

# ---------------------------------------------------------------------------
# MIMIC-IV discharge summaries
# ---------------------------------------------------------------------------
#
# Hand-curated, exactly like STANDARD_SECTIONS above -- deliberately NOT derived
# from header frequencies. A frequency filter cannot separate a document section
# from a field inside one, because both are frequent: on 2000 MIMIC notes
# "Vitals:" and "Neuro:" clear any threshold "Chief Complaint:" clears. A
# measured run of the derive-from-frequency approach (2026-08-26) returned 74
# buckets -- exam subfields promoted to top level, abbreviation duplicates of
# buckets already present ("Psh" beside "Past Surgical History"), medication-sig
# fragments off the "Disp:*30 Tablet Refills:*0" line, patient-instruction prose
# ("Your Bowels"), one entry truncated mid-sentence, and a pharmacy order-entry
# artifact. VUMC never had such a step; this list restores the procedure.
#
# Backbone: the HL7 C-CDA Discharge Summary template, document LOINC 18842-5,
# current published version C-CDA v5.0.0. Section names and LOINC codes below
# were checked against HL7's own published example document rather than from
# memory (HL7/C-CDA-Examples, Documents/Discharge Summary/Discharge_Summary.xml).
#
# Where MIMIC's header wording differs from C-CDA's section name, the C-CDA name
# is the bucket and MIMIC's wording is a SURFACE FORM. That is the whole point of
# the two-stage design: stage 1 maps many spellings onto one canonical section,
# stage 2 turns each observed spelling into its own regex. So "PSH:", "PMH:" and
# "ROS:" are patterns, not sections.
#
# The exclusions are the load-bearing part of a curated taxonomy:
#   * exam subfields (HEENT, Neuro, Abd, Ext, Pulm, Lungs, Resp, Cardiac, Chest,
#     Heart, Eyes, Neck, Skin, Psych, Extremities) -- these recur INSIDE the
#     Hospital Course narrative, so making them split points would chop the most
#     CPT-relevant section in the note.
#   * C-CDA's Vital Signs section, for that same reason: structured C-CDA carries
#     vitals as a section, MIMIC free text carries them as a line in the exam.
#   * discharge-instruction bullets (Wound Care, Pain Management, Your Bowels) --
#     prose inside Hospital Discharge Instructions.
#   * C-CDA Immunizations / Advance Directives / Nutrition / Problem List --
#     effectively absent as headers in MIMIC discharge summaries.
#
# Ordered by document flow, as STANDARD_SECTIONS is. Trailing comments give the
# C-CDA section and LOINC where one exists, then MIMIC's observed wording.
MIMIC_DISCHARGE_SECTIONS: List[str] = [
    # -- administrative; C-CDA carries these in the US Realm Header, MIMIC
    #    writes them as ordinary colon headers at the top of the note.
    "Patient Identification",          # MIMIC: Name / Unit No / Date of Birth / Sex
    "Admission & Discharge Dates",     # MIMIC: Admission Date / Discharge Date
    "Service & Attending",             # MIMIC: Service / Attending
    # -- pre-admission clinical story
    "Allergies and Intolerances",      # C-CDA 48765-2 (required); MIMIC: Allergies
    "Chief Complaint",                 # C-CDA 46239-0; MIMIC: Chief Complaint
    "History of Present Illness",      # C-CDA 10164-2
    "Admission Diagnosis",             # C-CDA 42347-5
    "Past Medical History",            # C-CDA 11348-0; MIMIC also "PMH"
    "Past Surgical History",           # no C-CDA section; MIMIC "PSH", CPT-relevant
    "Family History",                  # C-CDA 10157-6
    "Social History",                  # C-CDA 29762-2
    "Review of Systems",               # C-CDA 10187-3; MIMIC also "ROS"
    # -- inpatient findings and events
    "Physical Exam",                   # MIMIC: Physical Exam / Admission Physical Exam
    "Results",                         # C-CDA 30954-2 + Studies Summary 11493-4;
                                       # MIMIC: Pertinent Results / Imaging /
                                       # Admission Labs / Discharge Labs / Micro
    "Procedures",                      # C-CDA 47519-4 (required);
                                       # MIMIC: Major Surgical or Invasive Procedure
    "Hospital Consultations",          # C-CDA 18841-7 (required)
    "Hospital Course",                 # C-CDA 8648-8 (required);
                                       # MIMIC: Brief Hospital Course
    # -- discharge state
    "Hospital Discharge Physical",     # C-CDA 10184-0; MIMIC: Discharge Physical Exam
    "Discharge Diagnosis",             # C-CDA (required); MIMIC: Discharge Diagnosis
    "Discharge Condition",             # MIMIC: Discharge Condition
    "Functional Status",               # C-CDA 47420-5; MIMIC: Activity / Mental Status
    # -- medications and forward plan
    "Medications on Admission",        # MIMIC: Medications on Admission
    "Discharge Medications",           # C-CDA 75311-1; MIMIC: Discharge Medications
    "Discharge Disposition",           # MIMIC: Discharge Disposition
    "Hospital Discharge Instructions", # C-CDA 8653-8; MIMIC: Discharge Instructions
    "Plan of Treatment",               # C-CDA 18776-5 (required, "Plan of Care" in
                                       # older releases); MIMIC: Followup
                                       # Instructions / Transitional Issues
]

REQUIRED_COLS: List[str] = [
    "PAT_MRN_ID",
    "ENCOUNTER_CSN_ID",
    "NOTE_ID",
    "PROCEDURE_DATE",
    "NOTE_TIME",
    "NOTE_TEXT",
    "CPT_CODES",
]
