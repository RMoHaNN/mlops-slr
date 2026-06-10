# MLOps SLR — Generic vs Industrial AI Implementation

This repository contains a fully scripted systematic literature review (SLR)
comparing **Generic AI Implementation** (cloud-native, SaaS, enterprise
software ML deployment) with **Industrial AI Implementation** (manufacturing,
Industry 4.0, predictive maintenance, industrial IoT). It addresses two
research questions: **RQ1** — What operational constraints distinguish
Industrial AI implementation from Generic AI implementation? **RQ2** — What
design principles for Industrial AI can be identified from the gaps between
the two literatures? The pipeline screens 268 Scopus records in two LLM-based
stages (abstract, then full text), codes included papers on 10 structured
fields, and synthesises themes, cross-stream gaps, and design principles.

## Pipeline

1. **Abstract screening** (`run_abstract_screening.py`) — reads the two
   Scopus exports in `data/raw/`, deduplicates Stream B (industrial) into
   Stream A and tags every record with its stream, excludes books and
   conference-review volumes by document type (code `E_DOC`), and screens
   the rest with Gemini on title + abstract. One row per record is written
   to `screening/abstract_screening.csv`.
2. **Full-text screening** (`run_fulltext_screening.py`) — for every paper
   that passed abstract screening, looks up its PDF in `output/pdf_cache/`.
   If the PDF is not found, the paper is recorded as excluded with code
   `FE_NOPDF` ("full text could not be obtained"). Otherwise the text is
   extracted and Gemini confirms inclusion and extracts the 10 coding
   fields defined in `screening_config.py`. One row per paper is written
   to `screening/fulltext_screening.csv`.
3. **Thematic analysis** (`scripts/thematic_analysis.py`) — three LLM
   passes over the included papers: within-stream themes, cross-stream
   gap analysis, and design principles (RQ2). Writes five files to
   `analysis/results/`.
4. **PRISMA numbers** (`scripts/prisma_numbers.py`) — reproduces the PRISMA
   flow counts from the archived screening files.

## PRISMA flow (this review)

| Stage | Count |
|---|---|
| Records identified (Scopus, after deduplication) | 268 (212 generic, 56 industrial) |
| Excluded at abstract screening | 47 |
| Assessed for full-text eligibility | 221 |
| Excluded — full text could not be obtained (`FE_NOPDF`) | 16 |
| Excluded — full-text content criteria (FE1–FE5) | 25 |
| **Included in synthesis** | **180 (134 generic, 46 industrial)** |

## Repository structure

```
data/raw/                 Scopus exports (Stream A: 268 generic-MLOps records;
                          Stream B: 56 industrial-AI records, a subset of A)
                          and the documented search strings
output/pdf_cache/         Full-text PDFs (not in git — see "PDFs" below)
screening/                Authoritative record of all screening decisions:
                          abstract_screening.csv (268 rows),
                          fulltext_screening.csv (221 rows)
analysis/results/         themes_generic.json, themes_industrial.json,
                          cross_stream_gaps.json, design_principles.json,
                          thematic_report.md
screening_config.py       Models, prompt versions, system prompts, and the
                          10 full-text coding fields (the audit trail for
                          every LLM decision)
run_abstract_screening.py Stage 1 (see Pipeline above)
run_fulltext_screening.py Stage 2
scripts/thematic_analysis.py  Stage 3
scripts/prisma_numbers.py     PRISMA flow counts
```

## How to replicate

```bash
pip install google-genai pdfplumber pypdf

python run_abstract_screening.py
python run_fulltext_screening.py
python scripts/thematic_analysis.py
python scripts/prisma_numbers.py
```

Both screening scripts are resumable: they append one row per record as they
go and skip already-processed records on restart, so an interrupted run can
simply be started again.

## API key

Only **Google Gemini** is needed. Place the key in
`~/.config/academic-research/config.toml`:

```toml
[gemini]
api_key = "your-key-here"
```

## PDFs

The full-text PDFs are **not redistributed** (institutional subscription).
DOIs and titles for all papers are in `screening/abstract_screening.csv`.
To rebuild the corpus, obtain each PDF and place it in `output/pdf_cache/`
named as the DOI with `/` replaced by `_`, plus `.pdf` — for example DOI
`10.1109/ACCESS.2023.3311713` becomes `10.1109_ACCESS.2023.3311713.pdf`.
Papers without a DOI use `no-doi_<title-slug>.pdf`; the exact slug rule is
`cache_filename()` in `run_fulltext_screening.py`. Papers whose full text
cannot be obtained are recorded as `FE_NOPDF` — the pipeline never downloads
anything itself.

## Reproducibility note

Re-running the pipeline produces comparable but not bit-identical results —
LLM outputs vary slightly across API versions and time. The archived files in
`screening/` are the authoritative record of the decisions made for this
review. Reviewers are encouraged to compare re-run decisions against the
archived `reason` column rather than expecting identical text.

## Contact

Mohan Rajashekarappa, Chalmers University of Technology —
rmohan@chalmers.se
