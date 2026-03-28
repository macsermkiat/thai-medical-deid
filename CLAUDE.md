# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**De-Iden** — A de-identification pipeline for Thai medical text. Screens personal health information (PHI) from hospital clinical notes (progress notes) using the [`loolootech/no-name-ner-th`](https://github.com/loolootech/no-name-ner-th) NER model.

**Input**: `PTPHYSICALEXAM 10-122025.xlsx` — KCMH physical exam records with a `progressnote` column (index 25, 0-based) containing free-text Thai clinical notes.

**Model**: `loolootech/no-name-ner-th` — A CamembertForTokenClassification model (based on `clicknext/phayathaibert`) fine-tuned for Thai medical NER. Detects 7 entity types: `PERSON`, `PHONE`, `EMAIL`, `ADDRESS`, `DATE`, `NATIONAL_ID`, `HOSPITAL_IDS`. Runs on CPU. License: CC BY-NC 4.0 (non-commercial).

## Setup & Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Core dependencies: transformers[torch], sentencepiece, pandas, openpyxl, tqdm

# Run de-identification on the Excel file
python deidentify.py

# Run Gradio demo (if using app.py from no-name-ner-th)
python app.py
```

## Architecture

The pipeline:
1. **Read** Excel with openpyxl/pandas — extract `progressnote` column
2. **NER inference** via `transformers.pipeline("token-classification", model="loolootech/no-name-ner-th", device=-1)`
3. **Entity merging** — combine B-/I- tagged sub-tokens into full entity spans
4. **Replacement** — substitute detected entities with `[ENTITY_TYPE]` tokens (e.g., `[PERSON]`, `[PHONE]`), processing in reverse index order to preserve positions
5. **Output** — write de-identified text back to a new Excel/CSV

## Data Notes

- The Excel file has 50 columns. Key columns: `hn` (hashed), `vstdate`, `vsttime`, `cliniclct`, `progressnote` (col 25), `progressnote2` (col 48)
- `hn` values are already hashed/encoded in this export
- Progress notes contain Thai medical text with embedded PHI: patient names, doctor names (e.g., "อ.สมชาย"), phone numbers (e.g., "084-0660172"), dates, addresses
- ~440 of the first 500 rows have non-null `progressnote`

## Entity Types & Replacement Tokens

| Entity | Token | Examples in data |
|--------|-------|-----------------|
| PERSON | `[PERSON]` | Patient/doctor names |
| PHONE | `[PHONE]` | Thai mobile numbers |
| EMAIL | `[EMAIL]` | Email addresses |
| ADDRESS | `[ADDRESS]` | Street/district info |
| DATE | `[DATE]` | Dates of birth, visit dates |
| NATIONAL_ID | `[NATIONAL_ID]` | Thai 13-digit IDs |
| HOSPITAL_IDS | `[HOSPITAL_IDS]` | HN, AN, VN numbers |

## Key Constraints

- **Read-only on source data**: Never modify the original Excel file. Write output to a separate file.
- **PHI handling**: The goal is to remove PHI. Never log or print raw PHI during development.
- **CPU inference**: Use `device=-1` (CPU). No GPU required.
- **Non-commercial license**: The NER model is CC BY-NC 4.0 — cannot be used commercially without contacting `contact@looloohealth.com`.
- **Thai text**: Progress notes are mixed Thai/English medical text. The model handles Thai; English names may need additional handling.
