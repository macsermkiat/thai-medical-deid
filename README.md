# Thai Medical Text De-identification

A de-identification pipeline for Thai medical clinical notes. Detects and masks personal health information (PHI) using a hybrid approach: **regex pre-processing** for structured identifiers + **NER model** ([loolootech/no-name-ner-th](https://huggingface.co/loolootech/no-name-ner-th)) for names, addresses, and other free-text PHI.

Built for hospital progress notes (mixed Thai/English medical text). Runs on CPU.

## Why hybrid?

The NER model alone misses ~47% of hospital number (HN) patterns embedded in clinical notes. A regex pre-pass catches these reliably before the model handles the rest.

| Layer | Target | Accuracy |
|-------|--------|----------|
| Regex | HN numbers (`HN 35447/60`, `HN 8436260`) | ~100% |
| Regex | Thai national IDs (13-digit) | ~100% |
| NER model | Person names, phone numbers, emails, addresses, dates, other hospital IDs | Model-dependent |

## Entity types

| Entity | Replacement token | Examples |
|--------|-------------------|---------|
| PERSON | `[PERSON]` | Patient and doctor names |
| PHONE | `[PHONE]` | Thai mobile numbers |
| EMAIL | `[EMAIL]` | Email addresses |
| ADDRESS | `[ADDRESS]` | Street, district info |
| DATE | `[DATE]` | Dates of birth, visit dates |
| NATIONAL_ID | `[NATIONAL_ID]` | Thai 13-digit national IDs |
| HOSPITAL_IDS | `[HOSPITAL_IDS]` | HN, AN, VN numbers |

## Setup

```bash
pip install -r requirements.txt
```

Requirements: Python 3.9+, ~500 MB disk for model download on first run.

## Usage

### 1. Configure your input

Edit the constants at the top of `deidentify.py`:

```python
INPUT_FILE = "your_file.xlsx"
OUTPUT_FILE = "your_file_deidentified.xlsx"
COLUMN = "progressnote"    # column name containing clinical text
START_ROW = 10             # first data row to process (1-based)
END_ROW = 100              # last data row to process (inclusive)
```

### 2. Run

```bash
python deidentify.py
```

### 3. Output

- **De-identified Excel** file with PHI replaced by tokens
- **Detection log** (CSV) with entity positions and types (no original PHI text)

## Example

**Before** (raw clinical note):
```
นางรุจี ศุภวิเชียร 64 ปี
HN 37838/59
302
-----------
Known case
#Rheumatic MS with AF S/P MVR (SJM, unknown size), TVA (#28) since 7/12/2558
- Warfarin 32.5 mg/wk
- Last INR = 3.4 (22/7/68), keep 2.5-3.5
Tel: 084-0660172
เลขที่บัตรประชาชน 3160101285616
```

**After** (de-identified):
```
[PERSON] 64 ปี
HN [HOSPITAL_IDS]
302
-----------
Known case
#Rheumatic MS with AF S/P MVR (SJM, unknown size), TVA (#28) since 7/12/2558
- Warfarin 32.5 mg/wk
- Last INR = 3.4 ([DATE]), keep 2.5-3.5
Tel: [PHONE]
เลขที่บัตรประชาชน [NATIONAL_ID]
```

## How it works

```
Input text
    |
    v
[1] Regex pre-processing
    - HN numbers -> HN [HOSPITAL_IDS]
    - 13-digit national IDs -> [NATIONAL_ID]
    |
    v
[2] NER model inference (loolootech/no-name-ner-th)
    - Token classification with CamembertForTokenClassification
    - Sub-token merging (B-/I- tags -> full entity spans)
    |
    v
[3] Entity replacement
    - Replace detected spans with [ENTITY_TYPE] tokens
    - Process in reverse index order to preserve positions
    |
    v
De-identified text
```

## Sample output

```
$ python deidentify.py

Reading PTPHYSICALEXAM 10-122025.xlsx ...
Total rows: 130525, Columns: 50
Loading NER model: loolootech/no-name-ner-th ...
Model loaded.
Rows 10-100: 91 total, 80 with non-null progressnote
De-identifying: 100%|██████████| 80/80 [00:25<00:00,  3.12it/s]

De-identified output written to: PTPHYSICALEXAM_deidentified.xlsx

Summary: 312 entities detected across 80 notes
  PERSON: 142
  DATE: 68
  PHONE: 41
  HOSPITAL_IDS: 35
  ADDRESS: 18
  NATIONAL_ID: 6
  EMAIL: 2
```

## Model

[loolootech/no-name-ner-th](https://huggingface.co/loolootech/no-name-ner-th) -- a CamembertForTokenClassification model based on [clicknext/phayathaibert](https://huggingface.co/clicknext/phayathaibert), fine-tuned for Thai medical NER.

**License**: CC BY-NC 4.0 (non-commercial use only). For commercial licensing, contact `contact@looloohealth.com`.

## License

This pipeline code is provided as-is for research and non-commercial use, in accordance with the underlying NER model license (CC BY-NC 4.0).
