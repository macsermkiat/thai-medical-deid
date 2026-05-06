# Thai Medical Text De-identification

A production-grade de-identification pipeline for Thai hospital clinical notes. Detects and masks Personal Health Information (PHI) using a **three-layer hybrid approach**: regex pre-processing for structured identifiers + a Thai full-name gazetteer (PyThaiNLP) for untitled names + an NER model ([loolootech/no-name-ner-th](https://huggingface.co/loolootech/no-name-ner-th)) for the long tail.

Built for mixed Thai/English progress notes across OPD, IPD, and Radiology sheets. Runs on CPU or GPU (A100 tested via Slurm).

---

## Entity types

| Entity | Token | Examples |
|---|---|---|
| PERSON | `[PERSON]` | Patient names, doctor names, relative names |
| PHONE | `[PHONE]` | Thai mobile (06x/08x/09x), landlines, +66 international |
| EMAIL | `[EMAIL]` | Email addresses |
| ADDRESS | `[ADDRESS]` | Full structured addresses (≥2 components required) |
| NATIONAL_ID | `[NATIONAL_ID]` | Thai 13-digit ID, dashed format (1-XXXX-XXXXX-XX-X) |
| HOSPITAL_IDS | `[HOSPITAL_IDS]` | HN, AN, VN numbers |

> **DATE is intentionally excluded** — clinical visit dates are not PHI and redacting them destroys clinical meaning.

---

## Architecture

```
Input text
    │
    ▼
[1] Regex pre-processing  (fast, deterministic)
    ├─ HN / AN / VN  →  [HOSPITAL_IDS]
    ├─ 13-digit national ID (bare + dashed)  →  [NATIONAL_ID]
    ├─ Thai-titled names (nai/nang/nangsaw/dekkai/dekyling/nph/phya...)  →  [PERSON]
    ├─ Relative names (marada/bida/sami/phanraya/but/phuduai + name)  →  [PERSON]
    ├─ English-titled names: Mr./Mrs./Miss/Ms./Dr./Prof.  →  [PERSON]
    ├─ Phone (labeled with Thai/English prefix)  →  [PHONE]
    ├─ Phone (standalone mobile 06x/08x/09x with/without separators)  →  [PHONE]
    ├─ Phone (Bangkok landline 02-XXXX-XXXX)  →  [PHONE]
    ├─ Phone (provincial landline 0XX-XXX-XXXX)  →  [PHONE]
    ├─ Phone (+66 international format)  →  [PHONE]
    ├─ LINE ID (Thai/English prefix required)  →  [PERSON]
    ├─ Email  →  [EMAIL]
    ├─ Thai digit normalization (Thai numerals to Arabic)
    └─ Thai full-name gazetteer (PyThaiNLP first + family corpora)  →  [PERSON]
        - Catches untitled names like "ปฐวี คมกฤช"
        - Conservative: requires BOTH halves to match (either order)
    │
    ▼
[2] NER model inference  (loolootech/no-name-ner-th)
    ├─ Batched token classification (batch_size=64 on GPU, 16 on CPU)
    ├─ Sub-token merging (B-/I- tags -> full spans)
    └─ Only entities in ENTITY_MAP are kept (DATE suppressed)
    │
    ▼
[3] False-positive suppression
    ├─ Medical eponym blocklist (80+ terms: Foley, Tenckhoff, Candida, Rankin...)
    ├─ Organism genus rule: [A-Z][a-z]+(us|ia|coccus|bacillus...) -> not PERSON
    ├─ English medical verb blocklist (retain, wean, extubate, taper...)
    ├─ Low-confidence short-span filter: PERSON score < 0.70 AND span <= 3 chars
    └─ ADDRESS suppression: institution prefix OR < 2 structural components
    │
    ▼
[4] Entity replacement
    └─ Reverse-order substitution to preserve character offsets
    │
    ▼
De-identified text
```

---

## Example

**Before:**

> นายสมชาย ใจดี 55 ปี  HN 37838/59  
> Tel: 081-234-5678  LINE ID: somchai.md  
> มารดา นางสมหญิง ใจงาม  
> Known case DM, HT  
> เลขบัตรประชาชน 3-1001-04566-72-1  
> ส่งต่อ พญ.วิภา แผนกอายุรกรรม รพ.จุฬาลงกรณ์

**After:**

> [PERSON] 55 ปี  HN [HOSPITAL_IDS]  
> Tel: [PHONE]  LINE ID: [PERSON]  
> มารดา [PERSON]  
> Known case DM, HT  
> เลขบัตรประชาชน [NATIONAL_ID]  
> ส่งต่อ [PERSON] แผนกอายุรกรรม รพ.จุฬาลงกรณ์

Note: รพ.จุฬาลงกรณ์ is correctly NOT redacted — institution names are excluded from ADDRESS detection.

---

## Files

| File | Purpose |
|---|---|
| `deidentify_nun.py` | Main pipeline — processes 4-sheet hospital CSV export |
| `fix_regex.py` | Local post-processing — applies regex fixes to already-NER-processed CSVs without re-running the model |
| `write_job.py` | Generates `deidentify_nun.job` cleanly (avoids heredoc indentation issues) |
| `deidentify_nun.job` | Slurm job script for HPC cluster (A100 GPU, Singularity container) |
| `deidentify_colab.ipynb` | Google Colab notebook for smaller datasets |

---

## Setup

### Local / Colab

```bash
pip install -r requirements.txt
# transformers[torch] sentencepiece pandas openpyxl tqdm pythainlp
```

Python 3.9+, ~500 MB disk for the NER model + ~few MB for PyThaiNLP name corpora (downloaded on first run).

### HPC Cluster (Slurm + Singularity)

```bash
# 1. Upload files to cluster
scp deidentify_nun.py write_job.py user@cluster:/data/home/user/deiden/

# 2. Convert Excel sheets to CSV first (saves memory)
python3 -c "
import pandas as pd
xl = pd.ExcelFile('nun.xlsx')
for sheet in xl.sheet_names:
    xl.parse(sheet).to_csv(f'nun_{sheet}.csv', index=False)
    print(f'Saved nun_{sheet}.csv')
"

# 3. Generate and submit job
python3 write_job.py
sbatch deidentify_nun.job
```

---

## Usage — `deidentify_nun.py`

Configure the sheet-to-CSV mapping and text columns at the top of the file:

```python
SHEET_CSV_FILES = {
    "Admission_Record":  "nun_Admission_Record.csv",
    "IPT_PROGRESSNOTE":  "nun_IPT_PROGRESSNOTE.csv",
    "OPD_PROGRESSNOTE":  "nun_OPD_PROGRESSNOTE.csv",
    "Radiology":         "nun_Radiology.csv",
}

SHEET_TEXT_COLUMNS = {
    "OPD_PROGRESSNOTE": [
        "CHIEFCOMPLAIN", "HISTORY", "PHYSICALEXAM", "MANAGEMENT", "PLAN",
        "RECOMMENDATION", "NURSENOTE", "PROGRESSNOTE", "MANAGEPLAN", ...
    ],
    ...
}
```

Run:

```bash
python3 deidentify_nun.py
```

**Output:**
- `nun_deidentified_{Sheet}.csv` — de-identified data per sheet
- `nun_detections_log.csv` — detection log (entity type, position, score — no original PHI)

**Checkpoint/resume:** Progress is saved to `nun_checkpoint.json` after each 25,000-row chunk. Re-run to resume after interruption. The checkpoint is version-keyed (`PIPELINE_VERSION`) — changing the pipeline code auto-discards stale checkpoints and forces a clean run.

---

## Usage — `fix_regex.py`

When you have already-NER-processed CSVs and want to apply updated regex patterns without re-running the model (saves GPU hours):

```bash
python3 fix_regex.py
```

Reads `nun_deidentified_*.csv`, writes `nun_fixed_*.csv`. Also fixes `_x000D_` carriage-return artifacts from Excel exports.

---

## False positive handling

### Medical eponym blocklist (~80 terms)

Suppresses PERSON detections on device/procedure/syndrome eponyms and English medical verbs:

```
Device:    foley, hickman, tenckhoff, quinton, shiley, dobhoff, penrose…
Signs:     babinski, kernig, romberg, phalen, tinel, lhermitte…
Scales:    glasgow, rankin, barthel, braden, morse, apgar, wells…
Organisms: candida, klebsiella, pseudomonas, aspergillus, nocardia…
Verbs:     retain, wean, extubate, taper, flush, titrate, mobilize…
```

### Organism genus rule

Automatically suppresses capitalized Latin genus names matching:

```
^[A-Z][a-z]+(us|ia|ella|monas|coccus|bacillus|bacterium|virus)$
```

Catches all future genera without per-species blocklist entries.

### Thai full-name gazetteer (PyThaiNLP)

Many real names — especially uncommon Thai surnames like `คมกฤช` — are missed by the NER model when no title prefix (`นาย`, `พญ.`, etc.) is present. To recover these without sacrificing precision, the pipeline uses **PyThaiNLP**'s name corpora as a deterministic post-regex layer:

- Sources: `thai_male_names()`, `thai_female_names()`, `thai_family_names()` (~3 chars minimum to drop ambiguous tokens)
- Scans Thai-script tokens (3–15 chars) with a sliding window
- Masks an adjacent token pair only when **both** halves match the corpora — first+family OR family+first (some clinical notes flip the order)
- Common Thai phrases (e.g. `ผู้ป่วยมา OPD`) are not flagged because at most one half — usually neither — appears in the name dictionaries

Graceful fallback: if PyThaiNLP is unavailable, this layer is silently skipped and the pipeline reverts to the regex+NER behavior.

### ADDRESS suppression

NER-detected ADDRESS spans are suppressed if:
1. An institution keyword appears within ±30 characters (โรงพยาบาล, คลินิก, แผนก, ICU, ward…), OR
2. Fewer than 2 structural Thai address components are present (หมู่, ซอย, ถนน, ตำบล/แขวง, อำเภอ/เขต, จังหวัด, postal code)

Single geographic words (เขตบางรัก, จ.เชียงใหม่ alone) are not flagged.

---

## Performance

Tested on 4-sheet hospital export (~915,000 rows total):

| Sheet | Rows | GPU time (A100) |
|---|---|---|
| Admission_Record | 27,000 | ~8 min |
| IPT_PROGRESSNOTE | 274,000 | ~45 min |
| OPD_PROGRESSNOTE | 595,000 | ~90 min |
| Radiology | 15,000 | ~5 min |

Memory: 25,000-row chunks with incremental CSV write — stable at < 8 GB RAM.

---

## Known limitations

- Names without a title prefix rely on the PyThaiNLP gazetteer + NER model. Pairs where neither token appears in the Thai name corpora are still missed
- Single untitled names (one token only) are not gazetteer-masked to avoid false positives on common Thai words
- Very short spans (≤3 chars) with score < 0.70 are suppressed to reduce noise
- โทร as a verb ("called to inform") is not flagged — only โทร followed by digits within 8 characters is treated as a phone prefix
- NER model license (CC BY-NC 4.0) restricts commercial use

---

## Model

[loolootech/no-name-ner-th](https://huggingface.co/loolootech/no-name-ner-th) — CamembertForTokenClassification based on [clicknext/phayathaibert](https://huggingface.co/clicknext/phayathaibert), fine-tuned for Thai medical NER.

Detects: `PERSON`, `PHONE`, `EMAIL`, `ADDRESS`, `DATE`, `NATIONAL_ID`, `HOSPITAL_IDS`

License: CC BY-NC 4.0. Commercial use: contact `contact@looloohealth.com`.

---

## License

Pipeline code: MIT. Underlying NER model: CC BY-NC 4.0 (non-commercial).
