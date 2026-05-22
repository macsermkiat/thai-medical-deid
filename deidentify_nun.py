"""
De-identification pipeline for nun.xlsx — all 4 sheets.

Improvements over v2:
- DATE removed from redaction (clinical dates are not PHI)
- ADDRESS post-filter: institution names suppressed, 2-component minimum required
- Phone regex overhauled: all Thai/English prefixes, separators, landlines, +66 international
- LINE ID labeled pattern
- Thai title names: นาย/นาง/นพ./พญ. and relative names (มารดา/บิดา/สามี)
- AN/VN hospital identifiers added; HN pattern improved
- Dashed national ID format (1-1007-04366-72-2)
- Organism genus rule: suppresses bacteria/fungi falsely flagged as PERSON
- Expanded eponym/verb blocklist (organisms, scales, catheter eponyms, verbs)
- Low-confidence short-span filter: PERSON score<0.70 on spans ≤3 chars suppressed
- Email regex backup
- Thai digit normalization (๐-๙ → 0-9)
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline


MODEL_ID = "loolootech/no-name-ner-th"

ENTITY_MAP = {
    "PERSON": "[PERSON]",
    "PHONE": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "ADDRESS": "[ADDRESS]",
    "NATIONAL_ID": "[NATIONAL_ID]",
    "HOSPITAL_IDS": "[HOSPITAL_IDS]",
    # DATE excluded — clinical dates are not PHI; DOB is rare in progress notes
}

SHEET_CSV_FILES = {
    "Admission_Record":  "nun_Admission_Record.csv",
    "IPT_PROGRESSNOTE":  "nun_IPT_PROGRESSNOTE.csv",
    "OPD_PROGRESSNOTE":  "nun_OPD_PROGRESSNOTE.csv",
    "Radiology":         "nun_Radiology.csv",
}
OUTPUT_FILE = "nun_deidentified.xlsx"
CHECKPOINT_FILE = "nun_checkpoint.json"
PIPELINE_VERSION = "v4-2026-05-gazetteer"  # bump whenever regex/model logic changes to force re-run

SHEET_TEXT_COLUMNS = {
    "Admission_Record": [
        "PROGDESC", "PROGLIST", "SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN",
    ],
    "IPT_PROGRESSNOTE": ["FOCUS", "ACTION", "RESPONSE"],
    "OPD_PROGRESSNOTE": [
        "CHIEFCOMPLAIN", "HISTORY", "PHYSICALEXAM", "MANAGEMENT", "PLAN",
        "RECOMMENDATION", "NURSENOTE", "PROGRESSNOTE", "MANAGEPLAN",
        "CHIEFCOMPLAIN2", "HISTORY2", "PHYSICALEXAM2", "MANAGEMENT2",
        "PLAN2", "RECOMMENDATION2", "PROGRESSNOTE2", "MANAGEPLAN2",
    ],
    "Radiology": [
        "RESULT", "RESULT2",
        "TEXTRSL1", "TEXTRSL2", "TEXTRSL3", "TEXTRSL4",
        "TEXTRSL5", "TEXTRSL6", "TEXTRSL7", "TEXTRSL8",
    ],
}

MEDICAL_EPONYM_BLOCKLIST = {
    # Device/procedure eponyms
    "foley", "hickman", "broviac", "dobhoff", "salem", "penrose",
    "jackson", "pratt", "levin", "ewald", "blakemore", "sengstaken",
    "linton", "minnesota", "cantor", "miller", "abbott", "mallory",
    "weiss", "heimlich", "trendelenburg", "valsalva", "romberg",
    "babinski", "kernig", "brudzinski", "trousseau", "chvostek",
    # Additional catheter/device eponyms
    "tenckhoff", "quinton", "shiley", "portex", "nelaton", "pezzer", "pleurx",
    # Sign/test eponyms
    "mantoux", "widal", "coombs", "phalen", "tinel", "spurling",
    "mcmurray", "lachman", "apley", "finkelstein", "hawkins", "neer",
    "lhermitte", "adson",
    # Scale/scoring eponyms
    "rankin", "barthel", "braden", "norton", "morse", "caprini",
    "wells", "apgar", "bishop", "glasgow",
    # Syndrome/disease eponyms
    "marfan", "turner", "klinefelter", "wilson", "huntington",
    "parkinson", "alzheimer", "cushing", "addison", "lyell", "reye",
    "goodpasture", "wegener", "sjogren", "behcet", "felty",
    # Organisms (high false-positive risk — genus names look like Thai person names)
    "candida", "klebsiella", "pseudomonas", "enterococcus",
    "staphylococcus", "streptococcus", "acinetobacter", "stenotrophomonas",
    "burkholderia", "nocardia", "cryptococcus", "aspergillus",
    "pneumocystis", "mycobacterium", "listeria",
    # English medical verbs falsely flagged as person names
    "retain", "remove", "insert", "replace", "change", "monitor",
    "continue", "start", "stop", "hold", "apply", "check", "drain",
    "refer", "follow", "admit", "discharge", "transfer", "consult",
    "review", "adjust", "control", "manage", "observe", "maintain",
    "wean", "taper", "flush", "clamp", "irrigate", "ambulate",
    "extubate", "intubate", "resuscitate", "defibrillate", "cardiovert",
    "titrate", "escalate", "mobilize", "restrict", "supplement",
    "reposition", "stabilize", "optimize",
}

# Organism genus pattern: suppresses PERSON on Latin genus names
_ORGANISM_GENUS_RE = re.compile(
    r'^[A-Z][a-z]+(us|ia|ella|monas|coccus|bacillus|bacterium|virus|mycetes)$'
)

# Institution prefixes — ADDRESS spans adjacent to these are suppressed
_INSTITUTION_RE = re.compile(
    r'โรงพยาบาล|รพ\.|คลินิก|ศูนย์(?:การแพทย์|สุขภาพ)|สถาบัน|'
    r'แผนก|ตึก|ห้องผ่าตัด|ห้องพัก|ward|ICU|CCU|NICU|SICU|MICU|'
    r'ER|OPD|IPD|IPT|OR|PACU',
    re.IGNORECASE,
)

# Address component keywords — need ≥2 to flag as true ADDRESS
_ADDR_COMPONENTS = [
    re.compile(r'(?:บ้านเลขที่|เลขที่)\s*\d{1,5}(?:/\d{1,3})?'),
    re.compile(r'(?:หมู่(?:ที่)?|ม\.)\s*\d{1,3}'),
    re.compile(r'(?:ซอย|ซ\.)\s*[\wก-๙\d]+'),
    re.compile(r'(?:ถนน|ถ\.)\s*[\wก-๙]+'),
    re.compile(r'(?:ตำบล|ต\.|แขวง)\s*[\wก-๙]+'),
    re.compile(r'(?:อำเภอ|อ\.|เขต)\s*[\wก-๙]+'),
    re.compile(r'(?:จังหวัด|จ\.)\s*[\wก-๙]+'),
    re.compile(r'(?<!\d)[1-9]\d{4}(?!\d)'),  # postal code
]


import torch
DEVICE     = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 64 if DEVICE == 0 else 16

# ── Thai digit normalization ──────────────────────────────────────────────────
_THAI_DIGIT_TABLE = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')

# ── Hospital identifiers ──────────────────────────────────────────────────────
HN_PATTERN = re.compile(
    r'(?i)\bH[\s\-/]?N\s*[\.:;\-]?\s*(\d{4,8}(?:/\d{2,4})?)'
)
# AN/VN share HN's shape: 4-8 digits, optional /YY suffix (e.g. 9117/68, 21114/67).
AN_PATTERN = re.compile(r'\bAN\s*[\.:;\-]?\s*(\d{4,8}(?:/\d{2,4})?)')
VN_PATTERN = re.compile(r'\bVN\s*[\.:;\-]?\s*(\d{4,8}(?:/\d{2,4})?)')

# ── National ID ───────────────────────────────────────────────────────────────
NATIONAL_ID_PATTERN = re.compile(r'(?<!\d)\d{13}(?!\d)')
NATIONAL_ID_DASHED  = re.compile(
    r'(?<!\d)\d{1}[-\s]\d{4}[-\s]\d{5}[-\s]\d{2}[-\s]\d{1}(?!\d)'
)

# ── Age ─────────────────────────────────────────────────────────────────────
# Anchored on อายุ … ปี so we don't redact every "N ปี" (durations, etc.).
# Space between number and ปี is sometimes missing (e.g. "อายุ 54ปี").
AGE_PATTERN = re.compile(r'อายุ\s*\d{1,3}\s*ปี')

# ── Place / hospital names ──────────────────────────────────────────────────
# Thai has no word spaces, so capture โรงพยาบาล/รพ. plus following Thai chars,
# bounded by the next non-Thai char (space/digit/Latin/punct) and capped at 15.
# May occasionally grab a trailing word or clip a long run-on name.
PLACE_PATTERN = re.compile(r'(?:โรงพยาบาล|รพ\.?)\s?[ก-๙]{0,15}')

# ── Administrative locations (province/district/subdistrict) ─────────────────
# Full words are unambiguous. Abbreviations จ./อ./ต. require >=2 following Thai
# chars so they don't eat month/day abbreviations (ต.ค.=Oct, จ.=Mon, อ.=Tue)
# or collide with the อ. (อาจารย์) person title. Note: a real "อ.<name>" teacher
# reference will be masked here as [PLACE] rather than [PERSON] — still redacted.
LOCATION_PATTERN = re.compile(
    r'(?:จังหวัด|อำเภอ|ตำบล|เขต|แขวง)\s?[ก-๙]{1,15}'
    r'|(?:จ|อ|ต)\.\s?[ก-๙]{2,15}'
)

# ── Phone number building blocks ──────────────────────────────────────────────
_S   = r'[-.\s]{0,2}'          # separator between digit groups
_SEP = r'[^\d]{0,8}'           # gap between label and first digit

_MOBILE   = rf'0[689]\d{_S}\d{{3}}{_S}\d{{4}}'
_BKK      = rf'0{_S}2{_S}\d{{3}}{_S}\d{{4}}'
_PROV     = rf'0{_S}[3-9]\d{_S}\d{{3}}{_S}\d{{3,4}}'
_INTL     = rf'\+66{_S}\d{{1,2}}{_S}\d{{3}}{_S}\d{{4}}'
_ANY      = rf'(?:{_INTL}|{_MOBILE}|{_BKK}|{_PROV})'

# Labeled phone: Thai + English prefixes
PHONE_LABELED = re.compile(
    rf'(?:'
    rf'โทรศัพท์(?:มือถือ)?'
    rf'|เบอร์(?:โทร(?:ศัพท์)?|ติดต่อ)?'
    rf'|หมายเลข(?:โทรศัพท์)?'
    rf'|มือถือ'
    rf'|โทร'
    rf'|ติดต่อ(?:ที่|ได้ที่)?'
    rf'|สายด่วน|สายตรง'
    rf'|แฟ[็]?กซ์'
    rf'|Tel(?:ephone)?|TEL|Phone|Mobile|Fax|FAX|HP|Contact'
    rf')'
    rf'{_SEP}({_ANY})',
    re.IGNORECASE,
)

# Standalone mobile (no prefix required)
PHONE_MOBILE_BARE = re.compile(rf'(?<![/\d])({_MOBILE})(?!\d)')

# Standalone Bangkok landline
PHONE_BKK_BARE = re.compile(rf'(?<!\d)({_BKK})(?!\d)')

# Standalone provincial landline
PHONE_PROV_BARE = re.compile(rf'(?<!\d)({_PROV})(?!\d)')

# Parenthesized area code: (02)123-4567
PHONE_PAREN = re.compile(rf'\(0\d{{1,2}}\){_S}\d{{3,4}}{_S}\d{{3,4}}')

# ── LINE ID ───────────────────────────────────────────────────────────────────
LINE_ID_PATTERN = re.compile(
    r'(?:ไลน์|LINE|Line)'
    r'(?:\s*(?:ID|ไอดี|id))?'
    r'\s*[:=]?\s*'
    r'(@?[a-zA-Z0-9][a-zA-Z0-9._\-]{2,19})',
    re.IGNORECASE,
)

# ── Email ─────────────────────────────────────────────────────────────────────
EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
)

# ── Thai name gazetteer (PyThaiNLP) ──────────────────────────────────────────
def _load_thai_name_corpora() -> tuple[set[str], set[str]]:
    """Load Thai first-name and family-name sets from PyThaiNLP.

    Returns (first_names, family_names). Returns empty sets if PyThaiNLP
    is unavailable or its corpora cannot be downloaded — pipeline degrades
    gracefully to NER-only behaviour.
    """
    try:
        from pythainlp.corpus.common import (
            thai_male_names,
            thai_female_names,
            thai_family_names,
        )
        firsts = thai_male_names() | thai_female_names()
        families = thai_family_names()
        # Drop very short tokens (≤2 chars) — too collision-prone with common Thai words
        firsts = {n for n in firsts if len(n) >= 3}
        families = {n for n in families if len(n) >= 3}
        return firsts, families
    except Exception as exc:  # ImportError, corpus download failure, etc.
        print(f"WARNING: PyThaiNLP name corpora unavailable ({exc}); "
              "Thai full-name gazetteer disabled.")
        return set(), set()


_THAI_FIRST_NAMES, _THAI_FAMILY_NAMES = _load_thai_name_corpora()

# Thai-script token (3-15 chars) — used to tokenize text for sliding-window
# pair detection. Sliding-window beats a fixed pair regex because re.sub
# consumes failed matches and would skip 'A B C' pairs when only B-C is real.
_THAI_TOKEN_RE = re.compile(r'[ก-๙]{3,15}')


def _gazetteer_replace_thai_name_pair(text: str) -> str:
    """Mask Thai untitled full-name pairs (first + family) using the gazetteer.

    Conservative: requires BOTH halves to match the corpora (in either order)
    so common Thai phrases don't get redacted. Catches patterns like
    'ปฐวี คมกฤช' that the NER model misses.
    """
    if not _THAI_FIRST_NAMES or not _THAI_FAMILY_NAMES:
        return text

    firsts = _THAI_FIRST_NAMES
    families = _THAI_FAMILY_NAMES

    tokens = list(_THAI_TOKEN_RE.finditer(text))
    spans: list[tuple[int, int]] = []

    i = 0
    while i < len(tokens) - 1:
        t1, t2 = tokens[i], tokens[i + 1]
        gap = text[t1.end(): t2.start()]
        if gap and gap.strip() == "":
            a, b = t1.group(), t2.group()
            if (a in firsts and b in families) or (a in families and b in firsts):
                spans.append((t1.start(), t2.end()))
                i += 2
                continue
        i += 1

    for start, end in reversed(spans):
        text = text[:start] + "[PERSON]" + text[end:]
    return text


# ── Thai-titled names (personal + professional) ───────────────────────────────
THAI_TITLE_PATTERN = re.compile(
    r'(?:'
    r'นางสาว|นาง|นาย'
    r'|เด็กชาย|เด็กหญิง'
    r'|ด\.(?:ช|ญ)\.'
    r'|(?:รศ|ผศ|ศ)\.(?:นพ|พญ)\.'
    r'|(?:นพ|พญ)\.'
    r'|อ\.'
    r')\s*'
    r'([ก-๙a-zA-Z][ก-๙a-zA-Z\s]{1,30})',
)

# ── Relative names ────────────────────────────────────────────────────────────
RELATIVE_NAME_PATTERN = re.compile(
    r'(?:มารดา|บิดา|สามี|ภรรยา|บุตร|ผู้ดูแล)\s+'
    r'(?:นางสาว|นาง|นาย|เด็กชาย|เด็กหญิง)?\s*'
    r'([ก-๙][ก-๙\s]{2,30})',
)

# ── English-titled names ──────────────────────────────────────────────────────
NAME_ENGLISH_TITLE_PATTERN = re.compile(
    r'(?:Mrs?\.|Miss|Ms\.|Dr\.|Prof\.)\s+'
    r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})'
)


def regex_preprocess(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    # Normalize Thai digits to ASCII
    text = text.translate(_THAI_DIGIT_TABLE)

    # Hospital identifiers
    text = HN_PATTERN.sub(r'HN [HOSPITAL_IDS]', text)
    text = AN_PATTERN.sub(r'AN [HOSPITAL_IDS]', text)
    text = VN_PATTERN.sub(r'VN [HOSPITAL_IDS]', text)

    # Age (อายุ … ปี), place/hospital names, and administrative locations
    text = AGE_PATTERN.sub('อายุ [AGE] ปี', text)
    text = PLACE_PATTERN.sub('[PLACE]', text)
    text = LOCATION_PATTERN.sub('[PLACE]', text)

    # National ID
    text = NATIONAL_ID_DASHED.sub('[NATIONAL_ID]', text)
    text = NATIONAL_ID_PATTERN.sub('[NATIONAL_ID]', text)

    # Email (before phone to avoid @-handle confusion)
    text = EMAIL_PATTERN.sub('[EMAIL]', text)

    # LINE ID (before bare phone to avoid digit overlap)
    text = LINE_ID_PATTERN.sub('[PERSON]', text)

    # Phone numbers — labeled first (most specific), then standalone
    text = PHONE_LABELED.sub('[PHONE]', text)
    text = PHONE_PAREN.sub('[PHONE]', text)
    text = PHONE_MOBILE_BARE.sub('[PHONE]', text)
    text = PHONE_BKK_BARE.sub('[PHONE]', text)
    text = PHONE_PROV_BARE.sub('[PHONE]', text)

    # Named persons — titles anchor these patterns
    text = THAI_TITLE_PATTERN.sub('[PERSON]', text)
    text = RELATIVE_NAME_PATTERN.sub('[PERSON]', text)
    text = NAME_ENGLISH_TITLE_PATTERN.sub('[PERSON]', text)

    # Untitled Thai full-name pairs via gazetteer (catches NER misses
    # like 'ปฐวี คมกฤช' that have no preceding title)
    text = _gazetteer_replace_thai_name_pair(text)

    return text


def load_ner_pipeline():
    print(f"Loading NER model: {MODEL_ID} ...")
    print(f"Device: {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}, batch_size={BATCH_SIZE}")
    ner = pipeline(
        "token-classification",
        model=MODEL_ID,
        device=DEVICE,
        aggregation_strategy="simple",
    )
    print("Model loaded.")
    return ner


def merge_entities(entities):
    if not entities:
        return []
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    merged = []
    for ent in sorted_ents:
        label = ent["entity_group"]
        start = ent["start"]
        end   = ent["end"]
        score = ent["score"]
        if merged and merged[-1][2] == label and start <= merged[-1][1] + 1:
            prev_start, prev_end, prev_label, prev_score = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_label, max(prev_score, score))
        else:
            merged.append((start, end, label, score))
    return merged


def _is_suppressed(span: str, label: str, score: float, text: str, start: int, end: int) -> bool:
    """Return True if this NER detection should be suppressed (false positive)."""

    if label == "PERSON":
        # Low-confidence very short span
        if score < 0.70 and (end - start) <= 3:
            return True
        # Medical eponym blocklist (exact lowercase match)
        if span.strip().lower() in MEDICAL_EPONYM_BLOCKLIST:
            return True
        # Organism genus pattern (e.g. Klebsiella, Aspergillus)
        if _ORGANISM_GENUS_RE.match(span.strip()):
            return True

    if label == "ADDRESS":
        # Suppress if near institution prefix
        context = text[max(0, start - 30): end + 20]
        if _INSTITUTION_RE.search(context):
            return True
        # Suppress if fewer than 2 structural address components in span
        component_count = sum(
            1 for pat in _ADDR_COMPONENTS if pat.search(span)
        )
        if component_count < 2:
            return True

    return False


def anonymize_batch(ner_pipeline, texts: list) -> list:
    preprocessed = [
        regex_preprocess(t) if t and isinstance(t, str) else t
        for t in texts
    ]

    valid_indices = [
        i for i, t in enumerate(preprocessed)
        if t and isinstance(t, str) and t.strip()
    ]
    valid_texts = [preprocessed[i] for i in valid_indices]

    ner_results = {}
    if valid_texts:
        batch_out = ner_pipeline(valid_texts, batch_size=BATCH_SIZE)
        if valid_texts and not isinstance(batch_out[0], list):
            batch_out = [batch_out]
        for i, entities in zip(valid_indices, batch_out):
            ner_results[i] = entities

    results = []
    for i, text in enumerate(preprocessed):
        if i not in ner_results or not text or not isinstance(text, str) or not text.strip():
            results.append((preprocessed[i] if isinstance(preprocessed[i], str) else texts[i], []))
            continue

        merged   = merge_entities(ner_results[i])
        detected = []
        anonymized = text

        for start, end, label, score in reversed(merged):
            if label not in ENTITY_MAP:
                continue
            original_span = text[start:end]
            if _is_suppressed(original_span, label, score, text, start, end):
                continue
            token = ENTITY_MAP[label]
            anonymized = anonymized[:start] + token + anonymized[end:]
            detected.append({
                "entity_type": label,
                "original":    original_span,
                "start":       start,
                "end":         end,
                "score":       round(score, 4),
                "replacement": token,
            })

        detected.reverse()
        results.append((anonymized, detected))

    return results


def load_checkpoint() -> set:
    path = Path(CHECKPOINT_FILE)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if data.get("version") != PIPELINE_VERSION:
            print(f"Pipeline version changed ({data.get('version')} → {PIPELINE_VERSION}) — discarding stale checkpoint.")
            path.unlink()
            return set()
        return set(tuple(x) for x in data.get("done", []))
    return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"version": PIPELINE_VERSION, "done": [list(x) for x in done]}, f)


def process_sheet(ner, df: pd.DataFrame, sheet_name: str, done: set, log_path: Path, write_header: bool):
    cols = [c for c in SHEET_TEXT_COLUMNS.get(sheet_name, []) if c in df.columns]
    total_entity_counts: dict = {}

    for col in cols:
        key = (sheet_name, col)
        if key in done:
            print(f"  [{sheet_name}] {col}: already done, skipping.")
            continue

        df[col] = df[col].astype(object)
        mask    = df[col].notna() & (df[col].astype(str).str.strip() != "")
        indices = df.index[mask].tolist()
        print(f"\n  [{sheet_name}] {col}: {len(indices)} non-empty rows")

        col_detections = []

        for batch_start in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"    {col}"):
            batch_indices = indices[batch_start: batch_start + BATCH_SIZE]
            batch_texts   = [str(df.at[idx, col]) for idx in batch_indices]
            batch_results = anonymize_batch(ner, batch_texts)

            for idx, (anonymized, detected) in zip(batch_indices, batch_results):
                df.at[idx, col] = anonymized
                for det in detected:
                    total_entity_counts[det["entity_type"]] = (
                        total_entity_counts.get(det["entity_type"], 0) + 1
                    )
                    col_detections.append({
                        "sheet":      sheet_name,
                        "row_index":  idx,
                        "row_number": idx + 2,
                        "column":     col,
                        **{k: v for k, v in det.items() if k != "original"},
                    })

        if col_detections:
            log_df = pd.DataFrame(col_detections)
            log_df.to_csv(log_path, mode="a", index=False, header=write_header)
            write_header = False

        done.add(key)
        save_checkpoint(done)

    return df, total_entity_counts, write_header


ROW_CHUNK_SIZE = 25000


def main():
    import gc

    done = load_checkpoint()
    if done:
        print(f"Resuming — {len(done)} item(s) already done: {done}")

    ner = load_ner_pipeline()

    total_entity_counts: dict = {}
    log_path     = Path("nun_detections_log.csv")
    write_header = not log_path.exists()

    for sheet_name, csv_file in SHEET_CSV_FILES.items():
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"\nSkipping {sheet_name} — {csv_file} not found.")
            continue

        out_csv = Path(f"nun_deidentified_{sheet_name}.csv")
        # Delete stale output when starting fresh (no checkpoint resume)
        if not done and out_csv.exists():
            out_csv.unlink()
        first_chunk = not out_csv.exists()

        print(f"\n=== Sheet: {sheet_name} ===")

        if sheet_name not in SHEET_TEXT_COLUMNS:
            import shutil
            shutil.copy(csv_path, out_csv)
            print(f"  No text cols — copied as-is to {out_csv}")
            continue

        for chunk_idx, chunk_df in enumerate(
            pd.read_csv(csv_path, chunksize=ROW_CHUNK_SIZE, low_memory=False)
        ):
            chunk_key = (sheet_name, f"chunk_{chunk_idx}")
            if chunk_key in done:
                print(f"  chunk {chunk_idx}: skipping (already done).")
                first_chunk = False
                continue

            row_start = chunk_idx * ROW_CHUNK_SIZE
            print(f"  chunk {chunk_idx}: rows {row_start:,}–{row_start + len(chunk_df):,}")

            chunk_df, counts, write_header = process_sheet(
                ner, chunk_df, sheet_name, done, log_path, write_header
            )
            for etype, cnt in counts.items():
                total_entity_counts[etype] = total_entity_counts.get(etype, 0) + cnt

            chunk_df.to_csv(out_csv, mode="a", index=False, header=first_chunk)
            first_chunk = False

            done.add(chunk_key)
            save_checkpoint(done)
            print(f"  chunk {chunk_idx}: saved. Checkpoint updated.")

            del chunk_df
            gc.collect()

        print(f"  Written to {out_csv}")

    total = sum(total_entity_counts.values())
    print(f"\nTotal entities detected: {total}")
    for etype, count in sorted(total_entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    print(f"Output: nun_deidentified_*.csv")
    print(f"Detection log: {log_path}")

    Path(CHECKPOINT_FILE).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
