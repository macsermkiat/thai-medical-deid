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
PIPELINE_VERSION = "v5-2026-11-clinician-titles"  # bump whenever regex/model logic changes to force re-run

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
# AN/VN: uppercase pattern is separator-optional (full original coverage: AN 12345, AN:9117/68).
# A second case-insensitive pattern with required separator catches OCR/lowercase variants
# (an:9117) without matching the English article "an" before a plain number.
AN_PATTERN    = re.compile(r'\bAN\s*[\.:;\-]?\s*(\d{4,8}(?:/\d{2,4})?)')
AN_PATTERN_CI = re.compile(r'(?i)\bAN\s*[\.:;\-]\s*(\d{4,8}(?:/\d{2,4})?)')
VN_PATTERN    = re.compile(r'\bVN\s*[\.:;\-]?\s*(\d{4,8}(?:/\d{2,4})?)')
VN_PATTERN_CI = re.compile(r'(?i)\bVN\s*[\.:;\-]\s*(\d{4,8}(?:/\d{2,4})?)')

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
# รพ must be followed by "." or " " to avoid matching inside words like การพยาบาล.
PLACE_PATTERN = re.compile(
    r'((?:โรงพยาบาล|รพ\.)[ ]?)([ก-๙]{0,15})'
    r'|(รพ[ ])([ก-๙]{1,15})'
)

# ── Administrative locations (province/district/subdistrict) ─────────────────
# Prefix (group 1) is kept; only the trailing place name (group 2) is redacted.
# Abbreviations require >=2 following Thai chars so they don't eat month
# abbreviations (ต.ค.=Oct). อ. is intentionally EXCLUDED here: in these clinical
# notes "อ.<name>" is overwhelmingly อาจารย์ (a person title), so it is handled
# by THAI_TITLE_PATTERN as อ.[PERSON] rather than mislabelled as a place.
LOCATION_PATTERN = re.compile(
    r'((?:จังหวัด|อำเภอ|ตำบล|เขต|แขวง|จ\.|ต\.)[ ]?)([ก-๙]{2,15})'
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
# Require a real anchor after the keyword — an explicit ID word, a ':'/'='
# separator, or an '@' handle. A bare space (old behaviour) matched the "line"
# inside common English words (borderline, baseline, central line) and ate the
# following word as a fake handle.
LINE_ID_PATTERN = re.compile(
    r'(?:ไลน์|\bLINE\b)'
    r'(?:'
    r'\s*(?:ID\b|ไอดี)\s*[:=]?\s*'   # explicit ID word
    r'|\s*[:=]\s*'                    # or a ':'/'=' separator
    r'|\s*(?=@)'                      # or an '@' handle immediately after
    r')'
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
    r'((?:'
    r'นางสาว|นาง|นาย'
    r'|เด็กชาย|เด็กหญิง'
    r'|ด\.(?:ช|ญ)\.'
    r'|(?:รศ|ผศ|ศ)\.(?:นพ|พญ)\.'
    r'|(?:นพ|พญ)\.'
    r'|อ\.'
    r')\s*)'
    r'([ก-๙a-zA-Z][ก-๙a-zA-Z\s]{1,30})',
)

# ── Clinician role titles (แพทย์, residents, fellows) ──────────────────────
# Words that follow a doctor title in KCMH nurse notes as ordinary text
# ("รายงานแพทย์รับทราบ", "R1เวร"). Survey of 150k IPDNRFOCUSDT rows, 2026-11-18.
CLINICIAN_TITLE_STOPWORDS = frozenset((
    'ให้', 'นัด', 'เจ้าของ', 'รับทราบ', 'ทราบ', 'เวร', 'ทันที', 'มา', 'มื้อ',
    'แจ้ง', 'ทำ', 'หาก', 'ก่อน', 'คุย', 'สั่ง', 'ประเมิน', 'วิสัญญี', 'และ',
    'ขีด', 'เมื่อ', 'โทร', 'ปรับ', 'พิจารณา', 'จึง', 'ยัง', 'ดูแล', 'เพื่อ',
    'จะ', 'แล้ว', 'ที่', 'ดู', 'ตาม', 'พา', 'เรื่อง', 'ฟัง', 'อนุญาต', 'ได้',
    'ทุก', 'ประจำ', 'ผู้', 'เฉพาะ', 'ศัลย', 'อายุร', 'กุมาร', 'สูติ', 'ออร์โธ',
    'ระบบ', 'โรค', 'จักษุ', 'รังสี', 'ฉุกเฉิน', 'เจาะ', 'ใส่', 'ถอด', 'เปลี่ยน',
    'ตรวจ', 'เยี่ยม', 'รับ', 'ส่ง', 'อธิบาย', 'ยืนยัน', 'ไม่', 'ว่า', 'อีก',
    'ทาง', 'หัวหน้า', 'พยาบาล', 'ใช้', 'เข้า', 'ออก', 'ขอ', 'สอบถาม', 'เห็น',
    'บอก', 'งด', 'หยุด', 'เพิ่ม', 'ลด', 'วินิจฉัย', 'ระบุ', 'บันทึก', 'ลง',
    'เขียน', 'กลับ', 'อยู่', 'กำลัง', 'ยา', 'เลือด', 'เวชศาสตร์', 'แผน',
    'ทั่วไป', 'นักศึกษา', 'ทำแผล', 'สาย', 'เพิ่มเติม', 'ต่อ', 'ซึ่ง', 'หรือ',
    'กับ', 'ใน', 'จาก', 'คน', 'ท่าน', 'แต่', 'ขณะ', 'หลัง', 'ระหว่าง', 'โดย',
    'กรณี', 'คิด', 'นึกถึง', 'ร่วม', 'แพทย์', 'หน่วย', 'แนะนำ', 'เช็ดตัว',
    'ต้อง', 'ติด', 'เปิด', 'ปิด', 'กายภาพ', 'ฉีด', 'เก็บ', 'วาง', 'ดังนี้',
    'ผสม', 'คีย์', 'ทีม', 'กรอก', 'ถ้า', 'เกี่ยว', 'จิตเวช', 'เหน็บ', 'คือ',
    'อยาก', 'มี', 'อนุมัติ', 'ผ่าตัด', 'รักษา', 'ปฏิเสธ', 'ตัดสินใจ', 'ห้อง',
    'ห้าม', 'อาวุโส', 'นิติเวช', 'สภา', 'พี่เลี้ยง', 'สัมภาษณ์',
    # A title already in the note ('แพทย์ นพ.[PERSON]') stays readable.
    'นพ.', 'พญ.', 'ผศ.', 'รศ.', 'ศ.',
))
# Same-line separators only: a title ending a line must not mask the next
# line's first word. U+200B is a common invisible Thai word break.
_TITLE_RE = re.compile(
    r'(?:(?:แพทย์|เเพทย์)(?:เวร|เจ้าของไข้|ประจำบ้าน)?'
    r'|(?<![A-Za-z])(?:[Rr][1-4]|[Ff][1-3]|[Ff]ellow))'
    r'[ \t\xa0\u200b]*(?=[ก-ฮเ-ไ])'
)
_NAME_CHAR = re.compile(r'[ก-ฺเ-ํ]')
_SURNAME_GAP = re.compile(r'[ \t\xa0\u200b]+')
_NAME_MAX = 20


def _longest_prefix(text: str, words: frozenset | set) -> int:
    return next(
        (n for n in range(min(len(text), _NAME_MAX), 1, -1) if text[:n] in words),
        0,
    )


def _clinician_name_end(text: str, start: int) -> int:
    """End of the name that starts at ``start``, or ``start`` if none.

    A gazetteer first name is the minimum; the name then runs on to the next
    stop word or non-letter, at most _NAME_MAX characters. Without a
    gazetteer anchor only stop words of 3+ characters end the name, so a
    2-character stop word inside a name ('อุมาวดี') does not cut it short.
    A gazetteer surname after a space is included.
    """
    gaz = _longest_prefix(text[start:start + _NAME_MAX], _THAI_FIRST_NAMES)
    end = start + gaz
    while end < start + _NAME_MAX and end < len(text) and _NAME_CHAR.match(text[end]):
        stop = _longest_prefix(text[end:end + _NAME_MAX], CLINICIAN_TITLE_STOPWORDS)
        if end - start >= 2 and stop and (gaz or stop >= 3):
            break
        end += 1
    if end - start < 2:
        return start
    gap = _SURNAME_GAP.match(text, end)
    if gap:
        family = _longest_prefix(text[gap.end():gap.end() + _NAME_MAX], _THAI_FAMILY_NAMES)
        if family:
            end = gap.end() + family
    return end


def _mask_clinician_title_names(text: str) -> str:
    """Mask a name after a clinician title, keeping the title.

    'แพทย์' (and its common misspelling 'เเพทย์') is mostly an ordinary noun
    in nurse notes, so a following known word is left alone. A gazetteer
    first name longer than that word wins ('แพทย์มาลี' is a name, 'แพทย์มาดู'
    is not). Issue: blood_audit #273.
    """
    out, pos = [], 0
    for m in _TITLE_RE.finditer(text):
        start = m.end()
        if start < pos:
            continue
        tail = text[start:start + _NAME_MAX]
        stop = _longest_prefix(tail, CLINICIAN_TITLE_STOPWORDS)
        if stop and stop >= _longest_prefix(tail, _THAI_FIRST_NAMES):
            continue
        end = _clinician_name_end(text, start)
        if end == start:
            continue
        out.append(text[pos:start])
        out.append('[PERSON]')
        pos = end
    out.append(text[pos:])
    return ''.join(out)


# ── English-titled names ──────────────────────────────────────────────────────
NAME_ENGLISH_TITLE_PATTERN = re.compile(
    r'((?:Mrs?\.|Miss|Ms\.|Dr\.|Prof\.)\s+)'
    r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})'
)


def _keep_prefix(token: str):
    """Substitution that preserves a structural prefix (group 1) and redacts
    only the trailing name (group 2), e.g. 'อ.สมชาย' -> 'อ.[PERSON]'."""
    def repl(m):
        name = m.group(2)
        if name and name.strip():
            return m.group(1) + token
        return m.group(0)
    return repl


def _place_repl(m):
    """PLACE_PATTERN has two prefix/name group pairs (1,2) and (3,4)."""
    if m.group(1) is not None:
        prefix, name = m.group(1), m.group(2)
    else:
        prefix, name = m.group(3), m.group(4)
    if name and name.strip():
        return prefix + '[PLACE]'
    return m.group(0)


def regex_preprocess(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    # Normalize Thai digits to ASCII
    text = text.translate(_THAI_DIGIT_TABLE)

    # Hospital identifiers
    text = HN_PATTERN.sub(r'HN [HOSPITAL_IDS]', text)
    text = AN_PATTERN.sub(r'AN [HOSPITAL_IDS]', text)
    text = AN_PATTERN_CI.sub(r'AN [HOSPITAL_IDS]', text)
    text = VN_PATTERN.sub(r'VN [HOSPITAL_IDS]', text)
    text = VN_PATTERN_CI.sub(r'VN [HOSPITAL_IDS]', text)

    # Age (อายุ … ปี), place/hospital names, and administrative locations
    text = AGE_PATTERN.sub('อายุ [AGE] ปี', text)
    text = PLACE_PATTERN.sub(_place_repl, text)
    text = LOCATION_PATTERN.sub(_keep_prefix('[PLACE]'), text)

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

    # Named persons — titles anchor these patterns. Relative-name detection
    # (มารดา/บิดา/บุตร …) is delegated to the NER model: the regex greedily
    # grabbed trailing narrative as a name (e.g. "บุตร" + non-name text).
    text = THAI_TITLE_PATTERN.sub(_keep_prefix('[PERSON]'), text)
    text = NAME_ENGLISH_TITLE_PATTERN.sub(_keep_prefix('[PERSON]'), text)

    # Untitled Thai full-name pairs via gazetteer (catches NER misses
    # like 'ปฐวี คมกฤช' that have no preceding title)
    text = _gazetteer_replace_thai_name_pair(text)

    # After the pair pass, so a titled full name is masked whole.
    text = _mask_clinician_title_names(text)

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


# Regex-inserted replacement tokens, e.g. [PERSON], [PHONE], [HOSPITAL_IDS].
# The NER model otherwise re-tags the literal token text ("PERSON", "PHONE")
# as entities and corrupts them on replacement, so these regions are protected.
_PLACEHOLDER_RE = re.compile(r'\[[A-Z_]+\]')


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
        protected = [(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]
        detected = []
        anonymized = text

        for start, end, label, score in reversed(merged):
            if label not in ENTITY_MAP:
                continue
            # Skip spans overlapping a regex-inserted placeholder token.
            if any(start < pe and ps < end for ps, pe in protected):
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
