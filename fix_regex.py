"""
Local post-processing pass: apply updated regex patterns to already-downloaded
nun_deidentified_*.csv files without re-running the NER model.

Fixes:
- Thai title names (นาย/นาง/นพ./พญ.) missed by old code
- Mobile numbers with dashes/spaces (081-234-5678)
- Bangkok/provincial landlines
- All Thai phone prefixes (เบอร์โทร, มือถือ, ติดต่อ, etc.)
- LINE ID patterns
- AN/VN hospital identifiers
- Dashed national ID format
- English-titled names (Mrs./Mr./Dr.)
- Relative names (มารดา/บิดา/สามี)
- _x000D_ carriage return artifacts → newline
"""

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SHEET_CSV_FILES = {
    "Admission_Record": "nun_deidentified_Admission_Record.csv",
    "IPT_PROGRESSNOTE": "nun_deidentified_IPT_PROGRESSNOTE.csv",
    "OPD_PROGRESSNOTE": "nun_deidentified_OPD_PROGRESSNOTE.csv",
    "Radiology":        "nun_deidentified_Radiology.csv",
}

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

# ── Thai digit normalization ──────────────────────────────────────────────────
_THAI_DIGIT_TABLE = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# ── Hospital identifiers ──────────────────────────────────────────────────────
HN_PATTERN = re.compile(r"(?i)\bH[\s\-/]?N\s*[\.:;\-]?\s*(\d{4,8}(?:/\d{2,4})?)")
AN_PATTERN = re.compile(r"(?i)\bAN\s*[\.:;\-]?\s*(\d{6,12})")
VN_PATTERN = re.compile(r"(?i)\bVN\s*[\.:;\-]?\s*(\d{6,12})")

# ── National ID ───────────────────────────────────────────────────────────────
NATIONAL_ID_PATTERN = re.compile(r"(?<!\d)\d{13}(?!\d)")
NATIONAL_ID_DASHED  = re.compile(
    r"(?<!\d)\d{1}[-\s]\d{4}[-\s]\d{5}[-\s]\d{2}[-\s]\d{1}(?!\d)"
)

# ── Phone ─────────────────────────────────────────────────────────────────────
_S   = r"[-.\s]{0,2}"
_SEP = r"[^\d]{0,8}"

_MOBILE = rf"0[689]\d{_S}\d{{3}}{_S}\d{{4}}"
_BKK    = rf"0{_S}2{_S}\d{{3}}{_S}\d{{4}}"
_PROV   = rf"0{_S}[3-9]\d{_S}\d{{3}}{_S}\d{{3,4}}"
_INTL   = rf"\+66{_S}\d{{1,2}}{_S}\d{{3}}{_S}\d{{4}}"
_ANY    = rf"(?:{_INTL}|{_MOBILE}|{_BKK}|{_PROV})"

PHONE_LABELED = re.compile(
    rf"(?:"
    rf"โทรศัพท์(?:มือถือ)?"
    rf"|เบอร์(?:โทร(?:ศัพท์)?|ติดต่อ)?"
    rf"|หมายเลข(?:โทรศัพท์)?"
    rf"|มือถือ"
    rf"|โทร"
    rf"|ติดต่อ(?:ที่|ได้ที่)?"
    rf"|สายด่วน|สายตรง"
    rf"|แฟ[็]?กซ์"
    rf"|Tel(?:ephone)?|TEL|Phone|Mobile|Fax|FAX|HP|Contact"
    rf")"
    rf"{_SEP}({_ANY})",
    re.IGNORECASE,
)
PHONE_MOBILE_BARE = re.compile(rf"(?<![/\d])({_MOBILE})(?!\d)")
PHONE_BKK_BARE    = re.compile(rf"(?<!\d)({_BKK})(?!\d)")
PHONE_PROV_BARE   = re.compile(rf"(?<!\d)({_PROV})(?!\d)")
PHONE_PAREN       = re.compile(rf"\(0\d{{1,2}}\){_S}\d{{3,4}}{_S}\d{{3,4}}")

# ── LINE ID ───────────────────────────────────────────────────────────────────
LINE_ID_PATTERN = re.compile(
    r"(?:ไลน์|LINE|Line)"
    r"(?:\s*(?:ID|ไอดี|id))?"
    r"\s*[:=]?\s*"
    r"(@?[a-zA-Z0-9][a-zA-Z0-9._\-]{2,19})",
    re.IGNORECASE,
)

# ── Email ─────────────────────────────────────────────────────────────────────
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# ── Thai-titled names ─────────────────────────────────────────────────────────
THAI_TITLE_PATTERN = re.compile(
    r"(?:"
    r"นางสาว|นาง|นาย"
    r"|เด็กชาย|เด็กหญิง"
    r"|ด\.(?:ช|ญ)\."
    r"|(?:รศ|ผศ|ศ)\.(?:นพ|พญ)\."
    r"|(?:นพ|พญ)\."
    r"|อ\."
    r")\s*"
    r"([ก-๙a-zA-Z][ก-๙a-zA-Z\s]{1,30})",
)

# ── Relative names ────────────────────────────────────────────────────────────
RELATIVE_NAME_PATTERN = re.compile(
    r"(?:มารดา|บิดา|สามี|ภรรยา|บุตร|ผู้ดูแล)\s+"
    r"(?:นางสาว|นาง|นาย|เด็กชาย|เด็กหญิง)?\s*"
    r"([ก-๙][ก-๙\s]{2,30})",
)

# ── English-titled names ──────────────────────────────────────────────────────
NAME_ENGLISH_TITLE_PATTERN = re.compile(
    r"(?:Mrs?\.|Miss|Ms\.|Dr\.|Prof\.)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
)

# ── _x000D_ carriage return artifacts ────────────────────────────────────────
X000D_PATTERN = re.compile(r"_x000D_")


def fix_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    text = X000D_PATTERN.sub("\n", text)
    text = text.translate(_THAI_DIGIT_TABLE)

    text = HN_PATTERN.sub(r"HN [HOSPITAL_IDS]", text)
    text = AN_PATTERN.sub(r"AN [HOSPITAL_IDS]", text)
    text = VN_PATTERN.sub(r"VN [HOSPITAL_IDS]", text)

    text = NATIONAL_ID_DASHED.sub("[NATIONAL_ID]", text)
    text = NATIONAL_ID_PATTERN.sub("[NATIONAL_ID]", text)

    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = LINE_ID_PATTERN.sub("[PERSON]", text)

    text = PHONE_LABELED.sub("[PHONE]", text)
    text = PHONE_PAREN.sub("[PHONE]", text)
    text = PHONE_MOBILE_BARE.sub("[PHONE]", text)
    text = PHONE_BKK_BARE.sub("[PHONE]", text)
    text = PHONE_PROV_BARE.sub("[PHONE]", text)

    text = THAI_TITLE_PATTERN.sub("[PERSON]", text)
    text = RELATIVE_NAME_PATTERN.sub("[PERSON]", text)
    text = NAME_ENGLISH_TITLE_PATTERN.sub("[PERSON]", text)

    return text


def process_csv(sheet_name: str, csv_file: str):
    path = Path(csv_file)
    if not path.exists():
        print(f"  SKIP — {csv_file} not found")
        return

    cols = SHEET_TEXT_COLUMNS.get(sheet_name, [])
    out_path = Path(csv_file.replace("nun_deidentified_", "nun_fixed_"))
    if out_path.exists():
        out_path.unlink()

    print(f"\n=== {sheet_name} ===")
    first_chunk = True

    for chunk_df in tqdm(
        pd.read_csv(path, chunksize=25000, low_memory=False, dtype=str),
        desc="  chunks",
    ):
        for col in cols:
            if col not in chunk_df.columns:
                continue
            mask = chunk_df[col].notna() & (chunk_df[col].astype(str).str.strip() != "")
            chunk_df.loc[mask, col] = chunk_df.loc[mask, col].apply(fix_text)

        chunk_df.to_csv(out_path, mode="a", index=False, header=first_chunk)
        first_chunk = False

    print(f"  Written: {out_path}")


def spot_check():
    print("\n=== Spot check after fix ===")
    patterns = {
        "Bare mobile numbers": re.compile(r"0[689]\d[-. ]?\d{3}[-. ]?\d{4}"),
        "Thai titles":         re.compile(r"(?:นาย|นางสาว|นาง|นพ\.|พญ\.)[ก-๙a-zA-Z]{2,}"),
    }
    for label, pat in patterns.items():
        count = 0
        for f in sorted(Path(".").glob("nun_fixed_*.csv")):
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    count += len(pat.findall(line))
        print(f"  {label}: {count:,}")


def main():
    for sheet_name, csv_file in SHEET_CSV_FILES.items():
        process_csv(sheet_name, csv_file)
    spot_check()
    print("\nDone. Final output: nun_fixed_*.csv")


if __name__ == "__main__":
    main()
