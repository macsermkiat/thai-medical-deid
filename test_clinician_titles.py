"""Clinician names after แพทย์ / R1 titles (blood_audit #273).

Run: uv run --with pytest --with 'transformers[torch]' --with sentencepiece \
     --with pythainlp --with pandas --with tqdm --with openpyxl pytest -q test_clinician_titles.py
The names below are generic gazetteer entries, not names from any note.
"""
import pytest

import deidentify_nun as d


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        # Both spellings of แพทย์: the name is masked, the clinical text kept.
        ("รายงานเเพทย์สมชายรับทราบ", "รายงานเเพทย์[PERSON]รับทราบ"),
        ("รายงานแพทย์ สมหญิง ให้ LPRC 2 unit", "รายงานแพทย์ [PERSON] ให้ LPRC 2 unit"),
        # Resident and fellow titles, with and without a space.
        ("แจ้ง R1สมชาย รับทราบ", "แจ้ง R1[PERSON] รับทราบ"),
        ("แจ้ง R2 สมหญิง", "แจ้ง R2 [PERSON]"),
        ("Fellow สมชาย มาดู", "Fellow [PERSON] มาดู"),
        # A gazetteer name longer than a stop word is still a name.
        ("แพทย์มาลีรับทราบ", "แพทย์[PERSON]รับทราบ"),
        # A name outside the gazetteer runs to the next stop word.
        ("แพทย์ฆฌญฎรับทราบ", "แพทย์[PERSON]รับทราบ"),
        # PR #4 review: a titled full name is masked whole, surname included.
        ("แจ้ง R1 สมชาย ทองดี รับทราบ", "แจ้ง R1 [PERSON] รับทราบ"),
        ("แจ้ง R1 ฆฌญฎ ทองดี รับทราบ", "แจ้ง R1 [PERSON] รับทราบ"),
        # A name longer than its gazetteer prefix is masked to its end.
        ("แจ้งแพทย์มาลีวรรณรับทราบ", "แจ้งแพทย์[PERSON]รับทราบ"),
        ("แจ้งแพทย์สมชายศรีรับทราบ", "แจ้งแพทย์[PERSON]รับทราบ"),
        # A 2-letter stop word inside an unknown name does not cut it short.
        ("แจ้งแพทย์อุมาวดีรับทราบ", "แจ้งแพทย์[PERSON]รับทราบ"),
        # Names after a role qualifier, lowercase titles, a zero-width space.
        ("แจ้งแพทย์เวร สมชาย รับทราบ", "แจ้งแพทย์เวร [PERSON] รับทราบ"),
        ("แจ้ง r1 สมชาย", "แจ้ง r1 [PERSON]"),
        ("แจ้งแพทย์\u200bสมชายรับทราบ", "แจ้งแพทย์\u200b[PERSON]รับทราบ"),
    ],
)
def test_masks_name_after_clinician_title(note, expected):
    assert d.regex_preprocess(note) == expected


@pytest.mark.parametrize(
    "note",
    [
        # แพทย์ as an ordinary noun: masking here would erase the clinical
        # text the audit reads (who acted, what was ordered).
        "รายงานแพทย์รับทราบ ให้ LPRC 2 unit",
        "แจ้งแพทย์เวร",
        "แพทย์ให้ off NSS",
        "แพทย์มาดูอาการ",
        "นัดพบแพทย์เจ้าของไข้",
        "ศัลยแพทย์ประเมินแผล",
        "R1เวรรับทราบ",
        # ASCII tokens that merely contain R1 / F1.
        "PR1สูง",
        # PR #4 review: audit-relevant verbs after the title survive.
        "แพทย์อนุมัติให้เลือด",
        "แพทย์มีคำสั่งให้ off",
        "แพทย์ผ่าตัดเสร็จ",
        # A title ending a line does not mask the next line.
        "รายงานแพทย์\nวันที่ 2 อาการดี",
        "รายงานแพทย์\nเวลา 10.00 น. ผู้ป่วยหลับ",
    ],
)
def test_keeps_ordinary_text_after_title(note):
    assert d.regex_preprocess(note) == note


def test_existing_title_after_แพทย์_stays_readable():
    assert d.regex_preprocess("แจ้งแพทย์ นพ.สมชาย") == "แจ้งแพทย์ นพ.[PERSON]"
