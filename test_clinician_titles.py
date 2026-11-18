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
    ],
)
def test_keeps_ordinary_text_after_title(note):
    assert d.regex_preprocess(note) == note
