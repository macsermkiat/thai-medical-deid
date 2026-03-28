"""
Test: Pre-process HN patterns before NER model, then compare results.

Finds rows with 'HN <number>' patterns in progressnote,
applies regex-based replacement, then runs the NER model,
and writes a comparison file.
"""

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

MODEL_ID = "loolootech/no-name-ner-th"
INPUT_FILE = "PTPHYSICALEXAM 10-122025.xlsx"
OUTPUT_FILE = "hn_deiden_comparison.xlsx"
COLUMN = "progressnote"

# Matches: HN 35447/60, HN 8436260, HN155568, HN:123456, HN. 1234/56
HN_PATTERN = re.compile(
    r'(?i)\bHN\s*[\.:;]?\s*(\d{4,8}(?:/\d{2,4})?)'
)


def replace_hn_regex(text: str) -> tuple[str, list[dict]]:
    """Replace HN number patterns with [HOSPITAL_IDS] using regex.

    Returns (replaced_text, list of match details).
    """
    if not text or not isinstance(text, str):
        return text, []

    matches = list(HN_PATTERN.finditer(text))
    if not matches:
        return text, []

    detections = []
    result = text
    # Replace in reverse to preserve positions
    for m in reversed(matches):
        full_match = m.group(0)
        detections.append({
            "matched_text": full_match,
            "start": m.start(),
            "end": m.end(),
        })
        result = result[:m.start()] + "HN [HOSPITAL_IDS]" + result[m.end():]

    detections.reverse()
    return result, detections


def merge_entities(entities: list[dict]) -> list[tuple]:
    """Merge adjacent NER entities of the same type."""
    if not entities:
        return []

    sorted_ents = sorted(entities, key=lambda e: e["start"])
    merged = []

    for ent in sorted_ents:
        label = ent["entity_group"]
        start = ent["start"]
        end = ent["end"]
        score = ent["score"]

        if merged and merged[-1][2] == label and start <= merged[-1][1] + 1:
            prev_start, prev_end, prev_label, prev_score = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_label, max(prev_score, score))
        else:
            merged.append((start, end, label, score))

    return merged


ENTITY_MAP = {
    "PERSON": "[PERSON]",
    "PHONE": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "ADDRESS": "[ADDRESS]",
    "DATE": "[DATE]",
    "NATIONAL_ID": "[NATIONAL_ID]",
    "HOSPITAL_IDS": "[HOSPITAL_IDS]",
}


def anonymize_with_ner(ner_pipe, text: str) -> str:
    """Run NER model and replace detected entities."""
    if not text or not isinstance(text, str) or not text.strip():
        return text

    results = ner_pipe(text)
    merged = merge_entities(results)

    anonymized = text
    for start, end, label, _score in reversed(merged):
        token = ENTITY_MAP.get(label, f"[{label}]")
        anonymized = anonymized[:start] + token + anonymized[end:]

    return anonymized


def main() -> None:
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found")
        return

    # Read data
    print(f"Reading {INPUT_FILE} ...")
    df = pd.read_excel(input_path, engine="openpyxl")

    # Find rows with HN patterns
    mask = df[COLUMN].notna() & df[COLUMN].astype(str).str.contains(
        r'(?i)\bHN\s*[\.:;]?\s*\d{4,}', regex=True
    )
    hn_rows = df.loc[mask].copy()
    print(f"Found {len(hn_rows)} rows with HN patterns")

    # Cap at 30 rows for this test
    hn_rows = hn_rows.head(30)
    print(f"Processing {len(hn_rows)} rows ...")

    # Load NER model
    print(f"Loading NER model: {MODEL_ID} ...")
    ner = pipeline(
        "token-classification",
        model=MODEL_ID,
        device=-1,
        aggregation_strategy="simple",
    )
    print("Model loaded.")

    # Process each row: model-only vs regex-then-model
    rows_out = []
    for idx in tqdm(hn_rows.index, desc="Processing"):
        original = str(hn_rows.at[idx, COLUMN])

        # Path A: NER model only
        model_only = anonymize_with_ner(ner, original)

        # Path B: Regex pre-process, then NER model
        regex_replaced, regex_matches = replace_hn_regex(original)
        regex_then_model = anonymize_with_ner(ner, regex_replaced)

        # Check if model alone caught the HN
        hn_still_present_model = bool(HN_PATTERN.search(model_only))
        hn_still_present_combined = bool(HN_PATTERN.search(regex_then_model))

        rows_out.append({
            "excel_row": idx + 2,
            "original_snippet": original[:300],
            "model_only_snippet": model_only[:300],
            "regex_then_model_snippet": regex_then_model[:300],
            "hn_missed_by_model": hn_still_present_model,
            "hn_missed_by_combined": hn_still_present_combined,
            "regex_matches": str([m["matched_text"] for m in regex_matches]),
        })

    result_df = pd.DataFrame(rows_out)

    # Summary
    missed_model = result_df["hn_missed_by_model"].sum()
    missed_combined = result_df["hn_missed_by_combined"].sum()
    total = len(result_df)

    print(f"\n{'='*60}")
    print(f"Results ({total} rows with HN patterns):")
    print(f"  Model only  - HN still present: {missed_model}/{total}")
    print(f"  Regex+Model - HN still present: {missed_combined}/{total}")
    print(f"  Improvement: {missed_model - missed_combined} additional HNs caught")
    print(f"{'='*60}")

    # Write comparison file
    result_df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
    print(f"\nComparison written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
