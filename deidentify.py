"""
De-identification pipeline for Thai medical progress notes.
Uses loolootech/no-name-ner-th NER model to detect and mask PHI.
"""

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
    "DATE": "[DATE]",
    "NATIONAL_ID": "[NATIONAL_ID]",
    "HOSPITAL_IDS": "[HOSPITAL_IDS]",
}

INPUT_FILE = "PTPHYSICALEXAM 10-122025.xlsx"
OUTPUT_FILE = "PTPHYSICALEXAM_deidentified.xlsx"
COLUMN = "progressnote"
START_ROW = 10   # 1-based data row (row 10 in the sheet = index 9 in 0-based)
END_ROW = 100    # inclusive


# Matches: HN 35447/60, HN 8436260, HN155568, HN:123456, HN. 1234/56
HN_PATTERN = re.compile(r'(?i)\bHN\s*[\.:;]?\s*(\d{4,8}(?:/\d{2,4})?)')

# Matches: Thai national ID — 13 consecutive digits, not preceded/followed by digits
# e.g. 3160101285616, 1100704366722
NATIONAL_ID_PATTERN = re.compile(r'(?<!\d)\d{13}(?!\d)')


def regex_preprocess(text: str) -> str:
    """Apply regex-based PHI replacement before NER model.

    Catches patterns the model frequently misses:
    - HN numbers (~47% miss rate by model alone)
    - Thai 13-digit national ID numbers
    """
    if not text or not isinstance(text, str):
        return text
    text = HN_PATTERN.sub(r'HN [HOSPITAL_IDS]', text)
    text = NATIONAL_ID_PATTERN.sub('[NATIONAL_ID]', text)
    return text


def load_ner_pipeline():
    """Load the NER model. Downloads on first run."""
    print(f"Loading NER model: {MODEL_ID} ...")
    ner = pipeline(
        "token-classification",
        model=MODEL_ID,
        device=-1,  # CPU
        aggregation_strategy="simple",
    )
    print("Model loaded.")
    return ner


def merge_entities(entities):
    """
    Merge adjacent entities of the same type that overlap or touch.
    Returns a list of (start, end, entity_group, score) tuples sorted by start.
    """
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
            # extend the previous span
            prev_start, prev_end, prev_label, prev_score = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_label, max(prev_score, score))
        else:
            merged.append((start, end, label, score))

    return merged


def anonymize_text(ner_pipeline, text):
    """
    Run NER on text and replace detected entities with placeholder tokens.
    Returns (anonymized_text, list_of_detected_entities).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return text, []

    # Pre-process: catch HN and national ID patterns the model frequently misses
    text = regex_preprocess(text)

    results = ner_pipeline(text)
    merged = merge_entities(results)

    detected = []
    anonymized = text

    # Replace in reverse order to preserve character positions
    for start, end, label, score in reversed(merged):
        original_span = text[start:end]
        token = ENTITY_MAP.get(label, f"[{label}]")
        anonymized = anonymized[:start] + token + anonymized[end:]
        detected.append({
            "entity_type": label,
            "original": original_span,
            "start": start,
            "end": end,
            "score": round(score, 4),
            "replacement": token,
        })

    detected.reverse()
    return anonymized, detected


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found in {Path.cwd()}")
        sys.exit(1)

    # Load data
    print(f"Reading {INPUT_FILE} ...")
    df = pd.read_excel(input_path, engine="openpyxl")
    print(f"Total rows: {len(df)}, Columns: {len(df.columns)}")

    if COLUMN not in df.columns:
        print(f"Error: column '{COLUMN}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    # Select row range (convert to 0-based index)
    start_idx = START_ROW - 1
    end_idx = END_ROW  # pandas slice is exclusive on end
    subset = df.iloc[start_idx:end_idx].copy()
    non_null_mask = subset[COLUMN].notna() & (subset[COLUMN].astype(str).str.strip() != "")
    texts_to_process = subset.loc[non_null_mask, COLUMN]

    print(f"Rows {START_ROW}-{END_ROW}: {len(subset)} total, {len(texts_to_process)} with non-null {COLUMN}")

    if texts_to_process.empty:
        print("No text to process. Exiting.")
        sys.exit(0)

    # Load model
    ner = load_ner_pipeline()

    # Process
    all_detections = []
    deidentified_col = subset[COLUMN].copy()

    for idx in tqdm(texts_to_process.index, desc="De-identifying"):
        text = str(subset.at[idx, COLUMN])
        anonymized, detected = anonymize_text(ner, text)
        deidentified_col.at[idx] = anonymized

        for det in detected:
            all_detections.append({
                "row_index": idx,
                "row_number": idx + 2,  # Excel row (1-based + header)
                **det,
            })

    # Write results
    subset[COLUMN] = deidentified_col
    output_path = Path(OUTPUT_FILE)
    subset.to_excel(output_path, index=False, engine="openpyxl")
    print(f"\nDe-identified output written to: {output_path}")

    # Summary
    entity_counts = {}
    for det in all_detections:
        t = det["entity_type"]
        entity_counts[t] = entity_counts.get(t, 0) + 1

    print(f"\nSummary: {len(all_detections)} entities detected across {len(texts_to_process)} notes")
    for etype, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    # Save detection log (without original text to avoid PHI leakage in logs)
    if all_detections:
        log_df = pd.DataFrame(all_detections)
        log_path = Path("detections_log.csv")
        log_df.drop(columns=["original"], errors="ignore").to_csv(log_path, index=False)
        print(f"Detection log (no PHI): {log_path}")


if __name__ == "__main__":
    main()
