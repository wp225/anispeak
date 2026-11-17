#!/usr/bin/env python3
# extract_keys.py
#
# RUN INSTRUCTIONS:
# 1. Save this file as `extract_keys.py`.
# 2. Run it from your terminal using the command:
#    python extract_keys.py
#
# Description:
# Reads the species_cache.json file, extracts all keys, applies a basic
# normalization to deduplicate minor variants (case/punctuation), and
# writes the shortened, unique list to a text file.

import json
import os
import re

def normalize_key(k: str) -> str:
    """Lowercases and removes non-alphanumeric characters for grouping."""
    assert isinstance(k, str), "Input key must be a string."
    return re.sub(r'[^a-z0-9]', '', k.lower().strip())

def extract_and_dedupe(input_json: str, output_txt: str) -> None:
    # Ensure the input file exists
    assert os.path.exists(input_json), f"Input file not found at: {input_json}"

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure the JSON structure is as expected (a dictionary of cache entries)
    assert isinstance(data, dict), "The root of the JSON file must be a dictionary."

    unique_normalized = set()
    deduped_keys = []

    for key in data.keys():
        assert isinstance(key, str), "All keys in the JSON must be strings."

        norm = normalize_key(key)
        # Only keep the first variant we see for this normalized string
        if norm not in unique_normalized:
            unique_normalized.add(norm)
            deduped_keys.append(key)

    # Sanity checks on the output data
    assert len(deduped_keys) > 0, "No keys were extracted."
    assert len(deduped_keys) <= len(data.keys()), "Deduped list cannot be larger than the original list."

    # Write the deduped keys to the output text file
    with open(output_txt, 'w', encoding='utf-8') as f:
        for key in sorted(deduped_keys):
            f.write(f"{key}\n")

    print(f"[OK] Extracted {len(deduped_keys)} unique keys (down from {len(data)}) to {output_txt}")

if __name__ == "__main__":
    INPUT_FILE = "data/species_cache.json"
    OUTPUT_FILE = "unique_species_keys.txt"

    extract_and_dedupe(INPUT_FILE, OUTPUT_FILE)
