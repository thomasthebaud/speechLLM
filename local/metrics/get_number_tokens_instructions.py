import os
import math
import statistics
from typing import List

import torch
from transformers import AutoTokenizer

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        # keep non-empty lines (strip newline, preserve internal whitespace)
        lines = [ln.rstrip("\n") for ln in f]
    # drop completely empty / whitespace-only lines
    return [ln for ln in lines if ln.strip() != ""]

def main(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    file_path: str = "instructions.txt",
):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find: {file_path}")

    # Tokenizer only is enough for counting tokens
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    lines = load_lines(file_path)
    if not lines:
        raise ValueError(f"No non-empty lines found in {file_path}")

    token_counts: List[int] = []
    for i, text in enumerate(lines, start=1):
        # add_special_tokens=False counts only the text itself
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(ids))

    avg = statistics.mean(token_counts)
    # Use population std (pstdev). If you want sample std, use statistics.stdev.
    std = statistics.pstdev(token_counts) if len(token_counts) > 1 else 0.0
    mn = min(token_counts)
    mx = max(token_counts)

    print(f"Model: {model_id}")
    print(f"File:  {file_path}")
    print(f"Lines analyzed (non-empty): {len(token_counts)}")
    print(f"Tokens per line: avg={avg:.3f}, std={std:.3f}, min={mn}, max={mx}")

    # Optional: show a quick histogram-ish summary
    # (remove if you don't want extra output)
    print("\nFirst 5 lines token counts:", token_counts[:5])

if __name__ == "__main__":
    main()
