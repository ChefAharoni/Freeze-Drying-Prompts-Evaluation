#!/usr/bin/env python3
"""Download and normalize prompt-injection datasets into JSONL files."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_split(dataset_dict) -> Tuple[str, Dataset]:
    """Pick a deterministic split preferring test > validation > train > first."""
    for name in ("test", "validation", "dev", "train"):
        if name in dataset_dict:
            return name, dataset_dict[name]
    # fallback to first available
    name = sorted(dataset_dict.keys())[0]
    return name, dataset_dict[name]


def detect_field(example: Dict, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in example:
            return c
    return None


def normalize_label(raw) -> str:
    if isinstance(raw, bool):
        return "INJECTION" if raw else "LEGIT"
    if isinstance(raw, (int, float)):
        return "INJECTION" if raw == 1 else "LEGIT"
    if raw is None:
        return "UNKNOWN"
    text = str(raw).strip().upper()
    if text in {"PROMPT_INJECTION", "INJECTION", "ATTACK", "JAILBREAK"}:
        return "INJECTION"
    if text in {"LEGIT", "BENIGN", "NORMAL", "SAFE"}:
        return "LEGIT"
    return text or "UNKNOWN"


def select_columns(example: Dict) -> Tuple[str, str]:
    text_field = detect_field(example, ["text", "prompt", "input", "content", "body"])
    label_field = detect_field(example, ["label", "class", "category", "type", "annotation", "tag"])
    if text_field is None:
        raise KeyError("Could not find text field in example")
    if label_field is None:
        raise KeyError("Could not find label field in example")
    return text_field, label_field


def export_jsonl(dataset: Dataset, dataset_key: str, split_name: str, limit: int | None, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    first_example = dataset[0]
    text_field, label_field = select_columns(first_example)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(tqdm(dataset, desc=f"writing {dataset_key}")):
            record = {
                "id": row.get("id", row.get("_id", idx)),
                "dataset": dataset_key,
                "split": split_name,
                "text": row[text_field],
                "label": normalize_label(row[label_field]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_deepset():
    ds_dict = load_dataset("deepset/prompt-injections")
    # concatenate all splits to hit the full 662 examples cited in the paper
    all_splits = []
    for split in ds_dict.values():
        all_splits.append(split)
    combined = concatenate_datasets(all_splits)
    out_path = os.path.join(DATA_DIR, "deepset.jsonl")
    export_jsonl(combined, "deepset", "all", None, out_path)


def process_combined(limit: int = 10000):
    ds_dict = load_dataset("geekyrakshit/prompt-injection-dataset")
    split_name, split = pick_split(ds_dict)

    # deterministic selection up to limit
    if len(split) > limit:
        split = split.shuffle(seed=42).select(range(limit))

    out_path = os.path.join(DATA_DIR, "combined.jsonl")
    export_jsonl(split, "combined", split_name, limit, out_path)


def main():
    ensure_dir(DATA_DIR)
    process_deepset()
    process_combined(limit=10000)
    print("Done. Files written to data/deepset.jsonl and data/combined.jsonl")


if __name__ == "__main__":
    main()
