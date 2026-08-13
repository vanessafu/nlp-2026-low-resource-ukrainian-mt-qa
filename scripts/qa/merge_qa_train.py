#!/usr/bin/env python3
"""
Merge the chat-format QA training files into train_data/QA/train_merged.jsonl.

Sources
-------
  ukr_zno_qa_train.jsonl        — built by scripts/qa/build_cot.py
  ukr_mmlu_qa_train_chat.jsonl  — MMLU, chat-format (produced separately, not
                                   part of this repo yet — skipped if missing)

Run:
  python scripts/qa/merge_qa_train.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
QA_DIR = ROOT / "train_data" / "QA"
OUT    = QA_DIR / "train_merged.jsonl"
SEED   = 42

SOURCES = [
    QA_DIR / "ukr_zno_qa_train.jsonl",
    QA_DIR / "ukr_mmlu_qa_train_chat.jsonl",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [skip] {path.name} (not found)")
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"  {path.name}: {len(rows):,} items")
    return rows


def main() -> None:
    rows: list[dict] = []
    for path in SOURCES:
        rows.extend(read_jsonl(path))

    random.Random(SEED).shuffle(rows)
    print(f"  total: {len(rows):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  saved → {OUT}")


if __name__ == "__main__":
    main()
