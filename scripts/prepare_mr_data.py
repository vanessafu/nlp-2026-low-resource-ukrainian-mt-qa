#!/usr/bin/env python3
"""Download GSM8K and competition_math (MATH) training data for MR."""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "train_data" / "MR"
GSM8K_JSONL = OUT_DIR / "gsm8k_train.jsonl"
COMP_MATH_JSONL = OUT_DIR / "competition_math_train.jsonl"
COMP_MATH_RAW = OUT_DIR / "competition_math"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_gsm8k_final_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", solution)
    if matches:
        return matches[-1].strip()
    return solution.strip()


def export_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for i, row in enumerate(ds):
        rows.append(
            {
                "dataset_id": "gsm8k_train",
                "id": f"gsm8k-{i:05d}",
                "difficulty": "low",
                "source": "gsm8k",
                "question_en": row["question"],
                "answer_en": row["answer"],
                "final_answer": extract_gsm8k_final_answer(row["answer"]),
            }
        )
    write_jsonl(GSM8K_JSONL, rows)
    return len(rows)


def export_competition_math():
    rows = []
    idx = 0
    for train_path in sorted(COMP_MATH_RAW.rglob("train*.parquet")):
        if train_path.parent.parent == COMP_MATH_RAW and train_path.parent.name == "data":
            continue
        table = pq.read_table(train_path)
        for row in table.to_pylist():
            rows.append(
                {
                    "dataset_id": "competition_math_train",
                    "id": f"comp-math-{idx:05d}",
                    "difficulty": "medium",
                    "source": "competition_math",
                    "subject": row["type"],
                    "level": row["level"],
                    "question_en": row["problem"],
                    "answer_en": row["solution"],
                    "final_answer": extract_boxed_answer(row["solution"]),
                }
            )
            idx += 1
    write_jsonl(COMP_MATH_JSONL, rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gsm8k", action="store_true")
    parser.add_argument("--skip-comp-math", action="store_true")
    args = parser.parse_args()

    if not args.skip_gsm8k:
        n_gsm8k = export_gsm8k()
        print(f"Wrote {n_gsm8k} rows to {GSM8K_JSONL}")

    if not args.skip_comp_math:
        if not COMP_MATH_RAW.exists():
            raise FileNotFoundError(
                f"Missing {COMP_MATH_RAW}. Run: bash scripts/download_competition_math.sh"
            )
        n_comp = export_competition_math()
        print(f"Wrote {n_comp} rows to {COMP_MATH_JSONL}")


if __name__ == "__main__":
    main()
