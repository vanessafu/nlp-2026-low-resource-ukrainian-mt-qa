#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Optional

from constants import (
    MARKERS as _MARKERS,
    SYSTEM_ZNO as _SYSTEM_ZNO,
    SYSTEM_RC as _SYSTEM_RC,
    PREFIXES as _PREFIXES,
    load_qa_records as _load_qa_records,
)


def _format_user(
    question: str,
    answers:  list[dict],
    prefix:   str = "",
    passage:  str = "",
) -> str:
    parts: list[str] = []
    if passage:
        parts.append(passage)
        parts.append("")
    parts.append(prefix + question)
    parts.append("")
    for a in answers:
        parts.append(f"{a['marker']}: {a['text']}")
    return "\n".join(parts)


def _random_permutations(n: int, k: int, rng: random.Random) -> list[list[int]]:
    """Return up to k distinct random permutations of range(n)."""
    total = math.factorial(n)
    if k >= total:
        perms = list(itertools.permutations(range(n)))
        rng.shuffle(perms)
        return [list(p) for p in perms]
    seen: set[tuple] = set()
    result: list[list[int]] = []
    attempts = 0
    while len(result) < k and attempts < k * 20:
        p = list(range(n))
        rng.shuffle(p)
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            result.append(p)
        attempts += 1
    return result


def _apply_permutation(
    answers:        list[dict],
    correct_marker: str,
    perm:           list[int],
) -> tuple[list[dict], str]:
    n         = len(answers)
    markers   = _MARKERS[:n]
    reordered = [answers[perm[i]] for i in range(n)]
    new_answers = [{"marker": markers[i], "text": reordered[i]["text"]} for i in range(n)]
    orig_idx  = next(j for j, a in enumerate(answers) if a["marker"] == correct_marker)
    new_idx   = perm.index(orig_idx)
    return new_answers, markers[new_idx]



def augment(
    items:       list[dict],
    factor:      int  = 4,
    seed:        int  = 42,
    subjects:    Optional[set[str]] = None,
) -> list[dict]:
    """
    Return augmented records in internal format:
      {"passage", "question", "answers", "correct_answers", "subject", "_prefix"}

    factor controls how many variants per question (including identity).
    """
    output: list[dict] = []

    for item in items:
        answers = item.get("answers") or []
        correct = item.get("correct_answers") or []
        subject = item.get("subject", "")
        passage = item.get("passage", "")

        if subjects and subject not in subjects:
            continue
        if len(answers) == 0 or len(correct) != 1:
            continue
        if correct[0] not in {a["marker"] for a in answers}:
            continue

        n   = len(answers)
        rng = random.Random(seed + hash(item["question"]) % (2 ** 31))

        # --- variant 0: identity (original order, no prefix) ---
        output.append({
            "passage":         passage,
            "question":        item["question"],
            "answers":         answers,
            "correct_answers": correct,
            "subject":         subject,
            "_prefix":         "",
        })

        if factor <= 1:
            continue

        # --- variants 1…(factor-1): shuffled + optionally perturbed ---
        identity = list(range(n))
        n_shuffles = factor - 1
        perms = _random_permutations(n, n_shuffles + 4, rng)
        non_identity = [p for p in perms if p != identity]
        shuffles = (non_identity + [identity])[:n_shuffles]

        for vi, perm in enumerate(shuffles):
            new_answers, new_correct = _apply_permutation(answers, correct[0], perm)
            prefix = "" if vi == 0 else _PREFIXES[1 + (vi - 1) % (len(_PREFIXES) - 1)]
            output.append({
                "passage":         passage,
                "question":        item["question"],
                "answers":         new_answers,
                "correct_answers": [new_correct],
                "subject":         subject,
                "_prefix":         prefix,
            })

    return output

def _to_chat(record: dict) -> dict:
    system = _SYSTEM_RC if record.get("passage") else _SYSTEM_ZNO
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": _format_user(
                record["question"],
                record["answers"],
                prefix=record.get("_prefix", ""),
                passage=record.get("passage", ""),
            )},
            {"role": "assistant", "content": record["correct_answers"][0]},
        ]
    }


def _to_json_item(record: dict) -> dict:
    item = {
        "question":        record.get("_prefix", "") + record["question"],
        "answers":         record["answers"],
        "correct_answers": record["correct_answers"],
        "subject":         record["subject"],
    }
    if record.get("passage"):
        item["passage"] = record["passage"]
    return item


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",   "-i", type=Path, required=True,
                   help="Input JSON file (train.json or belebele_qa.json)")
    p.add_argument("--output",  "-o", type=Path, required=True,
                   help="Output file path")
    p.add_argument("--factor",        type=int, default=4,
                   help="Oversample factor N: N variants per question (default: 4 → ~4×)")
    p.add_argument("--output-format", choices=["jsonl", "json"], default="jsonl",
                   help="jsonl = chat format for fine-tuning; json = flat list (default: jsonl)")
    p.add_argument("--subjects",      nargs="+", default=None,
                   help="Keep only these subjects (default: all)")
    p.add_argument("--seed",          type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    data = _load_qa_records(args.input)
    print(f"Loaded {len(data):,} questions from {args.input}")

    subjects = set(args.subjects) if args.subjects else None

    records = augment(data, factor=args.factor, seed=args.seed, subjects=subjects)

    n_original = len(data) if not subjects else sum(
        1 for d in data
        if len(d.get("correct_answers") or []) == 1
        and d.get("subject", "") in (subjects or {d.get("subject", "")})
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "jsonl":
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(_to_chat(rec), ensure_ascii=False) + "\n")
    else:
        items = [_to_json_item(r) for r in records]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    from collections import Counter
    subj_counts = Counter(r["subject"] for r in records)
    print(f"\nAugmentation summary:")
    print(f"  Input questions : {len(data):,}")
    print(f"  Output variants : {len(records):,}")
    print(f"  Effective factor: {len(records) / max(1, n_original):.2f}×")
    print(f"\nSubject breakdown:")
    for s, n in subj_counts.most_common():
        print(f"  {n:6,}  {s}")
    print(f"\nWritten {len(records):,} records → {args.output}  [{args.output_format}]")


if __name__ == "__main__":
    main()
