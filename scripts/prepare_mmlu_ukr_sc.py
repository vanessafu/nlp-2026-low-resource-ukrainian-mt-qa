#!/usr/bin/env python3
"""Build Ukrainian SC training data from mmlu_ukr.

~50% sentences have one injected spelling error; ~50% are CORRECT (no error),
matching the dev SC label distribution.
"""

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
MMLU_UKR_VAL = ROOT / "train_data" / "QA" / "mmlu_ukr" / "data" / "validation-00000-of-00001.parquet"
OUT_JSONL = ROOT / "train_data" / "SC" / "mmlu_ukr_sc_train.jsonl"
DATASET_ID = "wmt2026_lrllm_train_sc_ukr"

WORD_RE = re.compile(
    r"[A-Za-zА-Яа-яІіЇїЄєҐґ']{4,}",
    flags=re.UNICODE,
)

UKR_LOWER = set("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'")
SIMILAR = {
    "и": "і",
    "і": "и",
    "е": "є",
    "є": "е",
    "о": "а",
    "а": "о",
    "у": "ю",
    "ю": "у",
    "т": "ь",
    "ь": "т",
    "ш": "щ",
    "щ": "ш",
    "п": "р",
    "р": "п",
    "н": "м",
    "м": "н",
}
LATIN_CONFUSABLES = {
    "c": "с",
    "C": "С",
    "p": "р",
    "P": "Р",
    "a": "а",
    "A": "А",
    "e": "е",
    "E": "Е",
    "o": "о",
    "O": "О",
    "i": "і",
    "I": "І",
    "y": "у",
    "Y": "У",
    "x": "х",
    "X": "Х",
}


def ukr_letter_count(word: str) -> int:
    return sum(1 for ch in word.lower() if ch in UKR_LOWER)


def find_candidates(sentence: str) -> list[tuple[int, int, str]]:
    out = []
    for m in WORD_RE.finditer(sentence):
        word = m.group(0)
        if ukr_letter_count(word) >= 3:
            out.append((m.start(), m.end(), word))
    return out


def swap_adjacent(word: str, rng: random.Random) -> str | None:
    if len(word) < 2:
        return None
    i = rng.randrange(len(word) - 1)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    typo = "".join(chars)
    return typo if typo != word else None


def delete_char(word: str, rng: random.Random) -> str | None:
    if len(word) < 5:
        return None
    i = rng.randrange(len(word))
    typo = word[:i] + word[i + 1 :]
    return typo if typo != word else None


def duplicate_char(word: str, rng: random.Random) -> str | None:
    i = rng.randrange(len(word))
    typo = word[:i] + word[i] + word[i:]
    return typo if typo != word else None


def substitute_char(word: str, rng: random.Random) -> str | None:
    indices = [i for i, ch in enumerate(word) if ch.lower() in SIMILAR or ch in LATIN_CONFUSABLES]
    if not indices:
        return None
    i = rng.choice(indices)
    ch = word[i]
    repl = LATIN_CONFUSABLES.get(ch) or SIMILAR.get(ch.lower())
    if not repl:
        return None
    if ch.isupper() and len(repl) == 1:
        repl = repl.upper()
    typo = word[:i] + repl + word[i + 1 :]
    return typo if typo != word else None


def latin_homoglyph(word: str, rng: random.Random) -> str | None:
    indices = [i for i, ch in enumerate(word) if ch in LATIN_CONFUSABLES]
    if not indices:
        return None
    i = rng.choice(indices)
    typo = word[:i] + LATIN_CONFUSABLES[word[i]] + word[i + 1 :]
    return typo if typo != word else None


def make_typo(word: str, rng: random.Random) -> str | None:
    strategies = [swap_adjacent, delete_char, duplicate_char, substitute_char, latin_homoglyph]
    rng.shuffle(strategies)
    for fn in strategies:
        typo = fn(word, rng)
        if typo and typo != word:
            return typo
    return None


def inject_typo(sentence: str, rng: random.Random) -> tuple[str, str, str] | None:
    candidates = find_candidates(sentence)
    if not candidates:
        return None
    rng.shuffle(candidates)
    for start, end, word in candidates:
        typo_word = make_typo(word, rng)
        if not typo_word:
            continue
        input_sentence = sentence[:start] + typo_word + sentence[end:]
        if input_sentence.count(typo_word) != 1:
            continue
        return input_sentence, typo_word, word
    return None


def iter_sentences(records):
    for idx, rec in enumerate(records):
        q = rec["question"].strip()
        if q:
            yield idx, "question", q, rec.get("subject") or ""
        for j, choice in enumerate(rec["choices"]):
            c = str(choice).strip()
            if len(c) >= 12 and ukr_letter_count(c) >= 4:
                yield idx, f"choice_{j}", c, rec.get("subject") or ""


def build_rows(records, seed: int, correct_ratio: float = 0.5):
    rng = random.Random(seed)
    sentences = list(iter_sentences(records))
    rows = []
    n_correct = 0
    n_error = 0

    for sid, (mmlu_idx, field, sentence, subject) in enumerate(sentences):
        if rng.random() < correct_ratio:
            input_sentence = sentence
            incorrect_word = "CORRECT"
            correct_word = "CORRECT"
            n_correct += 1
        else:
            injected = inject_typo(sentence, rng)
            if not injected:
                input_sentence = sentence
                incorrect_word = "CORRECT"
                correct_word = "CORRECT"
                n_correct += 1
            else:
                input_sentence, incorrect_word, correct_word = injected
                n_error += 1

        rows.append(
            {
                "dataset_id": DATASET_ID,
                "id": f"{sid:05d}",
                "mmlu_idx": mmlu_idx,
                "field": field,
                "subject": subject,
                "input_sentence": input_sentence,
                "original_sentence": sentence,
                "incorrect_word": incorrect_word,
                "correct_word": correct_word,
            }
        )

    return rows, n_correct, n_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=MMLU_UKR_VAL)
    parser.add_argument("--output", type=Path, default=OUT_JSONL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--correct-ratio",
        type=float,
        default=0.5,
        help="Fraction of sentences labeled CORRECT (no spelling error)",
    )
    args = parser.parse_args()

    table = pq.read_table(args.input)
    records = table.to_pydict()
    n_items = len(records["question"])
    records = [
        {
            "question": records["question"][i],
            "choices": records["choices"][i],
            "answer": records["answer"][i],
            "subject": records["subject"][i],
        }
        for i in range(n_items)
    ]

    rows, n_correct, n_error = build_rows(records, args.seed, args.correct_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"MMLU items: {n_items}")
    print(f"SC sentences: {len(rows)}")
    print(f"  CORRECT: {n_correct} ({100 * n_correct / len(rows):.1f}%)")
    print(f"  with error: {n_error} ({100 * n_error / len(rows):.1f}%)")
    print(f"Wrote {args.output}")
    if rows:
        ex = rows[0]
        print("Example:")
        print(f"  incorrect: {ex['incorrect_word']} -> {ex['correct_word']}")
        print(f"  input: {ex['input_sentence'][:120]}...")


if __name__ == "__main__":
    main()
