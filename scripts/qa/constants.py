"""Shared constants for all QA scripts."""

import json
import re
import unicodedata
from pathlib import Path

MARKERS   = ["А", "Б", "В", "Г", "Д"]
RC_HEADER = "**Прочитайте текст і виконайте завдання**"


def normalize(text: str) -> str:
    """Lowercase, NFC-normalize, and collapse to a plain word string for cross-source
    question matching (DUMY vs. local ZNO, CoT vs. augmented, RC dedup, ...)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\wа-яіїєґa-z0-9]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def load_qa_records(path) -> list[dict]:
    """
    Load ZNO/MMLU-style QA records and normalize to the internal shape used
    across scripts/qa/*: {"question", "answers": [{"marker","text"}],
    "correct_answers": [marker], "subject"}.

    Accepts either the old competition format (JSON array, options already
    lettered) or the 2026 official format (JSONL, one record per line, with
    "possible_answers": {"0": ..., "1": ...} and "correct_answer_num": int)
    — the only difference between the two is letters vs. numbered options.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

    out: list[dict] = []
    for r in records:
        if "possible_answers" in r:
            opts    = r["possible_answers"]
            n       = len(opts)
            markers = MARKERS[:n]
            out.append({
                "question":        r["question"],
                "answers":         [{"marker": markers[i], "text": opts[str(i)]} for i in range(n)],
                "correct_answers": [markers[r["correct_answer_num"]]],
                "subject":         r.get("subject", ""),
            })
        else:
            out.append(r)
    return out

PREFIXES = [
    "",
    "Оберіть правильну відповідь:\n",
    "Визначте правильний варіант:\n",
    "Укажіть правильну відповідь:\n",
    "Виберіть одну правильну відповідь:\n",
    "Яка відповідь є правильною?\n",
    "Знайдіть правильне твердження:\n",
]

SYSTEM_ZNO = (
    "Ти — експерт із підготовки до українських іспитів (ЗНО/НМТ). "
    "Тобі буде надано питання з кількома варіантами відповіді. "
    "Уважно прочитай питання та варіанти, потім відповідай лише літерою "
    "правильної відповіді (наприклад: А, Б, В або Г)."
)

SYSTEM_RC = (
    "Ти — експерт із читання та аналізу текстів. "
    "Тобі буде надано текст-уривок і питання з кількома варіантами відповіді. "
    "Уважно прочитай уривок, потім вибери правильну відповідь і вкажи лише її "
    "літеру (А, Б, В або Г)."
)

SYSTEM_MATH = (
    "Ти — викладач математики та підготовки до ЗНО/НМТ. "
    "Розв'яжи задачу крок за кроком у блоці <think>…</think>, "
    "потім дай коротку остаточну відповідь."
)
