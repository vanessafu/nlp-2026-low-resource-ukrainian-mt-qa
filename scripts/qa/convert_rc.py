#!/usr/bin/env python3
"""
Convert reading-comprehension sources to Ukrainian QA train.json format.

Sources
-------
  facebook/belebele  ukr_Cyrl  (HuggingFace, downloaded at runtime)
  QUA-RC/quarc.json            (local)

Belebele output modes (--belebele-format):
  embedded    Passage embedded in question field (ZNO RC style matching train.json).
              Output: {"question": "**Прочитайте...**\\n\\n{passage}\\n\\n{stem}", ...}
              Writes: data/training/QA/belebele_rc.json

  classified  Passage kept as separate "passage" field + subject classification
              via keyword matching (history / language-lit / reading-comprehension).
              Output: {"passage": "...", "question": "...", "subject": "<class>", ...}
              Writes: data/processed/belebele_qa.json

Usage
-----
  python scripts/qa/convert_rc.py                        # belebele(embedded) + quarc
  python scripts/qa/convert_rc.py --no-belebele          # quarc only
  python scripts/qa/convert_rc.py --belebele-format classified
  python scripts/qa/convert_rc.py --no-quarc             # belebele only
  python scripts/qa/convert_rc.py --subjects history-of-ukraine ukrainian-language-and-literature
  python scripts/qa/convert_rc.py --keep-all             # skip subject filter (classified only)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from constants import MARKERS, RC_HEADER

_REPO = Path(__file__).resolve().parent.parent.parent


def _embed(passage: str, stem: str) -> str:
    return f"{RC_HEADER}\n\n{passage}\n\n{stem}"


_HISTORY_RE = re.compile(
    r"(ві(йн|йськ)|революці|держав|імпер|республік|монарх|коро(ль|лів)|цар|гетьман|"
    r"парламент|конституці|незалежн|окупац|колоні|анексі|суверен|"
    r"(XVIII|XIX|XX|XXI)\s*стол|1[6-9]\d{2}|20\d{2}|"
    r"битв|похід|збройн|армі|флот|капітуляц|завоюванн|окупован|"
    r"президент|прем'єр|лідер|уряд|пол[іи]тич|дипломат|договір|альянс|"
    r"нація|народ|етн|плем|цивілізац|античн|середньовіч|феодал|рабовласн|"
    r"археологічн|музей|пам'ятк|спадщин)",
    re.IGNORECASE,
)

_LITERATURE_RE = re.compile(
    r"(поет|письменник|роман|повість|вірш|поема|оповіданн|драм|п'єс|байк|казк|"
    r"літератур|мистецтв|театр|кіно|живопис|скульптур|архітектур|"
    r"граматик|мовозна|синтаксис|морфологі|лексик|фонетик|орфографі|"
    r"слово|речення|наголос|відмінк|дієслово|прикметник|іменник|займенник|"
    r"автор|твор|жанр|стиль|образ|персонаж|герой|сюжет|тем[аи])",
    re.IGNORECASE,
)


def _classify(passage: str, question: str) -> str:
    text = passage + " " + question
    n_h  = len(_HISTORY_RE.findall(text))
    n_l  = len(_LITERATURE_RE.findall(text))
    if n_h == 0 and n_l == 0:
        return "reading-comprehension"
    return "history-of-ukraine" if n_h >= n_l else "ukrainian-language-and-literature"


_NUM_TO_MARKER = {str(i + 1): m for i, m in enumerate(MARKERS)}


def from_belebele_embedded() -> list[dict]:
    """Passage embedded in question field — matches ZNO train.json format."""
    from datasets import load_dataset  # type: ignore
    print("  Downloading facebook/belebele ukr_Cyrl …")
    rows = load_dataset("facebook/belebele", "ukr_Cyrl", trust_remote_code=True)["test"]
    out = []
    for row in rows:
        passage = row["flores_passage"].strip()
        stem    = row["question"].strip()
        correct = str(row["correct_answer_num"]).strip()
        if correct not in _NUM_TO_MARKER:
            continue
        answers = [{"marker": MARKERS[i], "text": row[f"mc_answer{i+1}"].strip()}
                   for i in range(4)]
        if any(not a["text"] for a in answers):
            continue
        out.append({
            "question":        _embed(passage, stem),
            "answers":         answers,
            "correct_answers": [_NUM_TO_MARKER[correct]],
            "subject":         "reading-comprehension",
        })
    return out


def from_belebele_classified(
    subjects: list[str] | None = None,
    keep_all: bool = False,
    cache_dir: str | None = None,
) -> list[dict]:
    """Passage separate field + keyword-based subject classification."""
    from datasets import load_dataset  # type: ignore
    print("  Downloading facebook/belebele ukr_Cyrl …", flush=True)
    kwargs = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("facebook/belebele", "ukr_Cyrl", **kwargs)
    split = ds["test"]

    converted, skipped = [], 0
    for row in split:
        passage  = (row.get("flores_passage") or "").strip()
        question = (row.get("question")       or "").strip()
        correct  = str(row.get("correct_answer_num", "")).strip()
        if not passage or not question or correct not in _NUM_TO_MARKER:
            skipped += 1
            continue
        answers = []
        ok = True
        for i, marker in enumerate(MARKERS[:4], 1):
            text = (row.get(f"mc_answer{i}") or "").strip()
            if not text:
                ok = False
                break
            answers.append({"marker": marker, "text": text})
        if not ok:
            skipped += 1
            continue
        converted.append({
            "passage":         passage,
            "question":        question,
            "answers":         answers,
            "correct_answers": [_NUM_TO_MARKER[correct]],
            "subject":         _classify(passage, question),
        })

    if skipped:
        print(f"  Skipped {skipped} malformed items")

    subj = Counter(it["subject"] for it in converted)
    print(f"  Subject distribution: {dict(subj.most_common())}")

    if not keep_all and subjects:
        keep_set = set(subjects)
        before    = len(converted)
        converted = [it for it in converted if it["subject"] in keep_set]
        print(f"  After subject filter: {len(converted):,} / {before:,}")

    return converted


def from_quarc(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out  = []
    for item in data:
        text = item["text"].strip()
        for mcq in item["mcqs"]:
            choices = [c.strip() for c in mcq["choices"] if c and c.strip()]
            key     = (mcq["key"] or "").strip()
            if not key or not (2 <= len(choices) <= 5) or key not in choices:
                continue
            answers = [{"marker": MARKERS[i], "text": c}
                       for i, c in enumerate(choices)]
            out.append({
                "question":        _embed(text, mcq["stem"].strip()),
                "answers":         answers,
                "correct_answers": [MARKERS[choices.index(key)]],
                "subject":         "ukrainian-language-and-literature",
            })
    return out


def _save(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    subj = Counter(it["subject"] for it in items)
    print(f"  {len(items):,} items → {path}")
    for s, n in subj.most_common():
        print(f"    {n:4d}  {s}")



def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--no-belebele",      action="store_true",
                   help="Skip HuggingFace belebele download")
    p.add_argument("--no-quarc",         action="store_true",
                   help="Skip QUA-RC conversion")
    p.add_argument("--belebele-format",  choices=["embedded", "classified"],
                   default="embedded",
                   help=(
                       "embedded: passage embedded in question field (default, ZNO RC style). "
                       "classified: separate passage field + keyword subject classification."
                   ))
    p.add_argument("--keep-all",         action="store_true",
                   help="Skip subject filter (classified mode only)")
    p.add_argument("--subjects",         nargs="+",
                   default=["history-of-ukraine",
                            "ukrainian-language-and-literature",
                            "reading-comprehension"],
                   help="Subjects to keep in classified mode (default: all three)")
    p.add_argument("--cache-dir",        default=None,
                   help="HuggingFace dataset cache directory")
    p.add_argument("--out-dir",          type=Path,
                   default=_REPO / "data" / "training" / "QA")
    args = p.parse_args(argv)

    if not args.no_belebele:
        print("belebele:")
        if args.belebele_format == "embedded":
            items = from_belebele_embedded()
            _save(items, args.out_dir / "belebele_rc.json")
        else:
            items = from_belebele_classified(
                subjects=args.subjects,
                keep_all=args.keep_all,
                cache_dir=args.cache_dir,
            )
            out = _REPO / "data" / "processed" / "belebele_qa.json"
            _save(items, out)

    if not args.no_quarc:
        print("QUA-RC:")
        _save(from_quarc(_REPO / "QUA-RC" / "quarc.json"),
              args.out_dir / "quarc_rc.json")


if __name__ == "__main__":
    main()
