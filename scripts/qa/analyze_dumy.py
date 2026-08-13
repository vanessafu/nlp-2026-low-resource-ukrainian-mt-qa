"""
Overlap analysis between NLPForUA/dumy-zno-ukrainian-math-history-geo-r1-o1
and the local 2026-official ZNO QA splits (train_data/QA/ukr_qa_*.jsonl).

For each overlapping question we check:
  - whether R1 / O1 answers match the local ground-truth
  - r1_score / o1_score already stored in DUMY

Run:
    python scripts/qa/analyze_dumy.py [--cache-dir <path>]
"""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from constants import load_qa_records, normalize

ROOT         = Path(__file__).resolve().parent.parent.parent
LOCAL_QA_DIR = ROOT / "train_data/QA"

SUBJECT_MAP = {
    "ukrainian": "ukrainian-language-and-literature",
    "history":   "history-of-ukraine",
}


def safe_parse(value):
    if value is None:
        return []
    if isinstance(value, (list, dict)):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def answer_letter(answer_options, correct_answer_list):
    if not correct_answer_list or not answer_options:
        return set()
    correct_texts = {normalize(a) for a in correct_answer_list}
    letters: set[str] = set()
    for opt in answer_options:
        if not isinstance(opt, dict):
            continue
        label = opt.get("answer") or opt.get("label") or opt.get("marker") or ""
        text  = opt.get("text") or opt.get("value") or ""
        if normalize(text) in correct_texts or normalize(label) in correct_texts:
            letters.add(label.strip().upper())
    if not letters:
        for a in correct_answer_list:
            a = a.strip().upper()
            if len(a) <= 2:
                letters.add(a)
    return letters


def predicted_letter(r1_answer: str, answer_options) -> str | None:
    if not r1_answer:
        return None
    m = re.search(r"\b([АБВГҐДЕЖЗИІКЛМНО])\b", r1_answer.strip().upper())
    if m:
        return m.group(1)
    if answer_options:
        r1_norm = normalize(r1_answer)
        for opt in answer_options:
            if not isinstance(opt, dict):
                continue
            text = opt.get("text") or opt.get("value") or ""
            if normalize(text) and normalize(text) in r1_norm:
                label = opt.get("answer") or opt.get("label") or opt.get("marker") or ""
                return label.strip().upper()
    return None


def main(cache_dir: str | None = None):
    print("Loading DUMY dataset …")
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("NLPForUA/dumy-zno-ukrainian-math-history-geo-r1-o1", **kwargs)

    dumy_items = []
    for split in ds:
        for item in ds[split]:
            subj = (item.get("subject") or "").lower().strip()
            if subj in SUBJECT_MAP:
                dumy_items.append(item)

    print(f"DUMY items in relevant subjects: {len(dumy_items)}")
    dumy_by_norm = {normalize(it["question"]): it for it in dumy_items}

    local_items = []
    for split_name in ["train", "dev"]:
        items = load_qa_records(LOCAL_QA_DIR / f"ukr_qa_{split_name}.jsonl")
        for it in items:
            it["_split"] = split_name
        local_items.extend(items)

    print(f"Local QA items total: {len(local_items)}")
    local_by_norm = {normalize(it["question"]): it for it in local_items}

    overlap_keys = set(dumy_by_norm) & set(local_by_norm)
    print(f"\nExact-normalized overlap: {len(overlap_keys)} questions")

    results = []
    for key in overlap_keys:
        d  = dumy_by_norm[key]
        lo = local_by_norm[key]
        gt_letters   = set(lo.get("correct_answers") or [])
        dumy_gt      = answer_letter(safe_parse(d.get("answer_options")), safe_parse(d.get("correct_answer")))
        opts_parsed  = safe_parse(d.get("answer_options"))
        r1_letter    = predicted_letter((d.get("r1_answer") or "").strip(), opts_parsed)
        o1_letter    = predicted_letter((d.get("o1_answer") or "").strip(), opts_parsed)
        r1_correct   = bool(r1_letter and gt_letters and r1_letter in gt_letters)
        o1_correct   = bool(o1_letter and gt_letters and o1_letter in gt_letters)
        results.append({
            "question_preview": d["question"][:80],
            "subject_local":    lo.get("subject"),
            "subject_dumy":     d.get("subject"),
            "split":            lo["_split"],
            "gt_local":         sorted(gt_letters),
            "gt_dumy":          sorted(dumy_gt),
            "gt_agree":         bool(gt_letters and dumy_gt and gt_letters == dumy_gt),
            "r1_letter":        r1_letter,
            "r1_correct":       r1_correct,
            "r1_score_stored":  d.get("r1_score"),
            "o1_letter":        o1_letter,
            "o1_correct":       o1_correct,
            "o1_score_stored":  d.get("o1_score"),
            "bad_reasoning":    d.get("bad_reasoning", False),
            "has_r1_reasoning": bool(d.get("r1_reasoning")),
        })

    n = len(results)
    if n == 0:
        print("No overlapping questions found.")
        return

    def pct(num, den): return f"{num/den*100:.1f}%" if den else "N/A"

    stored_r1 = [r["r1_score_stored"] for r in results if r["r1_score_stored"] is not None]
    stored_o1 = [r["o1_score_stored"] for r in results if r["o1_score_stored"] is not None]

    print("\n" + "=" * 60)
    print("OVERLAP ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Overlapping questions          : {n}")
    print(f"  Ground-truth answers agree   : {sum(r['gt_agree'] for r in results)}  ({pct(sum(r['gt_agree'] for r in results), n)})")
    print(f"  Questions with R1 CoT        : {sum(r['has_r1_reasoning'] for r in results)}")

    if stored_r1:
        print(f"  R1  (stored score)  : {sum(stored_r1)}/{len(stored_r1)}  ({pct(sum(stored_r1), len(stored_r1))})")
    if stored_o1:
        print(f"  O1  (stored score)  : {sum(stored_o1)}/{len(stored_o1)}  ({pct(sum(stored_o1), len(stored_o1))})")

    by_subj = defaultdict(list)
    for r in results:
        by_subj[r["subject_local"]].append(r)
    print("\nPer-subject breakdown:")
    for subj, rows in sorted(by_subj.items()):
        ns  = len(rows)
        r1s = sum(r["r1_score_stored"] or 0 for r in rows if r["r1_score_stored"] is not None)
        nr1 = sum(1 for r in rows if r["r1_score_stored"] is not None)
        print(f"  {subj}: {ns} questions, R1 acc {r1s}/{nr1} ({pct(r1s, nr1)})")

    print("\nCoverage of local QA by DUMY CoT:")
    for split_name in ["train", "dev"]:
        local_n   = sum(1 for it in local_items if it["_split"] == split_name)
        overlap_n = sum(1 for r in results if r["split"] == split_name)
        print(f"  {split_name}: {overlap_n}/{local_n}  ({pct(overlap_n, local_n)})")

    out_path = ROOT / "data" / "dumy_overlap_analysis.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {out_path}")

    r1_acc = sum(stored_r1) / len(stored_r1) if stored_r1 else sum(r["r1_correct"] for r in results) / n
    print("\n" + "=" * 60)
    if r1_acc >= 0.75:
        print(f"R1 accuracy = {r1_acc:.1%}  → Including DUMY CoT is likely BENEFICIAL.")
    elif r1_acc >= 0.50:
        print(f"R1 accuracy = {r1_acc:.1%}  → CoT quality is mixed; use r1_score=1 filter.")
    else:
        print(f"R1 accuracy = {r1_acc:.1%}  → R1 quality is LOW; adding it may HURT performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DUMY × local QA overlap analysis")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    main(cache_dir=args.cache_dir)
