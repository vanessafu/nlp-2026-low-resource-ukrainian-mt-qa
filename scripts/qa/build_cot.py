"""
Build QA training corpus.

CoT items (DUMY dataset, o1_score=1):
  70 % implicit CoT — reasoning between question and options in user turn;
                       assistant outputs just the answer letter.
  30 % explicit CoT — user = question + options; assistant = reasoning ending
                       with «Відповідь: Б».

ZNO 2× augmented:
  All 2026-official ukr_qa_train.jsonl questions, factor=3, two variants
  selected per question:
    CoT-overlap questions → variants[1:3]  (both shuffled; identity excluded)
    Other questions       → variants[0, 2] (identity + prefixed-shuffle)

RC:
  Belebele ukr_Cyrl  — data/training/QA/belebele_rc.json
  QUA-RC             — data/training/QA/quarc_rc.json

Outputs (train_data/QA/):
  zno_cot_items.jsonl   — CoT items only
  ukr_zno_qa_train.jsonl — CoT + ZNO + RC combined
  build_stats.json      — per-source counts

Run:
  python scripts/qa/build_cot.py [--cache-dir <hf-cache>] [--dry-run]
"""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from augment import augment as _augment, _apply_permutation, _format_user
from constants import SYSTEM_ZNO, SYSTEM_RC, SYSTEM_MATH, load_qa_records, normalize

ROOT      = Path(__file__).resolve().parent.parent.parent
QA_2026   = ROOT / "train_data/QA"
ZNO_TRAIN = QA_2026 / "ukr_qa_train.jsonl"
ZNO_DEV   = QA_2026 / "ukr_qa_dev.jsonl"
BELEBELE  = ROOT / "data/training/QA/belebele_rc.json"
QUARC     = ROOT / "data/training/QA/quarc_rc.json"
OUT_DIR   = QA_2026

SYSTEM_EXPLICIT_COT = (
    "Ти — експерт із підготовки до українських іспитів (ЗНО/НМТ). "
    "Тобі буде надано питання з кількома варіантами відповіді. "
    "Поміркуй крок за кроком, а потім вкажи відповідь у форматі «Відповідь: Б»."
)

def _shuffle_local(answers: list[dict], correct_marker: str, seed: int) -> tuple[list[dict], str]:
    """Return a non-identity shuffle of *answers* with the correct marker remapped.

    Used so the CoT variant of a local question differs from the identity
    variant produced by ZNO 2× augmentation.
    """
    n = len(answers)
    rng = random.Random(seed)
    perm = list(range(n))
    # Prefer a non-identity permutation; fall back to whatever we get.
    for _ in range(30):
        rng.shuffle(perm)
        if perm != list(range(n)):
            break
    return _apply_permutation(answers, correct_marker, perm)


def clean_o1(text: str) -> str:
    text = re.sub(r"^```[^\n]*\n?", "", text.strip())
    text = re.sub(r"\n?```$",        "", text.strip())
    return text.strip()


def is_implicit(question: str) -> bool:
    """Deterministic 70 % implicit / 30 % explicit split."""
    return int(hashlib.md5(question.encode()).hexdigest(), 16) % 10 < 7


def local_to_user(item: dict) -> str:
    return _format_user(item["question"], item.get("answers", []), prefix=item.get("_prefix", ""))


def make_std(user: str, answer: str, system: str = SYSTEM_ZNO) -> dict:
    return {"messages": [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": answer},
    ]}


def make_implicit_cot(question: str, reasoning: str, options_str: str,
                      answer_letter: str, system: str) -> dict:
    """Reasoning inserted between question stem and options in the user turn."""
    user = f"{question}\n\n{reasoning}\n\n{options_str}"
    return {"messages": [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": answer_letter},
    ]}


def make_explicit_cot(question: str, reasoning: str, options_str: str,
                      answer_letter: str, system: str) -> dict:
    """Reasoning in the assistant turn; guaranteed to end with «Відповідь: X»."""
    user = f"{question}\n\n{options_str}"
    r    = reasoning.rstrip()
    if not re.search(
        r"Відповідь\s*:\s*" + re.escape(answer_letter) + r"\s*$",
        r, re.I | re.U,
    ):
        r = r + f"\nВідповідь: {answer_letter}"
    return {"messages": [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": r},
    ]}



def main(cache_dir: str | None, dry_run: bool) -> None:

    # ── 1. ZNO 2026 official (train_data/QA) ─────────────────────────────── #
    local_train = load_qa_records(ZNO_TRAIN)
    eval_norms: set[str] = {normalize(it["question"]) for it in load_qa_records(ZNO_DEV)}
    local_by_norm = {normalize(it["question"]): it for it in local_train}
    print(f"Local train : {len(local_train)} items")
    print(f"Eval norms  : {len(eval_norms)} (dev, excluded from CoT)")

    # ── 2. DUMY ────────────────────────────────────────────────────────── #
    print("Loading DUMY …")
    kw = {"cache_dir": cache_dir} if cache_dir else {}
    ds = load_dataset("NLPForUA/dumy-zno-ukrainian-math-history-geo-r1-o1", **kw)
    dumy: list[dict] = [it for split in ds for it in ds[split]]
    print(f"  {len(dumy)} DUMY items (all subjects)")

    # ── 3. CoT items  ── #
    print("\nBuilding CoT items …")
    cot_items:  list[dict] = []
    cot_norms:  set[str]   = set()
    n_impl = n_expl = n_skip_eval = n_no_local = 0
    n = 0
    for item in dumy:
        if item.get("o1_score") != 1:
            continue
        reasoning = clean_o1(item.get("o1_answer") or "")
        if not reasoning:
            continue
        n += 1
        qnorm = normalize(item["question"])
        if qnorm in eval_norms:
            n_skip_eval += 1
            continue

        local = local_by_norm.get(qnorm)
        if local:
            # Shuffle local answers so this CoT variant differs from the
            # identity copy produced by ZNO 2× augmentation.
            raw_answers   = local["answers"]
            raw_correct   = (local.get("correct_answers") or [""])[0]
            answers, answer_letter = _shuffle_local(
                raw_answers, raw_correct,
                seed=hash(item["question"]) & 0xFFFFFFFF,
            )
        else:
            # No local match — use DUMY's own answer fields directly.
            answers = item.get("answers") or []
            correct = item.get("correct_answers") or []
            if not answers or not correct:
                cot_norms.add(qnorm)
                continue
            answer_letter = correct[0]
            n_no_local += 1

        opts_str = "\n".join(f"{a['marker']}: {a['text']}" for a in answers)
        subj     = item["subject"]
        sys_std  = SYSTEM_MATH if subj == "math" else SYSTEM_ZNO
        sys_cot  = SYSTEM_EXPLICIT_COT

        if is_implicit(item["question"]):
            cot_items.append(make_implicit_cot(
                item["question"], reasoning, opts_str, answer_letter, sys_std))
            n_impl += 1
        else:
            cot_items.append(make_explicit_cot(
                item["question"], reasoning, opts_str, answer_letter, sys_cot))
            n_expl += 1
        cot_norms.add(qnorm)

    print(f"  DUMY items     : {len(dumy)}")
    print(f"  valid (o1=1)   : {n}")
    print(f"  CoT items      : {len(cot_items)}"
          f"  (implicit={n_impl}, explicit={n_expl})")
    print(f"    with local    : {len(cot_items) - n_no_local} (answers shuffled)")
    print(f"    no local      : {n_no_local} (DUMY answers used directly)")
    print(f"  Skipped eval   : {n_skip_eval}")
    print(f"  CoT norms      : {len(cot_norms)}")

    # ── 4. ZNO 2× augment from local train.json ───────────────────────── #
    print("\nBuilding ZNO 2× items …")
    # Generate 3 variants per question; select 2 based on CoT overlap:
    #   CoT-overlap → variants[1:3]  (both shuffled; identity skipped)
    #   Regular     → variants[0, 2] (identity + prefixed-shuffle)
    aug_all = _augment(local_train, factor=3, seed=42)

    aug_by_norm: dict[str, list[dict]] = defaultdict(list)
    for rec in aug_all:
        aug_by_norm[normalize(rec["question"])].append(rec)

    zno_items:      list[dict] = []
    n_cot_q = n_reg_q = 0

    for qnorm, variants in aug_by_norm.items():
        if qnorm in cot_norms:
            use     = variants[1:3]              # both shuffled
            n_cot_q += 1
        elif len(variants) >= 3:
            use     = [variants[0], variants[2]] # identity + prefixed-shuffle
            n_reg_q += 1
        else:
            use     = variants[:2]
            n_reg_q += 1

        for rec in use:
            ans = rec.get("correct_answers") or []
            if ans:
                zno_items.append(make_std(local_to_user(rec), ans[0]))

    print(f"  ZNO 2× items : {len(zno_items)}")
    print(f"    CoT-overlap : {n_cot_q} questions (both shuffled, no identity)")
    print(f"    Regular     : {n_reg_q} questions (identity + prefixed-shuffle)")

    # ── 5. Belebele + QUA-RC ──────────────────────────────────────────── #
    with open(BELEBELE, encoding="utf-8") as f:
        belebele: list[dict] = json.load(f)
    with open(QUARC, encoding="utf-8") as f:
        quarc: list[dict] = json.load(f)

    seen_norms: set[str] = set(aug_by_norm.keys())
    rc_items:   list[dict] = []

    for item in belebele:
        key = normalize(item["question"])
        if key in seen_norms:
            continue
        seen_norms.add(key)
        ans = item.get("correct_answers") or []
        if ans:
            sys = SYSTEM_RC if item.get("subject") == "reading-comprehension" else SYSTEM_ZNO
            rc_items.append(make_std(local_to_user(item), ans[0], sys))

    for item in quarc:
        key = normalize(item["question"])
        if key in seen_norms:
            continue
        seen_norms.add(key)
        ans = item.get("correct_answers") or []
        if ans:
            rc_items.append(make_std(local_to_user(item), ans[0], SYSTEM_RC))

    print(f"  RC items (Belebele + QUA-RC) : {len(rc_items)}")

    # ── 6. Summary ─────────────────────────────────────────────────────── #
    all_items = cot_items + zno_items + rc_items
    print(f"\nTotal : {len(all_items)}"
          f"  ({len(cot_items)} CoT + {len(zno_items)} ZNO + {len(rc_items)} RC)")

    if dry_run:
        print("\n[DRY RUN] — no files written.")
        return

    # ── 7. Write ────────────────────────────────────────────────────────── #
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, items: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  Written: {path}  ({len(items)} items)")

    write_jsonl(OUT_DIR / "zno_cot_items.jsonl",    cot_items)
    write_jsonl(OUT_DIR / "ukr_zno_qa_train.jsonl", all_items)

    stats = {
        "cot_total":      len(cot_items),
        "cot_implicit":   n_impl,
        "cot_explicit":   n_expl,
        "cot_overlap_q":  n_cot_q,
        "zno_2x":         len(zno_items),
        "rc_total":       len(rc_items),
        "grand_total":    len(all_items),
    }
    with open(OUT_DIR / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Written: {OUT_DIR / 'build_stats.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(args.cache_dir, args.dry_run)
