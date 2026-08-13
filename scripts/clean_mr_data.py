#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raw files:
    gsm8k_train.jsonl            (7473)
    competition_math_train.jsonl (7500)
outpus:
    mr_train_answer_frac.jsonl   (14895)
"""
import json
import re
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GSM8K_PATH = ROOT / "train_data" / "MR" / "gsm8k_train.jsonl"
COMP_PATH  = ROOT / "train_data" / "MR" / "competition_math_train.jsonl"
OUT_PATH   = ROOT / "train_data" / "MR" / "mr_train.jsonl"
SYSTEM_EN = ("You are a mathematics assistant. Solve the problem with a short "
             "step-by-step solution in Ukrainian (a few concise steps, no "
             "repetition). Use fractions instead of decimals. At the end, write "
             "'Answer:' followed by the final answer in \\boxed{}.")

INSTR_UK = ("\nПриміт.: покажи коротке розв'язання (кілька кроків, "
            "використовуй дроби, не десяткові числа), потім напиши "
            "'Answer:' і остаточну відповідь у \\boxed{}.")


def strip_calc_markers(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"<<[^<>]*>>", "", text)
    return text.replace("<<", "").replace(">>", "")


_DEC = re.compile(r'(?<![\d.,])(\d*[.,]\d+)')
def _dec_to_frac(m):
    raw = m.group(0)
    norm = raw.replace(',', '.')
    if norm.startswith('.'):
        norm = '0' + norm
    try:
        f = Fraction(norm).limit_denominator(10000)
    except Exception:
        return raw
    if f.denominator == 1:
        return str(f.numerator)                       # 20.00 -> 20
    return f"\\frac{{{f.numerator}}}{{{f.denominator}}}"


def _protect_asy(text):
    blocks = []

    def stash(mm):
        blocks.append(mm.group(0))
        return f"@@ASY{len(blocks) - 1}@@"

    text = re.sub(r'\[asy\].*?\[/asy\]', stash, text, flags=re.DOTALL)
    return text, blocks


def _restore_asy(text, blocks):
    for i, b in enumerate(blocks):
        text = text.replace(f"@@ASY{i}@@", b)
    return text


def decimals_to_fractions(text: str) -> str:
    text = re.sub(r'(?<=[*/=\-+ ])([.,]\d+)', r'0\1', text)
    text, asy = _protect_asy(text)
    text = _DEC.sub(_dec_to_frac, text)
    text = _restore_asy(text, asy)
    text = re.sub(r'(?<![\d])[.,]00(?![\d])', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text

def extract_boxed(s: str):
    idx = s.rfind("\\boxed")
    if idx == -1:
        return None
    i = s.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
    return None


def unwrap_all_boxed(s: str) -> str:
    """把正文里所有 \boxed{X} 还原成普通文本 X (改动 4)。"""
    while True:
        idx = s.find("\\boxed")
        if idx == -1:
            return s
        i = s.find("{", idx)
        if i == -1:
            return s
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    s = s[:idx] + s[i + 1:j] + s[j + 1:]
                    break
        else:
            return s

def finalize(body: str, final: str) -> str:
    body = strip_calc_markers(body)          
    body = decimals_to_fractions(body)     
    body = unwrap_all_boxed(body) 
    body = re.sub(r"[ \t]+", " ", body).strip()
    return f"{body}\nAnswer: \\boxed{{{final.strip()}}}"


def clean_gsm8k(d):
    ans_uk = d["answer_uk"]
    m = re.search(r"####\s*([\-\d,\.]+)\s*$", ans_uk.strip())
    if not m:
        return None
    final = m.group(1).replace(",", "").strip() 
    body = ans_uk[:ans_uk.rfind("####")]
    return d["question_uk"], finalize(body, final)


def clean_competition(d):
    final = extract_boxed(d["answer_en"])   
    if final is None:
        return None
    ans_uk = d["answer_uk"]
    if "\\begin{масив}" in ans_uk or "\\text{ if }" in ans_uk:  
        return None
    return d["question_uk"], finalize(ans_uk.strip(), final)


def build_record(question, solution):
    return {"messages": [
        {"role": "system", "content": SYSTEM_EN},
        {"role": "user", "content": question.strip() + INSTR_UK},
        {"role": "assistant", "content": solution},
    ]}


def main():
    out = open(OUT_PATH, "w", encoding="utf-8")
    kept = {"gsm8k": 0, "comp": 0}
    dropped = {"gsm8k": 0, "comp": 0}

    for line in open(GSM8K_PATH, encoding="utf-8"):
        r = clean_gsm8k(json.loads(line))
        if r is None:
            dropped["gsm8k"] += 1
            continue
        out.write(json.dumps(build_record(*r), ensure_ascii=False) + "\n")
        kept["gsm8k"] += 1

    for line in open(COMP_PATH, encoding="utf-8"):
        r = clean_competition(json.loads(line))
        if r is None:
            dropped["comp"] += 1
            continue
        out.write(json.dumps(build_record(*r), ensure_ascii=False) + "\n")
        kept["comp"] += 1

    out.close()
    total = kept["gsm8k"] + kept["comp"]
    print(f"kept    : gsm8k={kept['gsm8k']}  comp={kept['comp']}  total={total}")
    print(f"dropped : gsm8k={dropped['gsm8k']}  comp={dropped['comp']}  "
          f"total={dropped['gsm8k'] + dropped['comp']}")
    print(f"written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
