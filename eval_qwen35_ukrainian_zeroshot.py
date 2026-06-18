import argparse
import json
import math
import re
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("llms-limited-resources2026/Ukrainian")
OUT_DIR = Path("outputs/qwen3_5_2b_ukrainian_zeroshot")

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def normalize_text(text):
    text = strip_thinking(str(text))
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_options(possible_answers):
    return "\n".join(f"{k}. {possible_answers[k]}" for k in sorted(possible_answers, key=lambda x: int(x)))


def build_examples():
    examples = {}

    mt_rows = read_jsonl(ROOT / "MT/en-ukr_mt_dev.jsonl")
    examples["MT"] = [
        {
            "id": row["sent_id"],
            "messages": [
                {"role": "system", "content": "You are a professional English-to-Ukrainian translator."},
                {"role": "user", "content": f"Translate to Ukrainian:\n{row['en']}"},
            ],
            "reference": row["uk"],
        }
        for row in mt_rows
    ]

    cs_mt_system = (
        "You are a professional Czech-to-Ukrainian translator, tasked with providing "
        "translations suitable for use in Ukraine (uk_UA). Your goal is to accurately "
        "convey the meaning and nuances of the original Czech text while adhering to "
        "Ukrainian grammar, vocabulary, and cultural sensitivities. Produce only the "
        "Ukrainian translation, without any additional explanations or commentary. "
        "Retain the paragraph breaks (double new lines) from the input text. "
        "Please translate the following Czech text into Ukrainian (uk_UA):"
    )
    cs_mt_rows = read_jsonl(ROOT / "MT/cs-ukr_mt_dev.jsonl")
    examples["MT_CS"] = [
        {
            "id": row["sent_id"],
            "messages": [
                {"role": "system", "content": cs_mt_system},
                {"role": "user", "content": row["cs"]},
            ],
            "reference": row["uk"],
        }
        for row in cs_mt_rows
    ]

    qa_rows = read_jsonl(ROOT / "QA/ukr_qa_dev.jsonl") + read_jsonl(ROOT / "QA/ukr_mmlu_qa_dev.jsonl")
    examples["QA"] = [
        {
            "id": f"{row['dataset_id']}::{i}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Ukrainian exam assistant. Answer the multiple choice question.",
                },
                {"role": "user", "content": f"{row['question']}\n\n{format_options(row['possible_answers'])}\n\nAnswer:"},
            ],
            "reference": str(row["correct_answer_num"]),
            "possible_answers": row["possible_answers"],
        }
        for i, row in enumerate(qa_rows)
    ]

    sc_rows = read_jsonl(ROOT / "SC/ukr_sc_dev.jsonl")
    examples["SC"] = [
        {
            "id": row["id"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Check the Ukrainian sentence for spelling errors. "
                        "Each sentence has at most one error. "
                        "First decide whether there is a problem: "
                        "if the sentence is correct, output 'CORRECT'; "
                        "if there is an error, output the wrong word and the correct word."
                    ),
                },
                {"role": "user", "content": row["input_sentence"]},
            ],
            "incorrect_word": row["incorrect_word"],
            "correct_word": row["correct_word"],
        }
        for row in sc_rows
    ]

    gc_rows = read_jsonl(ROOT / "GC/ukr_gc_dev.jsonl")
    examples["GC"] = [
        {
            "id": row["id"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Check the Ukrainian sentence for grammatical errors. "
                        "Each sentence has at most one error. "
                        "First decide whether there is a problem: "
                        "if the sentence is correct, output 'CORRECT'; "
                        "if there is an error, output the wrong word and the correct word."
                    ),
                },
                {"role": "user", "content": row["input_sentence"]},
            ],
            "incorrect_word": row["incorrect_word"],
            "correct_word": row["correct_word"],
        }
        for row in gc_rows
    ]

    mr_rows = read_jsonl(ROOT / "MR/ukr_mr_dev.jsonl")
    examples["MR"] = [
        {
            "id": row["id"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a mathematics assistant. "
                        "Solve the problem step by step and select the correct answer."
                    ),
                },
                {"role": "user", "content": f"{row['question']}\n\nAnswer:"},
            ],
            "reference": row["answer"],
        }
        for row in mr_rows
    ]

    return examples


def apply_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_task(model, tokenizer, task, examples, batch_size, max_new_tokens, force=False):
    out_path = OUT_DIR / f"{task}.jsonl"
    if force and out_path.exists():
        out_path.unlink()
    if out_path.exists():
        done_rows = read_jsonl(out_path)
        if len(done_rows) == len(examples):
            return done_rows
    else:
        done_rows = []

    start = len(done_rows)
    rows = list(done_rows)
    prompts = [apply_template(tokenizer, ex["messages"]) for ex in examples[start:]]

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc=task):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_examples = examples[start + batch_start : start + batch_start + len(batch_prompts)]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for ex, text in zip(batch_examples, texts):
            row = {"id": ex["id"], "prediction": normalize_text(text)}
            for key in ("reference", "incorrect_word", "correct_word", "possible_answers"):
                if key in ex:
                    row[key] = ex[key]
            rows.append(row)
        write_jsonl(out_path, rows)

    return rows


def parse_choice(prediction, possible_answers):
    text = strip_thinking(prediction)
    valid = set(possible_answers.keys())
    for pattern in [
        r"(?:Answer|Відповідь)\s*[:\-]?\s*([0-9]+)",
        r"^\s*([0-9]+)\s*[\).:\-]?",
        r"\b([0-9]+)\b",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1) in valid:
            return match.group(1)
    text_norm = normalize_text(text).casefold()
    for key, value in possible_answers.items():
        if normalize_text(value).casefold() in text_norm:
            return key
    return None


def parse_error_pair(prediction):
    text = normalize_text(prediction)
    low = text.casefold()
    no_error_markers = ["no error", "correct", "немає помилки", "без помилки", "помилки немає"]
    if any(marker in low for marker in no_error_markers):
        return "CORRECT", "CORRECT"

    patterns = [
        r"(?:wrong word|incorrect word|помилкове слово|неправильне слово)\s*[:\-]\s*([^\n,;]+).*?(?:correct word|correction|виправлення|правильне слово)\s*[:\-]\s*([^\n,;]+)",
        r"([^\s,;:\"'`]+)\s*(?:->|→|=>|—|-)\s*([^\s,;:\"'`]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_text(match.group(1)), normalize_text(match.group(2))

    quoted = re.findall(r"[\"'`«„“]([^\"'`»“]+)[\"'`»“]", text)
    if len(quoted) >= 2:
        return normalize_text(quoted[0]), normalize_text(quoted[1])

    tokens = re.findall(r"[\wА-Яа-яІіЇїЄєҐґ'’\-]+", text, flags=re.UNICODE)
    if len(tokens) >= 2:
        return normalize_text(tokens[0]), normalize_text(tokens[1])
    return None, None


def parse_number(prediction):
    text = strip_thinking(prediction)
    matches = re.findall(r"-?\d+(?:[.,]\d+)?", text.replace(" ", ""))
    if not matches:
        return None
    value = matches[-1].replace(",", ".")
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def score(task, rows):
    if task in {"MT", "MT_CS"}:
        preds = [row["prediction"] for row in rows]
        refs = [[row["reference"] for row in rows]]
        bleu = sacrebleu.corpus_bleu(preds, refs)
        chrf = sacrebleu.corpus_chrf(preds, refs)
        return {"n": len(rows), "BLEU": bleu.score, "chrF2": chrf.score}

    if task == "QA":
        correct = 0
        parsed = 0
        for row in rows:
            pred = parse_choice(row["prediction"], row["possible_answers"])
            parsed += pred is not None
            correct += pred == str(row["reference"])
        return {"n": len(rows), "accuracy": correct / len(rows), "parsed": parsed / len(rows)}

    if task in {"SC", "GC"}:
        detection = 0
        pair = 0
        parsed = 0
        for row in rows:
            wrong, correct_word = parse_error_pair(row["prediction"])
            parsed += wrong is not None and correct_word is not None
            ref_wrong = normalize_text(row["incorrect_word"])
            ref_correct = normalize_text(row["correct_word"])
            pred_wrong = normalize_text(wrong)
            pred_correct = normalize_text(correct_word)
            detection += pred_wrong.casefold() == ref_wrong.casefold()
            pair += (
                pred_wrong.casefold() == ref_wrong.casefold()
                and pred_correct.casefold() == ref_correct.casefold()
            )
        return {
            "n": len(rows),
            "detection_accuracy": detection / len(rows),
            "pair_accuracy": pair / len(rows),
            "parsed": parsed / len(rows),
        }

    if task == "MR":
        correct = 0
        parsed = 0
        for row in rows:
            pred = parse_number(row["prediction"])
            parsed += pred is not None
            correct += pred == row["reference"]
        return {"n": len(rows), "accuracy": correct / len(rows), "parsed": parsed / len(rows)}

    raise ValueError(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task list, e.g. MT,GC,SC (default: all)",
    )
    args = parser.parse_args()

    global OUT_DIR
    if args.output_dir is not None:
        OUT_DIR = args.output_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.lora_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True,
    )
    if args.lora_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    examples = build_examples()
    max_new_tokens = {"MT": 192, "MT_CS": 192, "QA": 16, "SC": 40, "GC": 40, "MR": 256}
    batch_sizes = {
        "MT": args.batch_size,
        "MT_CS": args.batch_size,
        "QA": args.batch_size,
        "SC": args.batch_size,
        "GC": args.batch_size,
        "MR": 4,
    }
    summary = {}

    task_list = ["MT", "QA", "SC", "GC", "MR"]
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for task in task_list:
        rows = generate_task(
            model, tokenizer, task, examples[task], batch_sizes[task], max_new_tokens[task], force=args.force
        )
        summary[task] = score(task, rows)
        print(task, json.dumps(summary[task], ensure_ascii=False), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
