#!/usr/bin/env python3
"""Run Ukrainian TEST inference with a LoRA checkpoint; export official pred JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_qwen35_ukrainian_zeroshot import (  # noqa: E402
    apply_template,
    extract_mr_final_answer,
    format_options,
    normalize_text,
    parse_choice,
    parse_error_pair,
    read_jsonl,
    write_jsonl,
)

DATA_ROOT = ROOT / "llms-limited-resources2026" / "Ukrainian"

MT_EN_SYSTEM = "You are a professional English-to-Ukrainian translator."
MT_CS_SYSTEM = (
    "You are a professional Czech-to-Ukrainian translator, tasked with providing "
    "translations suitable for use in Ukraine (uk_UA). Your goal is to accurately "
    "convey the meaning and nuances of the original Czech text while adhering to "
    "Ukrainian grammar, vocabulary, and cultural sensitivities. Produce only the "
    "Ukrainian translation, without any additional explanations or commentary. "
    "Retain the paragraph breaks (double new lines) from the input text. "
    "Please translate the following Czech text into Ukrainian (uk_UA):"
)
SC_SYSTEM = (
    "Check the Ukrainian sentence for spelling errors. "
    "Each sentence has at most one error. "
    "First decide whether there is a problem: "
    "if the sentence is correct, output 'CORRECT'; "
    "if there is an error, output the wrong word and the correct word."
)
GC_SYSTEM = (
    "Check the Ukrainian sentence for grammatical errors. "
    "Each sentence has at most one error. "
    "First decide whether there is a problem: "
    "if the sentence is correct, output 'CORRECT'; "
    "if there is an error, output the wrong word and the correct word."
)
QA_SYSTEM = (
    "You are a Ukrainian exam assistant. Answer the multiple choice question."
)
MR_SYSTEM = (
    "You are a mathematics assistant. "
    "Show your step-by-step solution, "
    "then give the final answer after 'Answer:'."
)

ALL_TASKS = ("MT", "MT_CS", "QA", "SC", "GC", "MR")


def split_oversized_block(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    lines = text.split("\n")
    current = ""
    for line in lines:
        candidate = line if not current else f"{current}\n{line}"
        if len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            sentence_parts = re.split(r"(?<=[.!?])\s+", line)
            buf = ""
            for part in sentence_parts:
                piece = part if not buf else f"{buf} {part}"
                if len(part) > max_chars:
                    if buf:
                        chunks.append(buf)
                        buf = ""
                    for i in range(0, len(part), max_chars):
                        chunks.append(part[i : i + max_chars])
                elif len(piece) <= max_chars:
                    buf = piece
                else:
                    if buf:
                        chunks.append(buf)
                    buf = part
            if buf:
                current = buf
            continue

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return chunks


def mt_source_chunks(text: str, max_chars: int) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    for para in paragraphs:
        chunks.extend(split_oversized_block(para, max_chars))
    return chunks


def mt_messages(system: str, source_text: str) -> list[dict]:
    if system == MT_EN_SYSTEM:
        user = f"Translate to Ukrainian:\n{source_text}"
    else:
        user = source_text
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def max_new_tokens_for_mt_chunk(chunk: str) -> int:
    return int(min(1024, max(384, len(chunk) * 0.75 + 64)))


@torch.inference_mode()
def generate_texts(
    model,
    tokenizer,
    messages_list: list[list[dict]],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    outputs: list[str] = []
    prompts = [apply_template(tokenizer, messages) for messages in messages_list]
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        texts = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        outputs.extend(normalize_text(text) for text in texts)
    return outputs


@torch.inference_mode()
def translate_mt_document(
    model,
    tokenizer,
    source_text: str,
    system: str,
    max_chunk_chars: int,
) -> str:
    paragraphs = source_text.split("\n\n")
    translated_paragraphs: list[str] = []
    for para in paragraphs:
        if not para.strip():
            translated_paragraphs.append("")
            continue
        subchunks = split_oversized_block(para, max_chunk_chars)
        translated_sub: list[str] = []
        for chunk in subchunks:
            if not chunk.strip():
                continue
            pred = generate_texts(
                model,
                tokenizer,
                [mt_messages(system, chunk)],
                batch_size=1,
                max_new_tokens=max_new_tokens_for_mt_chunk(chunk),
            )[0]
            translated_sub.append(pred)
        translated_paragraphs.append("\n".join(translated_sub))
    return "\n\n".join(translated_paragraphs)


def checking_preds(prediction: str) -> tuple[str, str]:
    wrong, correct = parse_error_pair(prediction)
    if wrong is None or correct is None:
        return "NO_OUTPUT", "NO_OUTPUT"
    if wrong == "CORRECT" or correct == "CORRECT":
        return "CORRECT", "CORRECT"
    return wrong, correct


def load_model(model_name: str, lora_path: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path or model_name,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True,
    )
    if lora_path is not None:
        from peft import PeftModel

        base = PeftModel.from_pretrained(base, str(lora_path))
    base.eval()
    return base, tokenizer


def infer_mt(
    model,
    tokenizer,
    split_name: str,
    src_key: str,
    system: str,
    out_path: Path,
    max_chunk_chars: int,
    force: bool,
) -> None:
    rows_in = read_jsonl(DATA_ROOT / "MT" / f"{split_name}_mt_test.jsonl")
    if not force and out_path.exists():
        done = read_jsonl(out_path)
        if len(done) == len(rows_in):
            print(f"skip {out_path.name}: already complete ({len(done)})")
            return

    out_rows = read_jsonl(out_path) if out_path.exists() and not force else []
    start = len(out_rows)
    for row in tqdm(rows_in[start:], desc=out_path.stem):
        source = row[src_key]
        pred = translate_mt_document(
            model,
            tokenizer,
            source,
            system,
            max_chunk_chars=max_chunk_chars,
        )
        out_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "sent_id": row["sent_id"],
                "source": source,
                "pred": pred,
            }
        )
        write_jsonl(out_path, out_rows)


def infer_batch_task(
    model,
    tokenizer,
    task: str,
    examples: list[dict],
    out_path: Path,
    batch_size: int,
    max_new_tokens: int,
    force: bool,
    row_builder,
) -> None:
    if not force and out_path.exists():
        done = read_jsonl(out_path)
        if len(done) == len(examples):
            print(f"skip {out_path.name}: already complete ({len(done)})")
            return

    out_rows = read_jsonl(out_path) if out_path.exists() and not force else []
    start = len(out_rows)
    pending = examples[start:]
    for batch_start in tqdm(range(0, len(pending), batch_size), desc=out_path.stem):
        batch_examples = pending[batch_start : batch_start + batch_size]
        messages_list = [ex["messages"] for ex in batch_examples]
        preds = generate_texts(
            model,
            tokenizer,
            messages_list,
            batch_size=len(batch_examples),
            max_new_tokens=max_new_tokens,
        )
        for ex, pred in zip(batch_examples, preds):
            out_rows.append(row_builder(ex, pred))
        write_jsonl(out_path, out_rows)


def build_qa_examples(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    return [
        {
            "row": row,
            "messages": [
                {"role": "system", "content": QA_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{row['question']}\n\n"
                        f"{format_options(row['possible_answers'])}\n\n"
                        "Answer:"
                    ),
                },
            ],
        }
        for row in rows
    ]


def build_sc_gc_examples(path: Path, system: str) -> list[dict]:
    rows = read_jsonl(path)
    return [
        {
            "row": row,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": row["input_sentence"]},
            ],
        }
        for row in rows
    ]


def mr_difficulty(example_id: str) -> str:
    if example_id.startswith("low-"):
        return "low"
    if example_id.startswith("medium-"):
        return "medium"
    return "default"


def build_mr_examples(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    return [
        {
            "row": row,
            "difficulty": mr_difficulty(row["id"]),
            "messages": [
                {"role": "system", "content": MR_SYSTEM},
                {"role": "user", "content": f"{row['question']}\n\nAnswer:"},
            ],
        }
        for row in rows
    ]


def infer_mr_task(
    model,
    tokenizer,
    examples: list[dict],
    out_path: Path,
    batch_size: int,
    tokens_by_difficulty: dict[str, int],
    force: bool,
    row_builder,
) -> None:
    if not force and out_path.exists():
        done = read_jsonl(out_path)
        if len(done) == len(examples):
            print(f"skip {out_path.name}: already complete ({len(done)})", flush=True)
            return

    if force and out_path.exists():
        out_path.unlink()

    grouped: dict[str, list[dict]] = {}
    for ex in examples:
        grouped.setdefault(ex["difficulty"], []).append(ex)

    print(
        "MR difficulty sizes: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(grouped.items())),
        flush=True,
    )
    print(f"MR max_new_tokens: {tokens_by_difficulty}", flush=True)

    preds_by_id: dict[str, dict] = {}
    for difficulty, group in sorted(grouped.items()):
        max_new_tokens = tokens_by_difficulty.get(
            difficulty,
            tokens_by_difficulty.get("default", 384),
        )
        for batch_start in tqdm(
            range(0, len(group), batch_size),
            desc=f"ukr-mr-{difficulty}",
        ):
            batch_examples = group[batch_start : batch_start + batch_size]
            messages_list = [ex["messages"] for ex in batch_examples]
            preds = generate_texts(
                model,
                tokenizer,
                messages_list,
                batch_size=len(batch_examples),
                max_new_tokens=max_new_tokens,
            )
            for ex, pred in zip(batch_examples, preds):
                preds_by_id[ex["row"]["id"]] = row_builder(ex, pred)

    out_rows = [preds_by_id[ex["row"]["id"]] for ex in examples]
    write_jsonl(out_path, out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=ROOT / "outputs/stage2_qa_mr_ukr_mix_cs_lora/checkpoint-1843",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/stage2_ukr_test_preds",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mt-chunk-chars", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--tasks",
        default=",".join(ALL_TASKS),
        help="Comma-separated: MT,MT_CS,QA,SC,GC,MR",
    )
    parser.add_argument(
        "--mr-max-tokens-low",
        type=int,
        default=512,
        help="max_new_tokens for low-difficulty MR (short arithmetic)",
    )
    parser.add_argument(
        "--mr-max-tokens-medium",
        type=int,
        default=2048,
        help="max_new_tokens for medium-difficulty MR (long LaTeX derivations)",
    )
    parser.add_argument(
        "--mr-batch-size",
        type=int,
        default=None,
        help="MR batch size (default: min(4, --batch-size))",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.lora_path)
    print(f"Loaded model with LoRA: {args.lora_path}", flush=True)

    if "MT" in tasks:
        infer_mt(
            model,
            tokenizer,
            "en-ukr",
            "en",
            MT_EN_SYSTEM,
            args.output_dir / "eng-ukr_preds.jsonl",
            args.mt_chunk_chars,
            args.force,
        )

    if "MT_CS" in tasks:
        infer_mt(
            model,
            tokenizer,
            "cs-ukr",
            "cs",
            MT_CS_SYSTEM,
            args.output_dir / "ces-ukr_preds.jsonl",
            args.mt_chunk_chars,
            args.force,
        )

    if "SC" in tasks:
        examples = build_sc_gc_examples(DATA_ROOT / "SC/ukr_sc_test.jsonl", SC_SYSTEM)
        infer_batch_task(
            model,
            tokenizer,
            "SC",
            examples,
            args.output_dir / "ukr-sc_preds.jsonl",
            args.batch_size,
            40,
            args.force,
            lambda ex, pred: {
                "dataset_id": ex["row"]["dataset_id"],
                "id": ex["row"]["id"],
                "input_sentence": ex["row"]["input_sentence"],
                "pred_incorrect": checking_preds(pred)[0],
                "pred_corrected": checking_preds(pred)[1],
            },
        )

    if "GC" in tasks:
        examples = build_sc_gc_examples(DATA_ROOT / "GC/ukr_gc_test.jsonl", GC_SYSTEM)
        infer_batch_task(
            model,
            tokenizer,
            "GC",
            examples,
            args.output_dir / "ukr-gc_preds.jsonl",
            args.batch_size,
            40,
            args.force,
            lambda ex, pred: {
                "dataset_id": ex["row"]["dataset_id"],
                "id": ex["row"]["id"],
                "input_sentence": ex["row"]["input_sentence"],
                "pred_incorrect": checking_preds(pred)[0],
                "pred_corrected": checking_preds(pred)[1],
            },
        )

    if "QA" in tasks:
        for name, path in [
            ("ukr-qa_preds.jsonl", DATA_ROOT / "QA/ukr_qa_test.jsonl"),
            ("ukr-qa_mmlu_preds.jsonl", DATA_ROOT / "QA/ukr_mmlu_qa_test.jsonl"),
        ]:
            examples = build_qa_examples(path)
            infer_batch_task(
                model,
                tokenizer,
                "QA",
                examples,
                args.output_dir / name,
                args.batch_size,
                32,
                args.force,
                lambda ex, pred, _name=name: {
                    "dataset_id": ex["row"]["dataset_id"],
                    "question_id": ex["row"]["question_id"],
                    "question": ex["row"]["question"],
                    "pred": int(choice)
                    if (choice := parse_choice(pred, ex["row"]["possible_answers"])) is not None
                    else -1,
                },
            )

    if "MR" in tasks:
        examples = build_mr_examples(DATA_ROOT / "MR/ukr_mr_test.jsonl")
        mr_batch_size = args.mr_batch_size or max(1, min(4, args.batch_size))
        infer_mr_task(
            model,
            tokenizer,
            examples,
            args.output_dir / "ukr-mr_preds.jsonl",
            mr_batch_size,
            {
                "low": args.mr_max_tokens_low,
                "medium": args.mr_max_tokens_medium,
                "default": args.mr_max_tokens_medium,
            },
            args.force,
            lambda ex, pred: {
                "dataset_id": ex["row"]["dataset_id"],
                "id": ex["row"]["id"],
                "question": ex["row"]["question"],
                "pred": extract_mr_final_answer(pred),
            },
        )

    print(f"Done. Predictions written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
