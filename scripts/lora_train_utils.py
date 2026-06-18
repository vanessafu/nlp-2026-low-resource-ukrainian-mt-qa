#!/usr/bin/env python3
"""Shared utilities for multi-task LoRA fine-tuning."""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import torch
from datasets import Dataset, interleave_datasets
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]

MT_SYSTEM = "You are a professional English-to-Ukrainian translator."
GC_SYSTEM = (
    "Check the Ukrainian sentence for grammatical errors. "
    "Each sentence has at most one error. "
    "First decide whether there is a problem: "
    "if the sentence is correct, output 'CORRECT'; "
    "if there is an error, output the wrong word and the correct word."
)
SC_SYSTEM = (
    "Check the Ukrainian sentence for spelling errors. "
    "Each sentence has at most one error. "
    "First decide whether there is a problem: "
    "if the sentence is correct, output 'CORRECT'; "
    "if there is an error, output the wrong word and the correct word."
)
QA_EN_SYSTEM = (
    "You are a bilingual exam assistant. "
    "Given an English multiple-choice question, provide the Ukrainian translation "
    "and the correct answer."
)
QA_UKR_SYSTEM = "You are a Ukrainian exam assistant. Answer the multiple choice question."
MR_SYSTEM = (
    "You are a bilingual mathematics assistant. "
    "Given an English math problem, provide the Ukrainian translation and solution."
)
MR_UKR_SYSTEM = (
    "You are a mathematics assistant. "
    "Solve the problem step by step and select the correct answer."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".jsonl.gz") else open
    mode = "rt" if opener is gzip.open else "r"
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_messages_field(raw) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    raise ValueError(f"Unsupported messages field type: {type(raw)}")


def apply_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def format_sc_gc_target(row: dict) -> str:
    if row["incorrect_word"] == "CORRECT":
        return "CORRECT"
    return f"{row['incorrect_word']} -> {row['correct_word']}"


def format_options_dict(choices: dict) -> str:
    return "\n".join(f"{k}. {choices[k]}" for k in sorted(choices, key=lambda x: int(x)))


def messages_to_text(tokenizer, messages: list[dict]) -> str:
    return {"text": apply_template(tokenizer, messages)}


# --- Stage 1 builders ---


def build_mt_messages(row: dict) -> list[dict]:
    src = row.get("src") or row.get("en", "")
    tgt = row.get("tgt") or row.get("uk", "")
    return [
        {"role": "system", "content": MT_SYSTEM},
        {"role": "user", "content": f"Translate to Ukrainian:\n{src}"},
        {"role": "assistant", "content": tgt},
    ]


def build_gc_messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": GC_SYSTEM},
        {"role": "user", "content": row["input_sentence"]},
        {"role": "assistant", "content": format_sc_gc_target(row)},
    ]


def build_sc_messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SC_SYSTEM},
        {"role": "user", "content": row["input_sentence"]},
        {"role": "assistant", "content": format_sc_gc_target(row)},
    ]


def load_mt_dataset(tokenizer, mt_path: Path) -> Dataset:
    rows = []
    if not mt_path.exists():
        raise FileNotFoundError(mt_path)
    for row in read_jsonl(mt_path):
        if row.get("lang_pair", "en_uk") != "en_uk":
            continue
        rows.append(messages_to_text(tokenizer, build_mt_messages(row)))
    return Dataset.from_list(rows)


def parse_mt_lang_pairs(lang_pairs: str) -> list[str]:
    lang_markers = {
        "en-uk": "English-to-Ukrainian",
        "cs-uk": "Czech-to-Ukrainian",
    }
    if lang_pairs in {"all", "both"}:
        return list(lang_markers.keys())
    pairs = [p.strip() for p in lang_pairs.split(",") if p.strip()]
    unknown = [p for p in pairs if p not in lang_markers]
    if unknown:
        raise ValueError(f"Unsupported lang_pair(s): {unknown}")
    return pairs


def load_mt_version1_dataset(tokenizer, mt_path: Path, lang_pairs: str = "en-uk") -> Dataset:
    """Load MT from version-1 combined corpus (messages field, gzip ok)."""
    if not mt_path.exists():
        raise FileNotFoundError(mt_path)
    lang_markers = {
        "en-uk": "English-to-Ukrainian",
        "cs-uk": "Czech-to-Ukrainian",
    }
    selected = parse_mt_lang_pairs(lang_pairs)
    markers = {lang_markers[p] for p in selected}

    rows = []
    for row in iter_jsonl(mt_path):
        messages = parse_messages_field(row["messages"])
        system = messages[0]["content"]
        if not any(marker in system for marker in markers):
            continue
        rows.append(messages_to_text(tokenizer, messages))
    return Dataset.from_list(rows)


def load_gc_dataset(tokenizer, path: Path) -> Dataset:
    return Dataset.from_list(
        [messages_to_text(tokenizer, build_gc_messages(r)) for r in read_jsonl(path)]
    )


def load_sc_dataset(tokenizer, path: Path) -> Dataset:
    return Dataset.from_list(
        [messages_to_text(tokenizer, build_sc_messages(r)) for r in read_jsonl(path)]
    )


def build_mt_only_dataset(
    tokenizer,
    mt_path: Path,
    mt_format: str = "version1",
    mt_lang_pairs: str = "en-uk,cs-uk",
) -> Dataset:
    if mt_format == "version1":
        mt = load_mt_version1_dataset(tokenizer, mt_path, lang_pairs=mt_lang_pairs)
    elif mt_format == "parallel":
        mt = load_mt_dataset(tokenizer, mt_path)
    else:
        raise ValueError(f"Unknown mt_format: {mt_format}")
    print(f"MT-only size: {len(mt)}", flush=True)
    return mt


def build_stage1_dataset(
    tokenizer,
    mt_path: Path,
    gc_path: Path,
    sc_path: Path,
    mt_format: str = "parallel",
    mt_lang_pairs: str = "en-uk",
) -> Dataset:
    """Interleave MT, GC, SC with equal probability (combined multi-task loss)."""
    if mt_format == "version1":
        mt = load_mt_version1_dataset(tokenizer, mt_path, lang_pairs=mt_lang_pairs)
    elif mt_format == "parallel":
        mt = load_mt_dataset(tokenizer, mt_path)
    else:
        raise ValueError(f"Unknown mt_format: {mt_format}")
    gc = load_gc_dataset(tokenizer, gc_path)
    sc = load_sc_dataset(tokenizer, sc_path)
    print(f"Stage1 sizes: MT={len(mt)} GC={len(gc)} SC={len(sc)}", flush=True)
    return interleave_datasets(
        [mt, gc, sc],
        probabilities=[1 / 3, 1 / 3, 1 / 3],
        seed=42,
        stopping_strategy="all_exhausted",
    )


# --- Stage 2 builders ---


def build_qa_en_messages(row: dict) -> list[dict]:
    q_en = row["question_en"]
    opts_en = format_options_dict(row["possible_answers_en"])
    q_uk = row["question_uk"]
    opts_uk = format_options_dict(row["possible_answers_uk"])
    ans = str(row["correct_answer_num"])
    assistant = (
        f"Question (Ukrainian): {q_uk}\n\n"
        f"{opts_uk}\n\n"
        f"Answer: {ans}"
    )
    return [
        {"role": "system", "content": QA_EN_SYSTEM},
        {"role": "user", "content": f"{q_en}\n\n{opts_en}\n\nAnswer:"},
        {"role": "assistant", "content": assistant},
    ]


def build_qa_ukr_messages(row: dict) -> list[dict]:
    question = row["question"]
    opts = format_options_dict(row["possible_answers"])
    ans = str(row["correct_answer_num"])
    return [
        {"role": "system", "content": QA_UKR_SYSTEM},
        {"role": "user", "content": f"{question}\n\n{opts}\n\nAnswer:"},
        {"role": "assistant", "content": ans},
    ]


def build_mr_messages(row: dict) -> list[dict]:
    q_en = row.get("question_en") or row.get("question", "")
    q_uk = row["question_uk"]
    a_uk = row["answer_uk"]
    assistant = f"Question (Ukrainian): {q_uk}\n\nSolution:\n{a_uk}"
    return [
        {"role": "system", "content": MR_SYSTEM},
        {"role": "user", "content": f"{q_en}\n\nAnswer:"},
        {"role": "assistant", "content": assistant},
    ]


def build_mr_ukr_messages(row: dict) -> list[dict]:
    """Ukrainian-only MR prompt aligned with dev eval."""
    q_uk = row["question_uk"]
    a_uk = row["answer_uk"]
    return [
        {"role": "system", "content": MR_UKR_SYSTEM},
        {"role": "user", "content": f"{q_uk}\n\nAnswer:"},
        {"role": "assistant", "content": a_uk},
    ]


def load_mmlu_ukr_qa(tokenizer, parquet_path: Path) -> Dataset:
    table = pq.read_table(parquet_path)
    records = table.to_pydict()
    n = len(records["question"])
    rows = []
    for i in range(n):
        row = {
            "question": records["question"][i],
            "possible_answers": {str(j): c for j, c in enumerate(records["choices"][i])},
            "correct_answer_num": records["answer"][i],
        }
        rows.append(messages_to_text(tokenizer, build_qa_ukr_messages(row)))
    return Dataset.from_list(rows)


def load_mr_bilingual(tokenizer, *paths: Path) -> Dataset:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            rows.append(messages_to_text(tokenizer, build_mr_messages(row)))
    return Dataset.from_list(rows)


def load_mr_ukr_dataset(tokenizer, *paths: Path) -> Dataset:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            if not row.get("question_uk") or not row.get("answer_uk"):
                continue
            rows.append(messages_to_text(tokenizer, build_mr_ukr_messages(row)))
    return Dataset.from_list(rows)


def build_stage2_dataset(
    tokenizer,
    mmlu_en_path: Path,
    mmlu_ukr_parquet: Path,
    mr_paths: list[Path],
) -> Dataset:
    qa_en = Dataset.from_list(
        [messages_to_text(tokenizer, build_qa_en_messages(r)) for r in read_jsonl(mmlu_en_path)]
    )
    qa_ukr = load_mmlu_ukr_qa(tokenizer, mmlu_ukr_parquet)
    mr = load_mr_bilingual(tokenizer, *mr_paths)
    print(f"Stage2 sizes: QA_en={len(qa_en)} QA_ukr={len(qa_ukr)} MR={len(mr)}", flush=True)
    qa = interleave_datasets(
        [qa_en, qa_ukr],
        probabilities=[0.5, 0.5],
        seed=42,
        stopping_strategy="all_exhausted",
    )
    return interleave_datasets(
        [qa, mr],
        probabilities=[0.5, 0.5],
        seed=42,
        stopping_strategy="all_exhausted",
    )


def load_model_and_tokenizer(
    model_name: str,
    lora_path: Path | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    tokenizer = AutoTokenizer.from_pretrained(
        str(lora_path) if lora_path else model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer


def run_sft(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_seq_length: int = 2048,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=epochs,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return output_dir
