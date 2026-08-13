#!/usr/bin/env python3
"""Shared utilities for multi-task LoRA fine-tuning."""

from __future__ import annotations

import ast
import gzip
import json
import re
from pathlib import Path
from typing import Iterator

import torch
from datasets import Dataset, interleave_datasets
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]

MT_SYSTEM = "You are a professional English-to-Ukrainian translator."
MT_SYSTEM_CS_UK = "You are a professional Czech-to-Ukrainian translator."
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
MR_UKR_SYSTEM = (
    "You are a mathematics assistant. "
    "Solve the problem with a short step-by-step solution in Ukrainian (a few concise steps, no repetition). "
    "Use fractions instead of decimals. At the end, write 'Answer:' followed by the final answer in \\boxed{}."
)

QA_ZNO_SYSTEM_EN = (
    "You are an expert tutor for Ukrainian standardized exams (ZNO/NMT). "
    "You will be given a multiple-choice question. Read the question and options "
    "carefully, then answer with only the number of the correct option "
    "(e.g., 0, 1, 2 or 3)."
)
QA_RC_SYSTEM_EN = (
    "You are an expert in reading comprehension and text analysis. "
    "You will be given a passage and a multiple-choice question about it. "
    "Read the passage carefully, then choose the correct answer and give only "
    "its number (0, 1, 2 or 3)."
)
QA_MATH_SYSTEM_EN = (
    "You are a mathematics tutor preparing students for the ZNO/NMT exam. "
    "Solve the problem step by step inside a <think>...</think> block, "
    "then give a short final answer."
)
QA_COT_SYSTEM_EN = (
    "You are an expert tutor for Ukrainian standardized exams (ZNO/NMT). "
    "You will be given a multiple-choice question. Think step by step, then "
    "give your answer in the format «Відповідь: 1»."
)

# ZNO/RC/CoT answers are baked in as Cyrillic markers (А/Б/В/Г/Д) on the option
# lines ("<marker>: <text>") and in the assistant turn (bare marker, or a
# trailing "Відповідь: <marker>" for explicit CoT). Dev QA data uses numeric
# markers instead, so load_qa_merged_dataset() rewrites both to match.
_QA_MARKER_TO_NUM = {"А": "0", "Б": "1", "В": "2", "Г": "3", "Д": "4"}
_QA_OPTION_LINE_RE = re.compile(r"(?m)^([АБВГД])(: )")
_QA_EXPLICIT_ANSWER_RE = re.compile(r"(Відповідь\s*:\s*)([АБВГД])(\s*)$", re.IGNORECASE | re.UNICODE)


def _qa_letters_to_numbers(messages: list[dict]) -> list[dict]:
    system, user, assistant = messages
    user_content = _QA_OPTION_LINE_RE.sub(
        lambda m: f"{_QA_MARKER_TO_NUM[m.group(1)]}{m.group(2)}", user["content"]
    )
    stripped = assistant["content"].strip()
    if stripped in _QA_MARKER_TO_NUM:
        assistant_content = _QA_MARKER_TO_NUM[stripped]
    else:
        assistant_content = _QA_EXPLICIT_ANSWER_RE.sub(
            lambda m: m.group(1) + _QA_MARKER_TO_NUM.get(m.group(2), m.group(2)) + m.group(3),
            assistant["content"],
        )
    return [
        system,
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

# Exact Ukrainian strings baked into train_data/QA/train_merged.jsonl by
# scripts/qa/constants.py (SYSTEM_ZNO/SYSTEM_RC/SYSTEM_MATH) and
# scripts/qa/build_cot.py (SYSTEM_EXPLICIT_COT). Kept in sync manually --
# load_qa_merged_dataset() swaps these for the English versions above.
_QA_MERGED_SYSTEM_MAP: dict[str, str] = {
    (
        "Ти — експерт із підготовки до українських іспитів (ЗНО/НМТ). "
        "Тобі буде надано питання з кількома варіантами відповіді. "
        "Уважно прочитай питання та варіанти, потім відповідай лише літерою "
        "правильної відповіді (наприклад: А, Б, В або Г)."
    ): QA_ZNO_SYSTEM_EN,
    (
        "Ти — експерт із читання та аналізу текстів. "
        "Тобі буде надано текст-уривок і питання з кількома варіантами відповіді. "
        "Уважно прочитай уривок, потім вибери правильну відповідь і вкажи лише її "
        "літеру (А, Б, В або Г)."
    ): QA_RC_SYSTEM_EN,
    (
        "Ти — викладач математики та підготовки до ЗНО/НМТ. "
        "Розв'яжи задачу крок за кроком у блоці <think>…</think>, "
        "потім дай коротку остаточну відповідь."
    ): QA_MATH_SYSTEM_EN,
    (
        "Ти — експерт із підготовки до українських іспитів (ЗНО/НМТ). "
        "Тобі буде надано питання з кількома варіантами відповіді. "
        "Поміркуй крок за кроком, а потім вкажи відповідь у форматі «Відповідь: Б»."
    ): QA_COT_SYSTEM_EN,
}



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


_MT_LANG_CONFIG: dict[str, tuple[str, str]] = {
    "en-uk": ("English-to-Ukrainian", MT_SYSTEM),
    "cs-uk": ("Czech-to-Ukrainian", MT_SYSTEM_CS_UK),
}


def parse_mt_lang_pairs(lang_pairs: str) -> list[str]:
    if lang_pairs in {"all", "both"}:
        return list(_MT_LANG_CONFIG.keys())
    pairs = [p.strip() for p in lang_pairs.split(",") if p.strip()]
    unknown = [p for p in pairs if p not in _MT_LANG_CONFIG]
    if unknown:
        raise ValueError(f"Unsupported lang_pair(s): {unknown}")
    return pairs


def load_mt_version1_dataset(tokenizer, mt_path: Path, lang_pairs: str = "en-uk") -> Dataset:
    """Load MT from version-1 combined corpus (messages field, gzip ok).

    The embedded system prompt (baked in by converter.py) is replaced with
    lora_train_utils's own MT_SYSTEM / MT_SYSTEM_CS_UK so training always sees
    the canonical prompt regardless of what generated the source file.
    """
    if not mt_path.exists():
        raise FileNotFoundError(mt_path)
    selected = parse_mt_lang_pairs(lang_pairs)

    rows = []
    for row in iter_jsonl(mt_path):
        messages = parse_messages_field(row["messages"])
        system = messages[0]["content"]
        pair = next((p for p in selected if _MT_LANG_CONFIG[p][0] in system), None)
        if pair is None:
            continue
        messages = [{"role": "system", "content": _MT_LANG_CONFIG[pair][1]}, *messages[1:]]
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


def build_mr_ukr_messages(row: dict) -> list[dict]:
    """Ukrainian-only MR prompt aligned with dev eval."""
    q_uk = row["question_uk"]
    a_uk = row["answer_uk"]
    return [
        {"role": "system", "content": MR_UKR_SYSTEM},
        {"role": "user", "content": f"{q_uk}\n\nAnswer:"},
        {"role": "assistant", "content": a_uk},
    ]


def load_mr_ukr_dataset(tokenizer, *paths: Path) -> Dataset:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            if not row.get("question_uk") or not row.get("answer_uk"):
                continue
            rows.append(messages_to_text(tokenizer, build_mr_ukr_messages(row)))
    return Dataset.from_list(rows)


def load_qa_merged_dataset(tokenizer, path: Path) -> Dataset:
    """Load train_data/QA/train_merged.jsonl (ZNO + MMLU): swap the baked-in
    Ukrainian system prompt for its English equivalent via _QA_MERGED_SYSTEM_MAP,
    and rewrite Cyrillic-letter answer markers to numeric ones so the format
    matches the official dev QA data (see _qa_letters_to_numbers)."""
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    unmatched = 0
    for row in iter_jsonl(path):
        messages = parse_messages_field(row["messages"])
        en_system = _QA_MERGED_SYSTEM_MAP.get(messages[0]["content"])
        if en_system is None:
            unmatched += 1
            en_system = messages[0]["content"]
        messages = [{"role": "system", "content": en_system}, *messages[1:]]
        messages = _qa_letters_to_numbers(messages)
        rows.append(messages_to_text(tokenizer, messages))
    if unmatched:
        print(
            f"[warn] {unmatched} QA rows had an unrecognized system prompt (left as-is) -- "
            "scripts/qa/constants.py or build_cot.py may have changed",
            flush=True,
        )
    return Dataset.from_list(rows)


def load_mr_cleaned_dataset(tokenizer, path: Path) -> Dataset:
    """Load train_data/MR/mr_train.jsonl, produced by scripts/clean_mr_data.py."""
    if not path.exists():
        raise FileNotFoundError(path)
    return Dataset.from_list(
        [messages_to_text(tokenizer, parse_messages_field(r["messages"])) for r in read_jsonl(path)]
    )


def build_stage2_dataset(
    tokenizer,
    qa_merged_path: Path,
    mr_path: Path,
) -> Dataset:
    qa = load_qa_merged_dataset(tokenizer, qa_merged_path)
    mr = load_mr_cleaned_dataset(tokenizer, mr_path)
    print(f"Stage2 sizes: QA={len(qa)} MR={len(mr)}", flush=True)
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
