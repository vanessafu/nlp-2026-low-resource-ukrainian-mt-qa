#!/usr/bin/env python3
"""Shared utilities for multi-task LoRA fine-tuning."""

from __future__ import annotations

import ast
import gzip
import json
import re
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import torch
from datasets import Dataset, concatenate_datasets, interleave_datasets
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_MAX_SEQ_LENGTH = 3072

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
QA_UKR_SYSTEM = (
    "You are a Ukrainian exam assistant. Answer the multiple choice question "
    "by selecting the correct option."
)
QA_CYRILLIC_LABELS = "АБВГД"
V2_QA_LABELS = QA_CYRILLIC_LABELS
MR_SYSTEM = (
    "You are a bilingual mathematics assistant. "
    "Given an English math problem, provide the Ukrainian translation and solution."
)
MR_UKR_SYSTEM = (
    "You are a mathematics assistant. "
    "Show your step-by-step solution, "
    "then give the final answer after 'Answer:'."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".jsonl.gz") else open
    mode = "rt" if opener is gzip.open else "r"
    try:
        with opener(path, mode, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except EOFError:
        # tolerate truncated .gz (e.g. incomplete version2 MT exports)
        return


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


def cyrillic_label(index: int | str) -> str:
    return QA_CYRILLIC_LABELS[int(index)]


def format_options_cyrillic(choices: dict) -> str:
    return "\n".join(
        f"{cyrillic_label(k)}: {choices[k]}"
        for k in sorted(choices, key=lambda x: int(x))
    )


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


def load_mt_version1_lang_dataset(
    tokenizer,
    mt_path: Path,
    lang_pair: str,
    n_max: int | None = None,
) -> Dataset:
    """Load one lang pair from version-1 corpus; optional cap (first n in file order)."""
    if not mt_path.exists():
        raise FileNotFoundError(mt_path)
    lang_markers = {
        "en-uk": "English-to-Ukrainian",
        "cs-uk": "Czech-to-Ukrainian",
    }
    if lang_pair not in lang_markers:
        raise ValueError(f"Unsupported lang_pair: {lang_pair}")
    marker = lang_markers[lang_pair]

    rows = []
    for row in iter_jsonl(mt_path):
        messages = parse_messages_field(row["messages"])
        system = messages[0]["content"]
        if marker not in system:
            continue
        rows.append(messages_to_text(tokenizer, messages))
        if n_max is not None and len(rows) >= n_max:
            break
    return Dataset.from_list(rows)


def load_mt_version1_dataset(tokenizer, mt_path: Path, lang_pairs: str = "en-uk") -> Dataset:
    """Load MT from version-1 combined corpus (messages field, gzip ok)."""
    selected = parse_mt_lang_pairs(lang_pairs)
    parts = [load_mt_version1_lang_dataset(tokenizer, mt_path, p) for p in selected]
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


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


def build_stage1_mt_sc_dataset(
    tokenizer,
    mt_path: Path,
    sc_path: Path,
    mt_format: str = "parallel",
    mt_lang_pairs: str = "en-uk",
) -> Dataset:
    """Interleave MT (en-uk) + SC only; GC deferred to stage 2."""
    if mt_format == "version1":
        mt = load_mt_version1_dataset(tokenizer, mt_path, lang_pairs=mt_lang_pairs)
    elif mt_format == "parallel":
        mt = load_mt_dataset(tokenizer, mt_path)
    else:
        raise ValueError(f"Unknown mt_format: {mt_format}")
    sc = load_sc_dataset(tokenizer, sc_path)
    print(f"Stage1 MT+SC sizes: MT={len(mt)} SC={len(sc)}", flush=True)
    return interleave_datasets(
        [mt, sc],
        probabilities=[0.5, 0.5],
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


def format_mr_ukr_assistant(row: dict) -> str:
    """Steps in assistant, final answer after 'Answer:'."""
    a_uk = (row.get("answer_uk") or "").strip()
    if re.search(r"(?i)Answer\s*:", a_uk):
        return a_uk
    final = row.get("final_answer")
    if final is not None and str(final).strip():
        steps = a_uk.rsplit("####", 1)[0].strip() if "####" in a_uk else a_uk
        return f"{steps}\n\nAnswer: {str(final).strip()}"
    return a_uk


def build_mr_ukr_messages(row: dict) -> list[dict]:
    """Ukrainian-only MR prompt aligned with dev eval."""
    q_uk = row["question_uk"]
    return [
        {"role": "system", "content": MR_UKR_SYSTEM},
        {"role": "user", "content": f"{q_uk}\n\nAnswer:"},
        {"role": "assistant", "content": format_mr_ukr_assistant(row)},
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


def load_mr_ukr_dataset(
    tokenizer,
    *paths: Path,
    difficulty: str | None = None,
) -> Dataset:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            if difficulty is not None and row.get("difficulty") != difficulty:
                continue
            if not row.get("question_uk") or not row.get("answer_uk"):
                continue
            rows.append(messages_to_text(tokenizer, build_mr_ukr_messages(row)))
    return Dataset.from_list(rows)


def build_qa_en_translated_ukr_messages(row: dict) -> list[dict]:
    """Ukrainian QA from translated mmlu_en_train (numeric options + Answer:)."""
    q_uk = row["question_uk"]
    opts_uk = format_options_dict(row["possible_answers_uk"])
    ans = str(row["correct_answer_num"])
    return [
        {"role": "system", "content": QA_UKR_SYSTEM},
        {"role": "user", "content": f"{q_uk}\n\n{opts_uk}\n\nAnswer:"},
        {"role": "assistant", "content": ans},
    ]


def load_qa_en_translated_ukr(tokenizer, jsonl_path: Path) -> Dataset:
    rows = []
    for row in read_jsonl(jsonl_path):
        if not row.get("question_uk") or not row.get("possible_answers_uk"):
            continue
        rows.append(messages_to_text(tokenizer, build_qa_en_translated_ukr_messages(row)))
    return Dataset.from_list(rows)


def load_zno_ukr_qa(tokenizer, jsonl_path: Path) -> Dataset:
    """Load Ukrainian ZNO QA jsonl (shared-task train split)."""
    rows = []
    for row in read_jsonl(jsonl_path):
        rows.append(messages_to_text(tokenizer, build_qa_ukr_messages(row)))
    return Dataset.from_list(rows)


def parse_version2_qa_fields(user: str, assistant: str) -> dict | None:
    """Convert version2 QA (Cyrillic labels) to dev-aligned numeric QA fields."""
    opts = re.findall(r"(?:^|\n)([АБВГД])\s*:\s*([^\n]+)", user)
    if not opts:
        return None

    opt_start = user.rfind(f"\n{opts[0][0]}:")
    if opt_start < 0:
        opt_start = user.find(f"{opts[0][0]}:")
    question = user[:opt_start].strip() if opt_start >= 0 else user.strip()
    question = re.sub(r"\nВідповідь\s*:\s*[АБВГД]\s*$", "", question, flags=re.IGNORECASE).strip()
    if "\nТЕМА:" in question and len(assistant.strip()) <= 2:
        question = question.split("\nТЕМА:")[0].strip()

    ans_letter = None
    if len(assistant.strip()) <= 2:
        ans_letter = assistant.strip()
    else:
        match = re.search(r"Відповідь\s*:\s*([АБВГД])", assistant, flags=re.IGNORECASE)
        if match:
            ans_letter = match.group(1)
    if not ans_letter:
        match = re.search(r"Відповідь\s*:\s*([АБВГД])", question, flags=re.IGNORECASE)
        if match:
            ans_letter = match.group(1)
    label_to_idx = {label: str(i) for i, label in enumerate(V2_QA_LABELS)}
    if not ans_letter or ans_letter not in label_to_idx:
        return None
    if ans_letter not in {label for label, _ in opts}:
        return None

    choices = {str(i): text.strip() for i, (_, text) in enumerate(opts)}
    return {
        "question": question,
        "possible_answers": choices,
        "correct_answer_num": label_to_idx[ans_letter],
    }


def build_qa_ukr_messages_from_v2(question: str, opts: list[tuple[str, str]], ans_letter: str) -> list[dict]:
    options = "\n".join(f"{label}: {text}" for label, text in opts)
    return [
        {"role": "system", "content": QA_UKR_SYSTEM},
        {"role": "user", "content": f"{question}\n\n{options}"},
        {"role": "assistant", "content": ans_letter},
    ]


def load_version2_qa_ukr(tokenizer, jsonl_path: Path) -> Dataset:
    """Load version2 QA converted to numeric Answer: format."""
    rows = []
    skipped = 0
    for row in iter_jsonl(jsonl_path):
        messages = parse_messages_field(row["messages"])
        user = next(m["content"] for m in messages if m["role"] == "user")
        assistant = next(m["content"] for m in messages if m["role"] == "assistant")
        parsed = parse_version2_qa_fields(user, assistant)
        if not parsed:
            skipped += 1
            continue
        rows.append(messages_to_text(tokenizer, build_qa_ukr_messages(parsed)))
    print(f"version2 QA loaded (numeric): {len(rows)} (skipped {skipped})", flush=True)
    return Dataset.from_list(rows)


def load_messages_dataset(tokenizer, jsonl_path: Path) -> Dataset:
    """Load pre-formatted chat messages (e.g. version2 QA/MT jsonl.gz)."""
    rows = []
    for row in iter_jsonl(jsonl_path):
        messages = parse_messages_field(row["messages"])
        rows.append(messages_to_text(tokenizer, messages))
    return Dataset.from_list(rows)


def _maybe_sample(ds: Dataset, n: int | None, seed: int = 42) -> Dataset:
    if n is None:
        return ds
    n = int(n)
    if n <= 0:
        return ds.select([])
    if n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


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


def build_stage2_ukr_dataset(
    tokenizer,
    mmlu_en_path: Path,
    mmlu_ukr_parquet: Path,
    mr_paths: list[Path],
) -> Dataset:
    """Stage 2 with Ukrainian input/output only (dev-aligned QA + MR)."""
    qa_en_uk = load_qa_en_translated_ukr(tokenizer, mmlu_en_path)
    qa_ukr = load_mmlu_ukr_qa(tokenizer, mmlu_ukr_parquet)
    mr = load_mr_ukr_dataset(tokenizer, *mr_paths)
    print(
        f"Stage2 Ukrainian sizes: QA_en_uk={len(qa_en_uk)} QA_ukr={len(qa_ukr)} MR={len(mr)}",
        flush=True,
    )
    qa = interleave_datasets(
        [qa_en_uk, qa_ukr],
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


def build_stage2_ukr_mix_dataset(
    tokenizer,
    mmlu_en_path: Path,
    mmlu_ukr_parquet: Path,
    zno_train_path: Path,
    mr_paths: list[Path],
    mr_train_path: Path | None = None,
    mt_path: Path | None = None,
    mt_format: str = "parallel",
    mt_lang_pairs: str = "en-uk",
    gc_path: Path | None = None,
    n_mt: int | None = 5000,
    n_gc: int | None = 5000,
    mt_cs_path: Path | None = None,
    n_mt_cs: int | None = None,
    qa_v2_path: Path | None = None,
) -> Dataset:
    """Stage 2 (Ukrainian in/out) + ZNO train + small MT/GC to reduce forgetting.

    If qa_v2_path is set, add version2 QA converted to numeric Answer: format.
    mt_cs_path: version-1/2 gzip/jsonl; first n_mt_cs cs-uk rows (file order).
    mr_train_path: pre-formatted MR jsonl with messages field (overrides mr_paths).
    """
    if mr_train_path is not None:
        mr = load_messages_dataset(tokenizer, mr_train_path)
    else:
        mr = load_mr_ukr_dataset(tokenizer, *mr_paths)

    qa_en_uk = load_qa_en_translated_ukr(tokenizer, mmlu_en_path)
    qa_ukr = load_mmlu_ukr_qa(tokenizer, mmlu_ukr_parquet)
    qa_zno = load_zno_ukr_qa(tokenizer, zno_train_path)
    qa_parts = [qa_en_uk, qa_ukr, qa_zno]
    qa_log = f"QA_en_uk={len(qa_en_uk)} QA_ukr={len(qa_ukr)} QA_zno={len(qa_zno)}"

    if qa_v2_path is not None:
        qa_v2 = load_version2_qa_ukr(tokenizer, qa_v2_path)
        qa_parts.append(qa_v2)
        qa_log += f" QA_v2={len(qa_v2)}"

    mt = None
    if mt_path is not None:
        if mt_format == "version1":
            mt = load_mt_version1_dataset(tokenizer, mt_path, lang_pairs=mt_lang_pairs)
        elif mt_format == "parallel":
            mt = load_mt_dataset(tokenizer, mt_path)
        else:
            raise ValueError(f"Unknown mt_format: {mt_format}")
        mt = _maybe_sample(mt, n_mt, seed=42)

    mt_cs = None
    if mt_cs_path is not None and n_mt_cs is not None and n_mt_cs > 0:
        mt_cs = load_mt_version1_lang_dataset(tokenizer, mt_cs_path, "cs-uk", n_max=n_mt_cs)

    gc = None
    if gc_path is not None and n_gc != 0:
        gc = load_gc_dataset(tokenizer, gc_path)
        if n_gc is not None and n_gc > 0:
            gc = _maybe_sample(gc, n_gc, seed=42)

    parts = [*qa_parts, mr]
    if mt is not None:
        parts.append(mt)
    if mt_cs is not None:
        parts.append(mt_cs)
    if gc is not None:
        parts.append(gc)

    out = concatenate_datasets(parts)
    out = out.shuffle(seed=42)

    print(
        "Stage2 UKR+mix sizes: "
        f"{qa_log} MR={len(mr)} "
        + (f"MT_en_sampled={len(mt)} " if mt is not None else "")
        + (f"MT_cs={len(mt_cs)} " if mt_cs is not None else "")
        + (f"GC={len(gc)}" if gc is not None else ""),
        flush=True,
    )
    return out


def build_joint_ukr_mix_dataset(
    tokenizer,
    mt_path: Path,
    gc_path: Path,
    sc_path: Path,
    mmlu_en_path: Path,
    mmlu_ukr_parquet: Path,
    zno_train_path: Path,
    mr_paths: list[Path],
    mt_format: str = "parallel",
    mt_lang_pairs: str = "en-uk",
    mt_cs_path: Path | None = None,
    n_mt_cs: int | None = 10000,
    mix_strategy: str = "interleave",
) -> Dataset:
    """Single-stage joint training: all Stage-1 + Stage-2 task data from base model.

    Uses full MT/GC/SC (no anti-forgetting subsampling). MT cs-uk capped like Stage-2
    canonical recipe (first n_mt_cs rows in file order).
    """
    if mt_format == "version1":
        mt_en = load_mt_version1_dataset(tokenizer, mt_path, lang_pairs=mt_lang_pairs)
    elif mt_format == "parallel":
        mt_en = load_mt_dataset(tokenizer, mt_path)
    else:
        raise ValueError(f"Unknown mt_format: {mt_format}")

    gc = load_gc_dataset(tokenizer, gc_path)
    sc = load_sc_dataset(tokenizer, sc_path)

    qa_en_uk = load_qa_en_translated_ukr(tokenizer, mmlu_en_path)
    qa_ukr = load_mmlu_ukr_qa(tokenizer, mmlu_ukr_parquet)
    qa_zno = load_zno_ukr_qa(tokenizer, zno_train_path)
    mr = load_mr_ukr_dataset(tokenizer, *mr_paths)

    mt_cs = None
    if mt_cs_path is not None and n_mt_cs is not None and n_mt_cs > 0:
        mt_cs = load_mt_version1_lang_dataset(tokenizer, mt_cs_path, "cs-uk", n_max=n_mt_cs)

    print(
        "Joint sizes: "
        f"MT_en={len(mt_en)} "
        + (f"MT_cs={len(mt_cs)} " if mt_cs is not None else "")
        + f"GC={len(gc)} SC={len(sc)} "
        f"QA_en_uk={len(qa_en_uk)} QA_ukr={len(qa_ukr)} QA_zno={len(qa_zno)} "
        f"MR={len(mr)} mix={mix_strategy}",
        flush=True,
    )

    if mix_strategy == "interleave":
        task_parts = [mt_en, gc, sc, qa_en_uk, qa_ukr, qa_zno, mr]
        if mt_cs is not None:
            task_parts.insert(1, mt_cs)
        n = len(task_parts)
        return interleave_datasets(
            task_parts,
            probabilities=[1 / n] * n,
            seed=42,
            stopping_strategy="all_exhausted",
        )
    if mix_strategy == "shuffle":
        parts = [mt_en]
        if mt_cs is not None:
            parts.append(mt_cs)
        parts.extend([gc, sc, qa_en_uk, qa_ukr, qa_zno, mr])
        out = concatenate_datasets(parts)
        return out.shuffle(seed=42)
    raise ValueError(f"Unknown mix_strategy: {mix_strategy}")


def load_model_and_tokenizer(
    model_name: str,
    lora_path: Path | None = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
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
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    save_steps: int = 250,
    resume: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if epochs >= 2:
        save_kwargs = {"save_strategy": "epoch", "save_total_limit": epochs}
    else:
        save_kwargs = {
            "save_strategy": "steps",
            "save_steps": save_steps,
            "save_total_limit": 4,
        }
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        **save_kwargs,
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
    resume_ckpt = None
    if resume:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )
        if checkpoints:
            resume_ckpt = True
            print(f"Resuming from latest checkpoint in {output_dir}", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return output_dir
