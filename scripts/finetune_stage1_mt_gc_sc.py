#!/usr/bin/env python3
"""Stage 1: LoRA fine-tune on MT + GC + SC (combined multi-task loss, 5 epochs)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_train_utils import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_R,
    DEFAULT_MAX_SEQ_LENGTH,
    ROOT,
    build_stage1_dataset,
    load_model_and_tokenizer,
    run_sft,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "stage1_mt_gc_sc_lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument(
        "--mt",
        type=Path,
        default=ROOT / "train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl",
        help="MT training data path",
    )
    parser.add_argument(
        "--mt-format",
        choices=["parallel", "version1"],
        default="parallel",
        help="parallel: src/tgt jsonl; version1: messages field (e.g. version 1/train.jsonl.gz)",
    )
    parser.add_argument(
        "--mt-lang-pairs",
        default="en-uk",
        help="For version1 format: en-uk, cs-uk, or comma-separated (e.g. en-uk,cs-uk / all)",
    )
    parser.add_argument("--gc", type=Path, default=ROOT / "train_data/GC/ua_gec_train_single_error.jsonl")
    parser.add_argument("--sc", type=Path, default=ROOT / "train_data/SC/mmlu_ukr_sc_train.jsonl")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    dataset = build_stage1_dataset(
        tokenizer,
        args.mt,
        args.gc,
        args.sc,
        mt_format=args.mt_format,
        mt_lang_pairs=args.mt_lang_pairs,
    )
    run_sft(
        model,
        tokenizer,
        dataset,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
    )
    print(f"Stage 1 done: {args.output_dir}")


if __name__ == "__main__":
    main()
