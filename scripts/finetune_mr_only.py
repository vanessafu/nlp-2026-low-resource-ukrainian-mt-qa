#!/usr/bin/env python3
"""MR-only LoRA fine-tune with Ukrainian input and Ukrainian solution (dev-aligned)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_train_utils import (
    DEFAULT_MAX_SEQ_LENGTH,
    ROOT,
    load_model_and_tokenizer,
    load_mr_ukr_dataset,
    run_sft,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--base-lora",
        type=Path,
        default=ROOT / "outputs/stage1_mt_gc_sc_lora/checkpoint-3931",
        help="LoRA checkpoint to continue from (default: best stage-1)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mr_only_ukr_lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--gsm8k", type=Path, default=ROOT / "train_data/MR/gsm8k_train.jsonl")
    parser.add_argument("--comp-math", type=Path, default=ROOT / "train_data/MR/competition_math_train.jsonl")
    parser.add_argument("--eval-after", action="store_true", help="Run MR dev eval when training finishes")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, lora_path=args.base_lora)
    dataset = load_mr_ukr_dataset(tokenizer, args.gsm8k, args.comp_math)
    print(f"MR-only Ukrainian dataset size: {len(dataset)}", flush=True)
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
    print(f"MR-only training done: {args.output_dir}")

    if args.eval_after:
        eval_dir = ROOT / "outputs/mr_only_ukr_dev_eval"
        cmd = [
            sys.executable,
            str(ROOT / "eval_qwen35_ukrainian_zeroshot.py"),
            "--model",
            args.model,
            "--lora-path",
            str(args.output_dir),
            "--output-dir",
            str(eval_dir),
            "--force",
            "--tasks",
            "MR",
        ]
        print(f"Running MR dev eval -> {eval_dir}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        print(f"MR dev eval: {summary.get('MR')}")


if __name__ == "__main__":
    main()
