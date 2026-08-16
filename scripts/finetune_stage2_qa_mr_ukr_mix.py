#!/usr/bin/env python3
"""Stage 2 (Ukrainian-only + mix): QA + MR + (ZNO train) + small MT/GC to reduce forgetting."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_train_utils import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_R,
    DEFAULT_MAX_SEQ_LENGTH,
    ROOT,
    build_stage2_ukr_mix_dataset,
    load_model_and_tokenizer,
    run_sft,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--stage1-lora", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "stage2_qa_mr_ukr_mix_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)

    # QA
    parser.add_argument("--mmlu-en", type=Path, default=ROOT / "train_data/QA/mmlu_en_train.jsonl")
    parser.add_argument(
        "--mmlu-ukr",
        type=Path,
        default=ROOT / "train_data/QA/mmlu_ukr/data/validation-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--zno-train",
        type=Path,
        default=ROOT / "llms-limited-resources2026/Ukrainian/QA/ukr_qa_train.jsonl",
        help="Shared-task ZNO train split (Ukrainian-only).",
    )

    # MR
    parser.add_argument(
        "--mr-train",
        type=Path,
        default=None,
        help="Pre-formatted MR jsonl with messages field (replaces gsm8k+comp-math).",
    )
    parser.add_argument("--gsm8k", type=Path, default=ROOT / "train_data/MR/gsm8k_train.jsonl")
    parser.add_argument("--comp-math", type=Path, default=ROOT / "train_data/MR/competition_math_train.jsonl")

    # Anti-forgetting MT/GC (small sampled subsets)
    parser.add_argument("--mt", type=Path, default=None, help="Optional MT training data path (to reduce forgetting).")
    parser.add_argument("--mt-format", choices=["parallel", "version1"], default="parallel")
    parser.add_argument("--mt-lang-pairs", default="en-uk", help="For version1: en-uk, cs-uk, or comma-separated.")
    parser.add_argument("--gc", type=Path, default=None, help="Optional GC training data path (to reduce forgetting).")
    parser.add_argument("--n-mt", type=int, default=5000, help="Sample this many MT examples per epoch (0 disables).")
    parser.add_argument("--n-gc", type=int, default=5000, help="GC examples: N>0 sample N; 0=off; N<0=full train set.")
    parser.add_argument(
        "--mt-cs",
        type=Path,
        default=None,
        help="version-1 path for Czech-to-Ukrainian MT (e.g. version 1/train.jsonl.gz).",
    )
    parser.add_argument(
        "--n-mt-cs",
        type=int,
        default=0,
        help="First N cs-uk rows from --mt-cs (file order); 0 disables.",
    )
    parser.add_argument(
        "--qa-v2",
        type=Path,
        default=None,
        help="version2 QA train jsonl.gz; converted to numeric Answer: format and merged with QA.",
    )

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, lora_path=args.stage1_lora)
    dataset = build_stage2_ukr_mix_dataset(
        tokenizer,
        mmlu_en_path=args.mmlu_en,
        mmlu_ukr_parquet=args.mmlu_ukr,
        zno_train_path=args.zno_train,
        mr_paths=[args.gsm8k, args.comp_math],
        mr_train_path=args.mr_train,
        mt_path=args.mt,
        mt_format=args.mt_format,
        mt_lang_pairs=args.mt_lang_pairs,
        gc_path=args.gc,
        n_mt=args.n_mt,
        n_gc=args.n_gc,
        mt_cs_path=args.mt_cs,
        n_mt_cs=args.n_mt_cs,
        qa_v2_path=args.qa_v2,
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
    print(f"Stage 2 UKR+mix done: {args.output_dir}")


if __name__ == "__main__":
    main()

