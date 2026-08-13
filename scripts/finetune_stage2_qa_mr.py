#!/usr/bin/env python3
"""Stage 2: LoRA fine-tune on QA + MR from best stage-1 checkpoint (5 epochs)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_train_utils import ROOT, build_stage2_dataset, load_model_and_tokenizer, run_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--stage1-lora", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "stage2_qa_mr_lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--qa-merged", type=Path, default=ROOT / "train_data/QA/train_merged.jsonl")
    parser.add_argument("--mr", type=Path, default=ROOT / "train_data/MR/mr_train.jsonl")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, lora_path=args.stage1_lora)
    dataset = build_stage2_dataset(tokenizer, args.qa_merged, args.mr)
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
    print(f"Stage 2 done: {args.output_dir}")


if __name__ == "__main__":
    main()
