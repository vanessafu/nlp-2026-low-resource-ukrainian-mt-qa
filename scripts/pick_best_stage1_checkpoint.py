#!/usr/bin/env python3
"""Pick best stage-1 LoRA checkpoint by dev MT+GC+SC metrics."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def score_stage1(summary: dict) -> float:
    mt = summary.get("MT", {})
    gc = summary.get("GC", {})
    sc = summary.get("SC", {})
    bleu = mt.get("BLEU", 0) / 100.0
    gc_score = (
        gc.get("detection_accuracy", 0) + gc.get("correction_accuracy", 0)
    ) / 2
    sc_score = (
        sc.get("detection_accuracy", 0) + sc.get("correction_accuracy", 0)
    ) / 2
    return bleu + gc_score + sc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, default=ROOT / "outputs/stage1_mt_gc_sc_lora")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--out-file", type=Path, default=ROOT / "outputs/stage1_best_checkpoint.json")
    args = parser.parse_args()

    checkpoints = sorted(args.stage1_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        checkpoints = [args.stage1_dir] if (args.stage1_dir / "adapter_config.json").exists() else []

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {args.stage1_dir}")

    best_path = None
    best_score = -1.0
    results = {}

    for ckpt in checkpoints:
        eval_dir = ROOT / "outputs" / f"stage1_eval_{ckpt.name}"
        cmd = [
            sys.executable,
            str(ROOT / "eval_qwen35_ukrainian_zeroshot.py"),
            "--model",
            args.model,
            "--lora-path",
            str(ckpt),
            "--output-dir",
            str(eval_dir),
            "--force",
            "--tasks",
            "MT,GC,SC",
        ]
        print(f"Evaluating {ckpt.name}...", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        s = score_stage1(summary)
        results[str(ckpt)] = {"score": s, "summary": summary}
        print(f"  score={s:.4f} {summary}", flush=True)
        if s > best_score:
            best_score = s
            best_path = ckpt

    out = {"best_checkpoint": str(best_path), "best_score": best_score, "all": results}
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path} (score={best_score:.4f})")
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
