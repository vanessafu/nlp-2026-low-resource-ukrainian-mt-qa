#!/usr/bin/env python3
"""Pick best stage-2 LoRA checkpoint by dev MT+MT_CS+QA+SC+GC+MR metrics."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def score_stage2(summary: dict) -> float:
    mt = summary.get("MT", {})
    mt_cs = summary.get("MT_CS", {})
    qa = summary.get("QA", {})
    sc = summary.get("SC", {})
    gc = summary.get("GC", {})
    mr = summary.get("MR", {})
    bleu = mt.get("BLEU", 0) / 100.0
    chrf = mt_cs.get("chrF++", 0) / 100.0
    qa_acc = qa.get("accuracy", 0)
    mr_acc = mr.get("accuracy", 0)
    gc_score = (
        gc.get("detection_accuracy", 0) + gc.get("correction_accuracy", 0)
    ) / 2
    sc_score = (
        sc.get("detection_accuracy", 0) + sc.get("correction_accuracy", 0)
    ) / 2
    return bleu + chrf + qa_acc + mr_acc + gc_score + sc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Default: outputs/stage2_best_checkpoint.json next to stage2-dir parent",
    )
    parser.add_argument(
        "--eval-prefix",
        default="stage2_ckpt_eval",
        help="Subdir prefix under outputs/ for per-checkpoint eval",
    )
    args = parser.parse_args()

    stage2_dir = args.stage2_dir
    checkpoints = sorted(
        stage2_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints and (stage2_dir / "adapter_config.json").exists():
        checkpoints = [stage2_dir]

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {stage2_dir}")

    out_file = args.out_file or (ROOT / "outputs" / "stage2_best_checkpoint.json")

    best_path = None
    best_score = -1.0
    results = {}

    for ckpt in checkpoints:
        eval_dir = ROOT / "outputs" / f"{args.eval_prefix}_{ckpt.name}"
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
            "MT,MT_CS,QA,SC,GC,MR",
        ]
        print(f"Evaluating {ckpt.name}...", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        s = score_stage2(summary)
        results[str(ckpt)] = {"score": s, "summary": summary}
        print(f"  score={s:.4f}", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if s > best_score:
            best_score = s
            best_path = ckpt

    out = {
        "best_checkpoint": str(best_path),
        "best_score": best_score,
        "all": results,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path} (score={best_score:.4f})")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
