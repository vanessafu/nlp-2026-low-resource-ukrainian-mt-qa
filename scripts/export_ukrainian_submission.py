#!/usr/bin/env python3
"""Convert stage2 test preds into official Ukrainian/ submission folder (+ optional zip)."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = {k: row[k] for k in keys}
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {path} n={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=ROOT / "outputs" / "stage2_ukr_test_preds",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Ukrainian",
        help="Output folder named Ukrainian/ (submission layout)",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional zip path (e.g. Ukrainian_submission.zip)",
    )
    args = parser.parse_args()

    pred = args.pred_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    mt_rows = load_jsonl(pred / "ces-ukr_preds.jsonl") + load_jsonl(pred / "eng-ukr_preds.jsonl")
    write_jsonl(out / "ukr_mt_test.jsonl", mt_rows, ["dataset_id", "sent_id", "source", "pred"])

    qa_rows = load_jsonl(pred / "ukr-qa_preds.jsonl") + load_jsonl(pred / "ukr-qa_mmlu_preds.jsonl")
    write_jsonl(out / "ukr_qa_test.jsonl", qa_rows, ["dataset_id", "question_id", "question", "pred"])

    write_jsonl(
        out / "ukr_sc_test.jsonl",
        load_jsonl(pred / "ukr-sc_preds.jsonl"),
        ["dataset_id", "id", "input_sentence", "pred_incorrect", "pred_corrected"],
    )
    write_jsonl(
        out / "ukr_gc_test.jsonl",
        load_jsonl(pred / "ukr-gc_preds.jsonl"),
        ["dataset_id", "id", "input_sentence", "pred_incorrect", "pred_corrected"],
    )
    write_jsonl(
        out / "ukr_mr_test.jsonl",
        load_jsonl(pred / "ukr-mr_preds.jsonl"),
        ["dataset_id", "id", "question", "pred"],
    )

    if args.zip_path is not None:
        args.zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out.glob("*.jsonl")):
                zf.write(f, arcname=f"Ukrainian/{f.name}")
        print(f"Wrote {args.zip_path}", flush=True)

    print(f"Submission folder ready: {out}", flush=True)


if __name__ == "__main__":
    main()
