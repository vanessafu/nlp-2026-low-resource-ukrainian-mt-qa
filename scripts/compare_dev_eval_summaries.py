#!/usr/bin/env python3
"""Print side-by-side dev eval comparison from summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten_metrics(summary: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for task, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                out[f"{task}/{key}"] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    if len(args.labels) != len(args.summaries):
        raise SystemExit("--labels and --summaries must have the same length")

    rows: list[dict] = []
    all_keys: set[str] = set()
    flat_by_label: dict[str, dict[str, float]] = {}

    for label, path in zip(args.labels, args.summaries):
        summary = json.loads(path.read_text(encoding="utf-8"))
        flat = flatten_metrics(summary)
        flat_by_label[label] = flat
        all_keys.update(flat)
        rows.append({"label": label, "path": str(path), "summary": summary})

    keys = sorted(all_keys)
    table: dict[str, dict[str, float | None]] = {k: {} for k in keys}
    for label in args.labels:
        for k in keys:
            table[k][label] = flat_by_label[label].get(k)

    # Pretty print
    col_w = max(18, max(len(x) for x in args.labels) + 2)
    header = f"{'metric':<28}" + "".join(f"{lab:>{col_w}}" for lab in args.labels)
    if len(args.labels) >= 2:
        header += f"{'delta(last-first)':>18}"
    print(header)
    print("-" * len(header))
    for k in keys:
        vals = [flat_by_label[lab].get(k) for lab in args.labels]
        line = f"{k:<28}"
        for v in vals:
            line += f"{(f'{v:.4f}' if v is not None else 'n/a'):>{col_w}}"
        if len(vals) >= 2 and vals[0] is not None and vals[-1] is not None:
            line += f"{(vals[-1] - vals[0]):>+18.4f}"
        print(line)

    out = {"labels": args.labels, "paths": [str(p) for p in args.summaries], "table": table}
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out_file}")


if __name__ == "__main__":
    main()
