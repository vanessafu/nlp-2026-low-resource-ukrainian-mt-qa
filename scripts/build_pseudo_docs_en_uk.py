#!/usr/bin/env python3
"""Build paragraph-level EN→UK pseudo-documents via semantic clustering.

Takes sentence-level selected pairs (from filter_mt_en_uk.py) and clusters the
English side with paraphrase-multilingual-mpnet-base-v2 + MiniBatchKMeans.
Sentences inside each cluster are concatenated into longer training documents
at several chunk sizes, bridging the sentence→document length gap for MT.

Also builds short/medium pseudo-docs from official EN-UK dev sentences for
paragraph-level validation.

Outputs
-------
  data/pseudo_docs/train_pseudo_en_uk.jsonl
  data/pseudo_docs/dev_pseudo_en_uk.jsonl
  data/pseudo_docs/dev_pseudo_en_uk_short.jsonl
  data/pseudo_docs/dev_pseudo_en_uk_medium.jsonl

  # Stage-1 training default path (symlink/copy):
  train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl

Usage
-----
  python scripts/build_pseudo_docs_en_uk.py
  python scripts/build_pseudo_docs_en_uk.py --train-chunk-sizes 3 5 8 12
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

REPO = Path(__file__).resolve().parents[1]
SELECTED = REPO / "data" / "processed" / "selected_en_uk.jsonl"
PSEUDO_DIR = REPO / "data" / "pseudo_docs"
TRAIN_OUT = PSEUDO_DIR / "train_pseudo_en_uk.jsonl"
DEV_OUT = PSEUDO_DIR / "dev_pseudo_en_uk.jsonl"
DEV_SHORT = PSEUDO_DIR / "dev_pseudo_en_uk_short.jsonl"
DEV_MEDIUM = PSEUDO_DIR / "dev_pseudo_en_uk_medium.jsonl"
STAGE1_COPY = REPO / "train_data" / "data" / "pseudo_docs" / "train_pseudo_en_uk.jsonl"

DEV_EN = REPO / "llms-limited-resources2026" / "Ukrainian" / "MT" / "dev.en-uk.en"
DEV_UK = REPO / "llms-limited-resources2026" / "Ukrainian" / "MT" / "dev.en-uk.uk"
DEV_EN_ALT = REPO / "llms-limited-resources2025" / "Ukrainian" / "MT" / "dev.en-uk.en"
DEV_UK_ALT = REPO / "llms-limited-resources2025" / "Ukrainian" / "MT" / "dev.en-uk.uk"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def word_count(text: str) -> int:
    return len(text.split())


def read_jsonl_pairs(path: Path) -> tuple[list[str], list[str]]:
    src, tgt = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            s = (row.get("src") or row.get("en") or "").strip()
            t = (row.get("tgt") or row.get("uk") or "").strip()
            if s and t:
                src.append(s)
                tgt.append(t)
    return src, tgt


def read_parallel(src_path: Path, tgt_path: Path) -> tuple[list[str], list[str]]:
    ss = [l.strip() for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ts = [l.strip() for l in tgt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = min(len(ss), len(ts))
    return ss[:n], ts[:n]


def resolve_dev() -> tuple[Path, Path]:
    if DEV_EN.exists() and DEV_UK.exists():
        return DEV_EN, DEV_UK
    if DEV_EN_ALT.exists() and DEV_UK_ALT.exists():
        return DEV_EN_ALT, DEV_UK_ALT
    raise FileNotFoundError("Dev EN-UK files not found under llms-limited-resources2026/ or 2025/")


def cluster_and_concat(
    src_lines: list[str],
    tgt_lines: list[str],
    embedder: SentenceTransformer,
    chunk_size: int,
    batch_size: int = 256,
) -> list[dict]:
    n = len(src_lines)
    if n < max(2, chunk_size):
        return []
    n_clusters = max(1, n // chunk_size)
    print(f"  chunk_size={chunk_size} → n_clusters={n_clusters} over {n:,} sentences")

    emb = embedder.encode(
        src_lines,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=10_000,
        n_init=3,
        random_state=42,
    )
    labels = km.fit_predict(emb)

    buckets: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[int(lab)].append(i)

    docs: list[dict] = []
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs)
        src = " ".join(src_lines[i] for i in idxs)
        tgt = " ".join(tgt_lines[i] for i in idxs)
        docs.append(
            {
                "src": src,
                "tgt": tgt,
                "n_sentences": len(idxs),
                "word_count": word_count(src),
            }
        )
    return docs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  [{label}] empty")
        return
    wcs = [r["word_count"] for r in rows]
    print(
        f"  [{label}] n={len(rows):,}  "
        f"avg_words={np.mean(wcs):.1f}  min={min(wcs)}  max={max(wcs)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selected", type=Path, default=SELECTED)
    parser.add_argument("--out-dir", type=Path, default=PSEUDO_DIR)
    parser.add_argument("--train-chunk-sizes", type=int, nargs="+", default=[3, 5, 8, 12])
    parser.add_argument("--dev-chunk-sizes", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-stage1-copy", action="store_true")
    args = parser.parse_args()

    if not args.selected.exists():
        raise FileNotFoundError(
            f"Selected sentence pairs not found: {args.selected}\n"
            "Run: python scripts/filter_mt_en_uk.py"
        )

    print(f"Loading selected pairs from {args.selected}")
    src, tgt = read_jsonl_pairs(args.selected)
    print(f"  {len(src):,} sentence pairs")

    print(f"Loading embedder {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    train_docs: list[dict] = []
    for cs in args.train_chunk_sizes:
        docs = cluster_and_concat(src, tgt, embedder, cs, batch_size=args.batch_size)
        print(f"  → {len(docs):,} docs for chunk_size={cs}")
        train_docs.extend(docs)

    train_out = args.out_dir / "train_pseudo_en_uk.jsonl"
    write_jsonl(train_out, train_docs)
    print_stats("train_pseudo", train_docs)
    print(f"Wrote {train_out}")

    if not args.skip_stage1_copy:
        STAGE1_COPY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_out, STAGE1_COPY)
        print(f"Copied → {STAGE1_COPY} (Stage-1 default MT path)")

    if args.skip_dev:
        return

    dev_en_path, dev_uk_path = resolve_dev()
    print(f"Building dev pseudo-docs from {dev_en_path.name}")
    d_src, d_tgt = read_parallel(dev_en_path, dev_uk_path)
    print(f"  {len(d_src):,} dev sentences")

    dev_docs: list[dict] = []
    for cs in args.dev_chunk_sizes:
        docs = cluster_and_concat(d_src, d_tgt, embedder, cs, batch_size=args.batch_size)
        print(f"  → {len(docs):,} docs for chunk_size={cs}")
        dev_docs.extend(docs)

    short = [d for d in dev_docs if 50 <= d["word_count"] <= 250]
    medium = [d for d in dev_docs if 250 < d["word_count"] <= 800]

    write_jsonl(args.out_dir / "dev_pseudo_en_uk.jsonl", dev_docs)
    write_jsonl(args.out_dir / "dev_pseudo_en_uk_short.jsonl", short)
    write_jsonl(args.out_dir / "dev_pseudo_en_uk_medium.jsonl", medium)
    print_stats("dev_all", dev_docs)
    print_stats("dev_short", short)
    print_stats("dev_medium", medium)
    print("Done.")


if __name__ == "__main__":
    main()
