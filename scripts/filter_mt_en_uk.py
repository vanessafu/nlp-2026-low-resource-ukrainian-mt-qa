#!/usr/bin/env python3
"""Filter EN→UK MT pairs with basic length rules + SDKM similarity retrieval.

Pipeline
--------
1. Merge / load raw Moses files (train.eng / train.ukr).
2. Basic filter: drop pairs with <3 or >100 words, or length ratio outside [1/3, 3].
3. Embed Ukrainian sides of train + dev with Qwen2.5-3B-Instruct (mean-pooled
   last hidden state, max_length=128).
4. For each dev Ukrainian sentence, retrieve top-K most similar training sentences
   by cosine similarity (L2-normalized embeddings).
5. Deduplicate selected indices and write JSONL.

Outputs
-------
  data/processed/selected_en_uk.jsonl
  data/processed/embeddings_en_uk_train.npy  (optional cache)
  data/processed/embeddings_en_uk_dev.npy

Usage
-----
  python scripts/filter_mt_en_uk.py
  python scripts/filter_mt_en_uk.py --top-k 75 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
RAW_EN = REPO / "data" / "raw" / "eng-ukr" / "train.eng"
RAW_UK = REPO / "data" / "raw" / "eng-ukr" / "train.ukr"
DEV_EN = REPO / "llms-limited-resources2026" / "Ukrainian" / "MT" / "dev.en-uk.en"
DEV_UK = REPO / "llms-limited-resources2026" / "Ukrainian" / "MT" / "dev.en-uk.uk"
# Fallbacks for older shared-task layout
DEV_EN_ALT = REPO / "llms-limited-resources2025" / "Ukrainian" / "MT" / "dev.en-uk.en"
DEV_UK_ALT = REPO / "llms-limited-resources2025" / "Ukrainian" / "MT" / "dev.en-uk.uk"
OUT_DIR = REPO / "data" / "processed"
OUT_JSONL = OUT_DIR / "selected_en_uk.jsonl"

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def word_count(text: str) -> int:
    return len(text.split())


def basic_filter(src: str, tgt: str, min_words: int = 3, max_words: int = 100) -> bool:
    sw, tw = word_count(src), word_count(tgt)
    if sw < min_words or tw < min_words:
        return False
    if sw > max_words or tw > max_words:
        return False
    ratio = sw / max(tw, 1)
    if ratio > 3.0 or ratio < 1.0 / 3.0:
        return False
    return True


def read_parallel(src_path: Path, tgt_path: Path) -> list[tuple[str, str]]:
    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()
    n = min(len(src_lines), len(tgt_lines))
    return [(src_lines[i].strip(), tgt_lines[i].strip()) for i in range(n) if src_lines[i].strip() and tgt_lines[i].strip()]


def resolve_dev_paths() -> tuple[Path, Path]:
    if DEV_UK.exists() and DEV_EN.exists():
        return DEV_EN, DEV_UK
    if DEV_UK_ALT.exists() and DEV_EN_ALT.exists():
        return DEV_EN_ALT, DEV_UK_ALT
    raise FileNotFoundError(
        "Dev EN-UK files not found. Expected under llms-limited-resources2026/ "
        "or llms-limited-resources2025/ Ukrainian/MT/dev.en-uk.*"
    )


@torch.inference_mode()
def embed_texts(
    texts: list[str],
    model,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or next(model.parameters()).device
    vectors: list[np.ndarray] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for bi, start in enumerate(tqdm(range(0, len(texts), batch_size), total=n_batches, desc="embed")):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        vectors.append(pooled.float().cpu().numpy())
        if (bi + 1) % 1000 == 0:
            print(f"  embedded {min(start + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
    return np.concatenate(vectors, axis=0).astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def retrieve_top_k(
    train_emb: np.ndarray,
    query_emb: np.ndarray,
    top_k: int,
    chunk_size: int = 50_000,
) -> set[int]:
    """Cosine top-k via chunked matmul on L2-normalized embeddings."""
    train_n = l2_normalize(train_emb)
    query_n = l2_normalize(query_emb)
    selected: set[int] = set()
    n_train = train_n.shape[0]

    for qi, q in enumerate(tqdm(query_n, desc="retrieve")):
        best_scores = None
        best_idx = None
        for start in range(0, n_train, chunk_size):
            chunk = train_n[start : start + chunk_size]
            scores = chunk @ q  # cosine
            if scores.shape[0] <= top_k:
                idx_local = np.argsort(-scores)
            else:
                idx_local = np.argpartition(-scores, top_k)[:top_k]
                idx_local = idx_local[np.argsort(-scores[idx_local])]
            scores_local = scores[idx_local]
            idx_global = idx_local + start
            if best_scores is None:
                best_scores, best_idx = scores_local, idx_global
            else:
                scores_cat = np.concatenate([best_scores, scores_local])
                idx_cat = np.concatenate([best_idx, idx_global])
                keep = np.argsort(-scores_cat)[:top_k]
                best_scores, best_idx = scores_cat[keep], idx_cat[keep]
        selected.update(int(i) for i in best_idx)
        if (qi + 1) % 500 == 0:
            print(f"  queries {qi + 1:,}/{len(query_n):,} | selected so far {len(selected):,}", flush=True)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=RAW_EN)
    parser.add_argument("--tgt", type=Path, default=RAW_UK)
    parser.add_argument("--dev-src", type=Path, default=None)
    parser.add_argument("--dev-tgt", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Train embedding chunk for retrieval")
    parser.add_argument("--cache-embeddings", action="store_true", default=True)
    parser.add_argument("--no-cache-embeddings", action="store_false", dest="cache_embeddings")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.src.exists() or not args.tgt.exists():
        raise FileNotFoundError(
            f"Raw parallel files not found:\n  {args.src}\n  {args.tgt}\n"
            "Run: python scripts/download_mt_en_uk.py"
        )

    if args.dev_src is None or args.dev_tgt is None:
        args.dev_src, args.dev_tgt = resolve_dev_paths()

    print("Loading raw pairs …")
    raw_pairs = read_parallel(args.src, args.tgt)
    print(f"  raw: {len(raw_pairs):,}")

    filtered = [(s, t) for s, t in raw_pairs if basic_filter(s, t)]
    print(f"  after basic filter: {len(filtered):,}")
    if not filtered:
        raise SystemExit("No pairs left after basic filtering.")

    print("Loading dev Ukrainian sentences …")
    # Retrieval uses the Ukrainian side of dev only (SDKM-style domain matching).
    dev_pairs = read_parallel(args.dev_src, args.dev_tgt)
    dev_uk = [t for _, t in dev_pairs]
    print(f"  dev uk sentences: {len(dev_uk):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_emb_path = OUT_DIR / "embeddings_en_uk_train.npy"
    dev_emb_path = OUT_DIR / "embeddings_en_uk_dev.npy"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading embedder {args.model} on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    train_uk = [t for _, t in filtered]
    if args.cache_embeddings and train_emb_path.exists() and train_emb_path.stat().st_size > 0:
        print(f"Loading cached train embeddings: {train_emb_path}")
        train_emb = np.load(train_emb_path)
        if train_emb.shape[0] != len(train_uk):
            print("  cache size mismatch — recomputing")
            train_emb = embed_texts(train_uk, model, tokenizer, args.batch_size, args.max_length, device)
            np.save(train_emb_path, train_emb)
    else:
        train_emb = embed_texts(train_uk, model, tokenizer, args.batch_size, args.max_length, device)
        if args.cache_embeddings:
            np.save(train_emb_path, train_emb)
            print(f"Saved {train_emb_path}")

    if args.cache_embeddings and dev_emb_path.exists() and dev_emb_path.stat().st_size > 0:
        print(f"Loading cached dev embeddings: {dev_emb_path}")
        dev_emb = np.load(dev_emb_path)
        if dev_emb.shape[0] != len(dev_uk):
            print("  cache size mismatch — recomputing")
            dev_emb = embed_texts(dev_uk, model, tokenizer, args.batch_size, args.max_length, device)
            np.save(dev_emb_path, dev_emb)
    else:
        dev_emb = embed_texts(dev_uk, model, tokenizer, args.batch_size, args.max_length, device)
        if args.cache_embeddings:
            np.save(dev_emb_path, dev_emb)
            print(f"Saved {dev_emb_path}")

    print(f"Retrieving top-{args.top_k} per dev sentence …")
    selected = retrieve_top_k(train_emb, dev_emb, top_k=args.top_k, chunk_size=args.chunk_size)
    print(f"Unique selected indices: {len(selected):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in sorted(selected):
            src, tgt = filtered[i]
            row = {
                "src": src,
                "tgt": tgt,
                "word_count": word_count(src),
                "n_sentences": 1,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected):,} pairs → {args.out}")


if __name__ == "__main__":
    main()
