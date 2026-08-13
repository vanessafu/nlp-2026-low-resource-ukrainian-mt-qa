"""
Now only cs-uk MT training/validation corpus.

Usage
-----
  python scripts/loader.py <corpus> [options]
  python scripts/loader.py eubookshop_cs_uk --no-align
  python scripts/loader.py nllb --mode clean

Corpora: eubookshop_cs_uk, elrc_cs_uk, wikimedia_cs_uk, nllb

Every build_*() function downloads its own raw source, cleans it through
cleaning_pipeline.py, and writes the result under data/training/MT/cs-uk/ or
data/validation/MT/cz-uk/.
"""
from __future__ import annotations

import gzip
import io
import random
import zipfile
from pathlib import Path
from typing import Optional

import polars as pl
import requests
from datasets import load_dataset

from cleaning_pipeline import (
    FilterPipeline,
    PipelineConfig,
    compute_structural_score,
    compute_uk_quality_score,
    run_stages,
    _DEFAULT_THRESHOLDS,
)

SEED = 42
random.seed(SEED)

DATA_DIR  = Path(__file__).parent.parent / "data"
OUT_TRAIN = DATA_DIR / "training"   / "MT"
OUT_VAL   = DATA_DIR / "validation" / "MT"
RAW_DIR   = DATA_DIR / "raw"

def download(url: str) -> bytes:
    print(f"  Downloading {url} …")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    raw = b"".join(resp.iter_content(chunk_size=1 << 20))
    print(f"  Downloaded {len(raw) / 1e6:.1f} MB")
    return raw


def strip_lineno(line: str) -> str:
    parts = line.split("\t", 1)
    if len(parts) == 2 and parts[0].strip().isdigit():
        return parts[1]
    return line


def parse_moses_zip(raw: bytes, src_lang: str, tgt_lang: str = "uk") -> tuple[list[str], list[str], Optional[list[str]]]:
    """OPUS Moses zip → (src_lines, tgt_lines, ids_lines or None)."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names    = z.namelist()
        src_file = next((n for n in names if n.endswith(f".{src_lang}")), None)
        tgt_file = next((n for n in names if n.endswith(f".{tgt_lang}")), None)
        ids_file = next((n for n in names if n.endswith(".ids")), None)
        if src_file is None or tgt_file is None:
            raise ValueError(f"Expected .{src_lang} and .{tgt_lang} files in the ZIP.\nFound: {names}")

        def read_text(name: str) -> list[str]:
            text = z.read(name).decode("utf-8")
            return [strip_lineno(l) for l in text.splitlines() if l.strip()]

        src_lines = read_text(src_file)
        tgt_lines = read_text(tgt_file)
        ids_lines = None
        if ids_file:
            ids_lines = [l for l in z.read(ids_file).decode("utf-8").splitlines() if l.strip()]

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Line count mismatch: {src_lang}={len(src_lines):,}  {tgt_lang}={len(tgt_lines):,}")
    if ids_lines is not None and len(ids_lines) != len(src_lines):
        print(f"  [warn] IDs length ({len(ids_lines):,}) != sentence count ({len(src_lines):,}) — ignoring ids.")
        ids_lines = None
    return src_lines, tgt_lines, ids_lines


def doc_key(ids_line: str) -> str:
    parts = ids_line.split("\t", 2)
    return f"{parts[0]}\t{parts[1]}" if len(parts) >= 2 else ids_line


def build_paragraphs(
    src_lines: list[str], tgt_lines: list[str], ids_lines: Optional[list[str]],
    min_sents: int, max_sents: int, min_chars: int, max_chars: int,
) -> list[tuple[str, str]]:
    """
    Group sentences into paragraphs. Uses document boundaries from ids_lines
    when available, otherwise groups consecutive sentences into random-sized
    pseudo-documents. Either way, each document is then chunked into random
    MIN_SENTS-MAX_SENTS windows and filtered by char length.
    """
    n = len(src_lines)
    docs: list[list[int]] = []

    if ids_lines is not None:
        cur_key: Optional[str] = None
        cur_run: list[int] = []
        for i, ids_line in enumerate(ids_lines):
            key = doc_key(ids_line)
            if key != cur_key:
                if cur_run:
                    docs.append(cur_run)
                cur_key, cur_run = key, [i]
            else:
                cur_run.append(i)
        if cur_run:
            docs.append(cur_run)
    else:
        pos = 0
        while pos < n:
            size = random.randint(min_sents, max_sents)
            docs.append(list(range(pos, min(pos + size, n))))
            pos += size

    paragraphs: list[tuple[str, str]] = []
    for indices in docs:
        pos = 0
        while pos < len(indices):
            size  = random.randint(min_sents, max_sents)
            chunk = indices[pos: pos + size]
            pos  += len(chunk)
            if len(chunk) < min_sents:
                break
            src = " ".join(src_lines[i] for i in chunk).strip()
            tgt = " ".join(tgt_lines[i] for i in chunk).strip()
            if min_chars <= len(src) <= max_chars and min_chars <= len(tgt) <= max_chars:
                paragraphs.append((src, tgt))
    return paragraphs


def describe_pairs(src_lines: list[str], tgt_lines: list[str], src_label: str, tgt_label: str) -> None:
    n = len(src_lines)
    print(f"\n── Raw corpus analysis ({n:,} pairs) ───────────────────────────────")
    for label, lines in [(src_label, src_lines), (tgt_label, tgt_lines)]:
        chars = sorted(len(l) for l in lines)
        words = sorted(len(l.split()) for l in lines)
        pct = lambda lst, p: lst[max(0, int(len(lst) * p) - 1)]
        print(f"\n  {label}:")
        print(f"    chars  min={min(chars):5}  p25={pct(chars,0.25):5}  p50={pct(chars,0.50):5}  "
              f"p75={pct(chars,0.75):5}  p95={pct(chars,0.95):5}  max={max(chars):6}")
        print(f"    words  min={min(words):5}  p25={pct(words,0.25):5}  p50={pct(words,0.50):5}  "
              f"p75={pct(words,0.75):5}  p95={pct(words,0.95):5}  max={max(words):6}")
    for i in random.sample(range(n), min(4, n)):
        print(f"\n    src: {src_lines[i][:100]!r}\n    tgt: {tgt_lines[i][:100]!r}")
    print("\n" + "─" * 70)


def paragraph_quality_score(
    src: str, tgt: str, lang_pair: str, len_saturate: float,
    w_len: float = 0.55, w_struct: float = 0.25, w_uk: float = 0.20,
) -> float:
    avg_chars    = (len(src) + len(tgt)) / 2
    len_score    = min(1.0, avg_chars / len_saturate)
    struct_score = compute_structural_score(src, tgt, _DEFAULT_THRESHOLDS[lang_pair])[2]
    uk_score     = compute_uk_quality_score(tgt)
    return w_len * len_score + w_struct * struct_score + w_uk * uk_score


def build_pipeline(
    lang_pair: str, stages: list[int], *,
    near_dedup: bool = True, near_threshold: float = 0.90,
    embed_model: str = "intfloat/multilingual-e5-small", embed_sim_min: float = 0.85, embed_batch: int = 512,
) -> FilterPipeline:
    cfg = PipelineConfig(
        lang_pair=lang_pair, stages=stages,
        dedup_near=near_dedup, minhash_threshold=near_threshold,
        embed_model=embed_model, embed_sim_min=embed_sim_min, embed_batch=embed_batch,
    )
    pipe = FilterPipeline(cfg)
    pipe.setup()
    return pipe


def clean_pairs_in_chunks(pairs: list[tuple[str, str]], pipe: FilterPipeline, flush_size: int) -> pl.DataFrame:
    """Run a persistent pipeline over a list of (src, tgt) pairs in chunks, concatenate the clean rows."""
    chunks: list[pl.DataFrame] = []
    n_clean = 0
    for start in range(0, len(pairs), flush_size):
        batch = pairs[start: start + flush_size]
        df = pl.DataFrame({"source": [s for s, _ in batch], "target": [t for _, t in batch]})
        clean, _ = pipe.run_df(df)
        if len(clean):
            chunks.append(clean.select(["source", "target"]))
            n_clean += len(clean)
        print(f"  [flush] processed={start + len(batch):,}  clean={n_clean:,}", flush=True)
    return pl.concat(chunks) if chunks else pl.DataFrame({"source": [], "target": []})


def split_by_quality(clean: pl.DataFrame, max_train: int, n_val: int, score_fn) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rank by score_fn(src, tgt), take the best n_val rows as validation, cap the rest at max_train."""
    n_clean = len(clean)
    actual_val = max(50, min(n_val, int(n_clean * 0.05)))
    if actual_val >= n_clean:
        actual_val = max(1, n_clean // 10)
        print(f"  [warn] Small corpus — reducing validation to {actual_val}")

    clean = clean.with_columns(
        pl.struct(["source", "target"]).map_elements(
            lambda r: score_fn(r["source"], r["target"]), return_dtype=pl.Float64,
        ).alias("_score")
    ).sort("_score", descending=True)

    val_df   = clean.head(actual_val).select(["source", "target"])
    train_df = clean.tail(n_clean - actual_val).select(["source", "target"]).head(max_train)
    print(f"  Validation: {len(val_df):,} pairs   Training: {len(train_df):,} pairs")
    return train_df, val_df


def rank_and_cap(df: pl.DataFrame, max_n: int, score_fn) -> pl.DataFrame:
    if len(df) <= max_n:
        return df.select(["source", "target"])
    print(f"  Output {len(df):,} exceeds cap {max_n:,} — ranking by quality …")
    df = df.with_columns(
        pl.struct(["source", "target"]).map_elements(
            lambda r: score_fn(r["source"], r["target"]), return_dtype=pl.Float64,
        ).alias("_score")
    ).sort("_score", descending=True).head(max_n)
    return df.select(["source", "target"])


def save_tsv(df: pl.DataFrame, path: Path) -> None:
    df = df.filter(
        pl.col("source").is_not_null() & pl.col("target").is_not_null()
        & (pl.col("source").str.strip_chars() != "") & (pl.col("target").str.strip_chars() != "")
    ).select(["source", "target"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source\ttarget\n")
        for src, tgt in df.iter_rows():
            f.write(f"{' '.join(str(src).split())}\t{' '.join(str(tgt).split())}\n")
    print(f"  saved → {path}  ({len(df):,} pairs)")


def build_eubookshop_cs_uk(
    out_dir: Optional[Path] = None, val_dir: Optional[Path] = None,
    max_train: int = 100_000, n_val: int = 150, no_align: bool = False,
) -> None:
    train_path = (out_dir or OUT_TRAIN / "cs-uk") / "eubookshop.tsv"
    val_path   = (val_dir or OUT_VAL   / "cz-uk") / "eubookshop.tsv"

    cs_lines, uk_lines, ids_lines = parse_moses_zip(
        download("https://object.pouta.csc.fi/OPUS-EUbookshop/v2/moses/cs-uk.txt.zip"), "cs", "uk",
    )
    print(f"  Parsed: {len(cs_lines):,} sentence pairs  "
          f"(ids: {'found' if ids_lines else 'not found, using consecutive grouping'})")
    describe_pairs(cs_lines, uk_lines, "Source (cs)", "Target (uk)")

    print("\n── Building paragraphs (3–15 sents, 80–4000 chars) ──")
    paragraphs = build_paragraphs(cs_lines, uk_lines, ids_lines, 3, 15, 80, 4_000)
    print(f"  Paragraph pairs (pre-clean): {len(paragraphs):,}")

    stages = [1, 2, 3, 5, 6, 7] + ([] if no_align else [8])
    pipe   = build_pipeline("cs-uk", stages, near_threshold=0.90, embed_sim_min=0.78)
    clean  = clean_pairs_in_chunks(paragraphs, pipe, flush_size=10_000)
    print(f"\n  Post-pipeline: {len(clean):,} pairs")

    score_fn = lambda s, t: paragraph_quality_score(s, t, "cs-uk", 600, 0.55, 0.25, 0.20)
    train_df, val_df = split_by_quality(clean, max_train, n_val, score_fn)
    save_tsv(val_df, val_path)
    save_tsv(train_df, train_path)


# =============================================================================
# CORPUS — ELRC-5179-acts cs-uk (legal text, paragraph-level train + val)
# =============================================================================

def build_elrc_cs_uk(
    out_dir: Optional[Path] = None, val_dir: Optional[Path] = None,
    max_train: int = 70_000, n_val: int = 100, no_align: bool = False,
) -> None:
    train_path = (out_dir or OUT_TRAIN / "cs-uk") / "elrc_acts.tsv"
    val_path   = (val_dir or OUT_VAL   / "cz-uk") / "elrc_acts.tsv"

    cs_lines, uk_lines, ids_lines = parse_moses_zip(
        download("https://object.pouta.csc.fi/OPUS-ELRC-5179-acts_Ukrainian/v1/moses/cs-uk.txt.zip"), "cs", "uk",
    )
    print(f"  Parsed: {len(cs_lines):,} sentence pairs  "
          f"(ids: {'found' if ids_lines else 'not found, using consecutive grouping'})")
    describe_pairs(cs_lines, uk_lines, "Source (cs)", "Target (uk)")

    print("\n── Building paragraphs (2–12 sents, 80–5000 chars) ──")
    paragraphs = build_paragraphs(cs_lines, uk_lines, ids_lines, 2, 12, 80, 5_000)
    print(f"  Paragraph pairs (pre-clean): {len(paragraphs):,}")
    if not paragraphs:
        print("  [warn] No paragraphs built — check length settings.")
        return

    stages = [1, 2, 3, 5, 6, 7, 9] + ([] if no_align else [8])
    pipe   = build_pipeline("cs-uk", stages, near_threshold=0.90, embed_sim_min=0.89)
    clean  = clean_pairs_in_chunks(paragraphs, pipe, flush_size=5_000)
    print(f"\n  Post-pipeline: {len(clean):,} pairs")
    if not len(clean):
        print("  [warn] No clean pairs produced — check pipeline thresholds.")
        return

    score_fn = lambda s, t: paragraph_quality_score(s, t, "cs-uk", 700, 0.50, 0.30, 0.20)
    train_df, val_df = split_by_quality(clean, max_train, n_val, score_fn)
    save_tsv(val_df, val_path)
    save_tsv(train_df, train_path)

def build_wikimedia_cs_uk(out_dir: Optional[Path] = None, max_pairs: int = 200_000, no_align: bool = False) -> None:
    out_path = (out_dir or OUT_TRAIN / "cs-uk") / "wikimedia.tsv"

    cs_lines, uk_lines, _ = parse_moses_zip(
        download("https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/moses/cs-uk.txt.zip"), "cs", "uk",
    )
    print(f"  Parsed: {len(cs_lines):,} raw sentence pairs")
    describe_pairs(cs_lines, uk_lines, "Source (cs)", "Target (uk)")

    df = pl.DataFrame({"source": cs_lines, "target": uk_lines})
    n_in = len(df)
    df = df.filter(
        pl.col("source").is_not_null() & pl.col("target").is_not_null()
        & (pl.col("source").str.strip_chars() != "") & (pl.col("target").str.strip_chars() != "")
        & (pl.col("source").str.len_chars() >= 30) & (pl.col("target").str.len_chars() >= 30)
        & (pl.col("source").str.split(" ").list.len() >= 5) & (pl.col("target").str.split(" ").list.len() >= 5)
    )
    print(f"  Pre-filter (≥5 words, ≥30 chars/side): {n_in:,} → {len(df):,}")

    stages = [1, 2, 3, 4, 5, 6, 7] + ([] if no_align else [8])
    clean = run_stages(
        df, "cs-uk", stages=stages, dataset="wikimedia_cs_uk",
        dedup_near=True, minhash_threshold=0.93,
        embed_model="intfloat/multilingual-e5-small", embed_sim_min=0.87, embed_batch=512,
    )
    print(f"\n  Post-pipeline: {len(clean):,} pairs")

    score_fn = lambda s, t: paragraph_quality_score(s, t, "cs-uk", 400, 0.55, 0.25, 0.20)
    save_tsv(rank_and_cap(clean, max_pairs, score_fn), out_path)

NLLB_HF_CONFIG        = "ces_Latn-ukr_Cyrl"
NLLB_RAW_LASER_MIN    = 1.17
NLLB_LASER_MIN        = 1.17
NLLB_RAW_LID_SRC_MIN  = 0.95
NLLB_RAW_LID_TGT_MIN  = 0.98
NLLB_LID_SRC_MIN, NLLB_LID_SRC_MAX = 0.97, 1.01
NLLB_LID_TGT_MIN, NLLB_LID_TGT_MAX = 0.99, 1.01
NLLB_RAW_COLS         = ["source", "target", "laser_score", "source_sentence_lid", "target_sentence_lid"]
NLLB_STAGES_ROW       = [1, 3, 4, 5, 6, 7, 8]   # pass 1 (chunked): per-row filters
NLLB_STAGES_DEDUP     = [2]                      # pass 2 (global): exact + near-dedup
NLLB_DEFAULT_CHUNK    = 500_000


def _unpack_translation(series: pl.Series) -> pl.DataFrame:
    if hasattr(series.dtype, "fields"):
        fields = series.struct.fields
        return pl.DataFrame({"source": series.struct.field(fields[0]), "target": series.struct.field(fields[1])})
    sources, targets = [], []
    for row in series.to_list():
        if isinstance(row, dict):
            vals = list(row.values())
            sources.append(vals[0] if vals else None)
            targets.append(vals[1] if len(vals) > 1 else None)
        else:
            sources.append(None)
            targets.append(None)
    return pl.DataFrame({"source": sources, "target": targets})


def nllb_download_to_raw(raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Loading allenai/nllb {NLLB_HF_CONFIG} …")
    ds    = load_dataset("allenai/nllb", NLLB_HF_CONFIG, verification_mode="no_checks", trust_remote_code=True)
    split = "train" if "train" in ds else next(iter(ds))
    df    = pl.from_pandas(ds[split].to_pandas())
    print(f"  {len(df):,} total rows")

    filt = pl.col("laser_score").cast(pl.Float64, strict=False) >= NLLB_RAW_LASER_MIN
    if "source_sentence_lid" in df.columns:
        filt = filt & (pl.col("source_sentence_lid").cast(pl.Float64, strict=False) >= NLLB_RAW_LID_SRC_MIN)
    if "target_sentence_lid" in df.columns:
        filt = filt & (pl.col("target_sentence_lid").cast(pl.Float64, strict=False) >= NLLB_RAW_LID_TGT_MIN)
    df = df.filter(filt)
    print(f"  → {len(df):,} after loose filter (laser ≥ {NLLB_RAW_LASER_MIN})")

    text  = _unpack_translation(df["translation"])
    extra = {c: df[c] for c in NLLB_RAW_COLS[2:] if c in df.columns}
    raw = pl.DataFrame({"source": text["source"], "target": text["target"], **extra}).filter(
        pl.col("source").is_not_null() & pl.col("target").is_not_null()
        & (pl.col("source").str.strip_chars() != "") & (pl.col("target").str.strip_chars() != "")
    )
    with gzip.open(raw_path, "wt", encoding="utf-8") as fh:
        fh.write(raw.write_csv(separator="\t"))
    print(f"  {len(raw):,} rows saved → {raw_path}")


def nllb_filter_from_raw(raw_path: Path) -> pl.DataFrame:
    print(f"  Reading {raw_path} …")
    df = pl.read_csv(raw_path, separator="\t", quote_char=None, ignore_errors=True)
    print(f"  {len(df):,} rows in raw")

    df = df.filter(pl.col("laser_score").cast(pl.Float64, strict=False) >= NLLB_LASER_MIN)
    print(f"  → {len(df):,} after laser_score ≥ {NLLB_LASER_MIN}")
    if "source_sentence_lid" in df.columns:
        df = df.filter(pl.col("source_sentence_lid").cast(pl.Float64, strict=False)
                       .is_between(NLLB_LID_SRC_MIN, NLLB_LID_SRC_MAX))
    if "target_sentence_lid" in df.columns:
        df = df.filter(pl.col("target_sentence_lid").cast(pl.Float64, strict=False)
                       .is_between(NLLB_LID_TGT_MIN, NLLB_LID_TGT_MAX))
    print(f"  → {len(df):,} after LID bounds")

    return df.select(["source", "target"]).filter(
        pl.col("source").is_not_null() & pl.col("target").is_not_null()
        & (pl.col("source").str.strip_chars() != "") & (pl.col("target").str.strip_chars() != "")
    )


def build_nllb(
    out_dir: Optional[Path] = None, raw_dir: Optional[Path] = None,
    mode: str = "all", clean_chunk: int = NLLB_DEFAULT_CHUNK,
) -> None:
    raw_path = (raw_dir or RAW_DIR / "nllb") / "nllb_cs-uk_raw.tsv.gz"
    out_path = (out_dir or OUT_TRAIN / "cs-uk") / "nllb.tsv"

    if mode in ("download", "all", "download+clean"):
        print("\n=== download → raw ===")
        nllb_download_to_raw(raw_path)

    if mode in ("clean", "all", "download+clean"):
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw file not found: {raw_path}\n"
                "Run first:  python scripts/loader.py nllb --mode download"
            )
        print("\n=== filter from raw ===")
        filtered = nllb_filter_from_raw(raw_path)
        total    = len(filtered)

        print(f"\n=== clean → training TSV  ({total:,} pairs) ===")
        pipe  = build_pipeline("cs-uk", NLLB_STAGES_ROW)
        pairs = list(zip(filtered["source"].to_list(), filtered["target"].to_list()))
        chunk = clean_chunk if 0 < clean_chunk < total else max(total, 1)
        after_pass1 = clean_pairs_in_chunks(pairs, pipe, chunk)
        print(f"  after pass 1: {total:,} → {len(after_pass1):,}")

        print(f"  pass 2 (global dedup): rows={len(after_pass1):,}")
        final = run_stages(after_pass1, "cs-uk", stages=NLLB_STAGES_DEDUP, dataset="nllb_cs-uk_dedup")
        save_tsv(final, out_path)

CORPUS_BUILDERS = {
    "eubookshop_cs_uk": build_eubookshop_cs_uk,
    "elrc_cs_uk":        build_elrc_cs_uk,
    "wikimedia_cs_uk":   build_wikimedia_cs_uk,
    "nllb":              build_nllb,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a cs-uk MT training/validation corpus (single entry point).")
    parser.add_argument("corpus", choices=sorted(CORPUS_BUILDERS))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--val-dir", type=Path, help="eubookshop_cs_uk / elrc_cs_uk")
    parser.add_argument("--max-train", type=int, help="eubookshop_cs_uk / elrc_cs_uk")
    parser.add_argument("--n-val", type=int, help="eubookshop_cs_uk / elrc_cs_uk")
    parser.add_argument("--max-pairs", type=int, help="wikimedia_cs_uk")
    parser.add_argument("--no-align", action="store_true", help="skip the semantic-alignment stage")
    parser.add_argument("--mode", help="nllb: download/clean/download+clean/all")
    parser.add_argument("--raw-dir", type=Path, help="nllb: raw cache directory")
    parser.add_argument("--clean-chunk", type=int, help="nllb: rows per pass-1 chunk (0 = no chunking)")
    args = parser.parse_args()

    kwargs = {}
    if args.out_dir is not None:    kwargs["out_dir"] = args.out_dir
    if args.val_dir is not None:    kwargs["val_dir"] = args.val_dir
    if args.max_train is not None:  kwargs["max_train"] = args.max_train
    if args.n_val is not None:      kwargs["n_val"] = args.n_val
    if args.max_pairs is not None:  kwargs["max_pairs"] = args.max_pairs
    if args.no_align:               kwargs["no_align"] = True
    if args.mode is not None:       kwargs["mode"] = args.mode
    if args.raw_dir is not None:    kwargs["raw_dir"] = args.raw_dir
    if args.clean_chunk is not None: kwargs["clean_chunk"] = args.clean_chunk

    CORPUS_BUILDERS[args.corpus](**kwargs)
