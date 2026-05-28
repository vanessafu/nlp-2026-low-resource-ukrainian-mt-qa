"""
Download and filter parallel corpora for en-uk / cs-uk MT training.

Each filter function:
  1. Applies dataset-specific column-based quality filters
  2. Calls run_stages() from cleaning_pipeline for generic cleaning

Datasets with strong quality columns (MaCoCu, NLLB) use minimal pipeline
stages (S1 unicode + S2 dedup) since column filters already ensure quality.

Datasets without quality columns (OpenSubtitles, TildeMODEL, ParaCrawl,
DocHPLT) run the full lightweight pipeline (S1–S2 + S5–S7).

Usage
-----
  python scripts/loader.py          # build all training + validation sets
"""
from __future__ import annotations

import gzip
import io
import random
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import zipfile
from datasets import load_dataset

try:
    import polars as pl
    _POLARS = True
except ImportError:
    import pandas as pl   # type: ignore[no-redef]
    _POLARS = False

# =============================================================================
# CONFIG
# =============================================================================

SEED = 42
random.seed(SEED)

SENTENCE_RATIO       = 0.70   # target fraction of sentence-level samples in WMT24PP mix
SENTENCE_SAMPLE_RATE = 0.65   # fraction of eligible sentence rows to keep before mixing

PARAGRAPH_CFG: dict[str, tuple[int, int]] = {
    "news":     (6, 3),
    "social":   (3, 2),
    "speech":   (8, 4),
    "literary": (5, 2),
}

DATA_DIR  = Path(__file__).parent.parent / "data"
OUT_TRAIN = DATA_DIR / "training"  / "MT"
OUT_VAL   = DATA_DIR / "validation" / "MT"
RAW_DIR   = DATA_DIR / "raw"


# =============================================================================
# 1.  DOWNLOAD / LOAD RAW DATA
# =============================================================================

def load_huggingface(
    dataset_name: str,
    lang_pair: str,
    split: Optional[str] = None,
) -> "pl.DataFrame":
    """
    Load a HuggingFace dataset as a Polars DataFrame.
    split=None → uses 'train' if available, else first split.
    """
    ds = load_dataset(dataset_name, lang_pair)
    if split is None:
        split = "train" if "train" in ds else next(iter(ds))
    print(f"  [{dataset_name}/{lang_pair}]  split={split!r}  rows={len(ds[split]):,}")
    return pl.from_pandas(ds[split].to_pandas())


def download_macocu(lang_pair: str = "uk-en", out_dir: Optional[Path] = None) -> Path:
    """
    Download MaCoCu from CLARIN using curl (NOT OPUS).
    Returns local .gz path; skips download if already cached.
    """
    if out_dir is None:
        out_dir = RAW_DIR / "macocu"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"MaCoCu-{lang_pair}.doc.txt.gz"
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [cache] {out_path}  ({out_path.stat().st_size // 1_000_000} MB)")
        return out_path

    url = (
        "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1858"
        f"/{filename}"
    )
    print(f"  Downloading {url} …")
    subprocess.run(
        ["curl", "--fail", "--location", "--progress-bar", "--output", str(out_path), url],
        check=True,
    )
    print(f"  → {out_path.stat().st_size // 1_000_000} MB saved.")
    return out_path


def load_macocu_gz(gz_path: Path) -> "pl.DataFrame":
    """Read MaCoCu .doc.txt.gz (gzipped TSV with header) into a DataFrame."""
    print(f"  Reading {gz_path.name} …")
    with gzip.open(gz_path, "rb") as fh:
        content: bytes = fh.read()
    df = pl.read_csv(
        content,
        separator="\t",
        quote_char=None,
        infer_schema_length=20_000,
        ignore_errors=True,
    )
    print(f"  → {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_opus_moses(url: str, src_lang: str, tgt_lang: str = "uk") -> "pl.DataFrame":
    """
    Download an OPUS Moses-format ZIP; return DataFrame with source/target columns.
    Detects aligned text files by extension (.{src_lang} / .{tgt_lang}).
    """
    print(f"  Downloading {url} …")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    raw = b"".join(resp.iter_content(chunk_size=1 << 20))

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names    = z.namelist()
        src_file = next((n for n in names if n.endswith(f".{src_lang}")), None)
        tgt_file = next((n for n in names if n.endswith(f".{tgt_lang}")), None)
        if src_file is None or tgt_file is None:
            raise ValueError(
                f"Expected .{src_lang} and .{tgt_lang} files inside {url}.\n"
                f"Found: {names}"
            )
        src_lines = z.read(src_file).decode("utf-8").splitlines()
        tgt_lines = z.read(tgt_file).decode("utf-8").splitlines()

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Line count mismatch: {src_file}={len(src_lines):,}, "
            f"{tgt_file}={len(tgt_lines):,}"
        )
    print(f"  → {len(src_lines):,} raw line pairs from zip")
    return pl.DataFrame({"source": src_lines, "target": tgt_lines})


# =============================================================================
# 2.  DATASET-SPECIFIC FILTERS
#
# Each function:
#   a) Applies column-based quality filters unique to that dataset.
#   b) Calls run_stages() from cleaning_pipeline for generic cleaning.
#
# High-quality corpora (MaCoCu, NLLB): column filters are strong enough;
#   only run S1 (unicode) + S2 (dedup).
#
# Low-quality / no-quality-columns (OPUS Moses, HF without scores):
#   run S1 + S2 + S5 (structural) + S6 (entity) + S7 (copy-source).
# =============================================================================

def filtered_macocu(
    df: "pl.DataFrame",
    stats_dir: Optional[Path] = None,
) -> "pl.DataFrame":
    """
    MaCoCu uk-en column filters (bicleaner_ai / bifixer / bleualign / boilerplate).

    Column mapping:  src_text = Ukrainian → 'target',  trg_text = English → 'source'.
    High-quality corpus: only S1 unicode + S2 dedup from pipeline.
    """
    from cleaning_pipeline import run_stages

    _BOILERPLATE_OK = ["good", "neargood", "good-good"]
    df = df.with_columns(
        pl.col("bicleaner_ai_score").cast(pl.Float64, strict=False),
        pl.col("bifixer_score").cast(pl.Float64, strict=False),
        pl.col("bleualign_score").cast(pl.Float64, strict=False),
    ).filter(
        (pl.col("bicleaner_ai_score") >= 0.86)
        & (pl.col("bifixer_score")      >= 301.2)
        & (pl.col("bleualign_score")    >= 0.60)
        & pl.col("src_boilerplate").str.strip_chars().is_in(_BOILERPLATE_OK)
        & pl.col("trg_boilerplate").str.strip_chars().is_in(_BOILERPLATE_OK)
        & (pl.col("biroamer_entities_detected").str.strip_chars() == "No")
        & pl.col("src_text").is_not_null()
        & pl.col("trg_text").is_not_null()
        & (pl.col("src_text").str.strip_chars() != "")
        & (pl.col("trg_text").str.strip_chars() != "")
    )
    df = df.select(
        pl.col("trg_text").alias("source"),
        pl.col("src_text").alias("target"),
    )
    print(f"  → {len(df):,} after column filters")
    # Column filters already ensure high quality; add only copy-source check (S7)
    return run_stages(df, "en-uk", stages=[1, 2, 7], dataset="macocu", stats_dir=stats_dir)


def filter_nllb(
    df: "pl.DataFrame",
    lang_pair: str = "en-uk",
    stats_dir: Optional[Path] = None,
) -> "pl.DataFrame":
    """
    NLLB column filters (source/target LID + LASER score).

    High-quality corpus: only S1 unicode + S2 dedup from pipeline.
    lang_pair must be 'en-uk' or 'cs-uk' (not the HuggingFace lang code).
    """
    from cleaning_pipeline import run_stages

    df = df.filter(
        pl.col("source_sentence_lid").is_between(0.95, 1.01)
        & pl.col("target_sentence_lid").is_between(0.98, 1.01)
        & (pl.col("laser_score") >= 1.10)
    )
    df = _unpack_translation(df["translation"])
    print(f"  → {len(df):,} after LID + LASER filters")
    # LASER pre-filters alignment; add UK quality + structural + copy-source
    return run_stages(df, lang_pair, stages=[1, 2, 4, 5, 7], dataset=f"nllb_{lang_pair}",
                      stats_dir=stats_dir)


def filter_wmt24pp(df: "pl.DataFrame") -> "pl.DataFrame":
    """
    WMT24PP column filters (removes canary sentences and bad-source rows).
    Returns document_id / segment_id / domain columns needed by build_and_save_wmt24pp.
    No pipeline: validation data should not be deduplicated or structurally filtered.
    """
    return (
        df.filter(
            (pl.col("domain") != "canary")
            & (~pl.col("is_bad_source"))
            & pl.col("source").is_not_null()
            & pl.col("target").is_not_null()
            & (pl.col("source").str.strip_chars() != "")
            & (pl.col("target").str.strip_chars() != "")
        )
        .select(["document_id", "segment_id", "domain", "source", "target"])
    )


def filter_open_subtitles(
    df: "pl.DataFrame",
    lang_pair: str = "en-uk",
    stats_dir: Optional[Path] = None,
) -> "pl.DataFrame":
    """
    OpenSubtitles (OPUS Moses): highly noisy, no quality columns.
    Runs full pipeline including LangID (S3), UK-quality (S4), and semantic (S8).
    """
    from cleaning_pipeline import run_stages

    return run_stages(
        df, lang_pair, stages=[1, 2, 3, 4, 5, 6, 7, 8],
        dataset=f"open_subtitles_{lang_pair}", stats_dir=stats_dir,
    )


def filter_tilde(
    df: "pl.DataFrame",
    lang_pair: str = "en-uk",
    stats_dir: Optional[Path] = None,
) -> "pl.DataFrame":
    """TildeMODEL (OPUS Moses): basic alignment only, moderate noise — full pipeline."""
    from cleaning_pipeline import run_stages

    return run_stages(
        df, lang_pair, stages=[1, 2, 3, 4, 5, 6, 7, 8],
        dataset=f"tilde_{lang_pair}", stats_dir=stats_dir,
    )


def filter_paracrawl(
    df: "pl.DataFrame",
    lang_pair: str = "en-uk",
    stats_dir: Optional[Path] = None,
) -> "pl.DataFrame":
    """
    ParaCrawl (OPUS Moses): CLD2 LID + neural cleaning applied upstream,
    but still noisy at lower confidence ranges — full pipeline.
    """
    from cleaning_pipeline import run_stages

    return run_stages(
        df, lang_pair, stages=[1, 2, 3, 4, 5, 6, 7, 8],
        dataset=f"paracrawl_{lang_pair}", stats_dir=stats_dir,
    )


def _build_docholt_pairs(
    df: "pl.DataFrame",
    align_min: float = 0.70,
    biclean_min: float = 0.85,
    bifix_min: float = 0.80,
    para_win: int = 4,
    para_stride: int = 2,
    sent_sample_rate: float = 0.15,
) -> "pl.DataFrame":
    """
    Unpack DocHPLT document-level structure into source/target pairs.

    Input
    -----
    Each row is one document: a list of sentence-pair dicts
        {"src": [str], "tgt": [str], "aligner-score": float,
         "bicleaner-score": float, "bifixer-score": float}

    Processing
    ----------
    1. Per document, keep only sentence pairs that pass all three score thresholds.
    2. Build paragraph pairs with a sliding window over consecutive kept sentences.
    3. Sample a fraction of individual sentence pairs from sentences not used
       in any paragraph window.

    Output
    ------
    DataFrame with 'source' and 'target' columns (mix of paragraphs + sentences).
    """
    # Find which column holds the list-of-dicts structure
    doc_col: Optional[str] = None
    for col in df.columns:
        sample = df[col][0] if len(df) > 0 else None
        if isinstance(sample, list):
            doc_col = col
            break
    if doc_col is None:
        raise ValueError(
            f"No list-type column found in DocHPLT DataFrame.\n"
            f"Columns: {df.columns}"
        )

    para_pairs: list[tuple[str, str]] = []
    sent_pairs: list[tuple[str, str]] = []

    for doc in df[doc_col].to_list():
        if not isinstance(doc, list):
            continue

        # Filter sentence pairs by quality scores
        good: list[tuple[str, str]] = []
        for pair in doc:
            if not isinstance(pair, dict):
                continue
            align   = float(pair.get("aligner-score",  0) or 0)
            biclean = float(pair.get("bicleaner-score", 0) or 0)
            bifix   = float(pair.get("bifixer-score",   0) or 0)
            if align >= align_min and biclean >= biclean_min and bifix >= bifix_min:
                src_list = pair.get("src", [])
                tgt_list = pair.get("tgt", [])
                src_text = (src_list[0] if src_list else "").strip()
                tgt_text = (tgt_list[0] if tgt_list else "").strip()
                if src_text and tgt_text:
                    good.append((src_text, tgt_text))

        if not good:
            continue

        n = len(good)
        used_idx: set[int] = set()

        # Sliding-window paragraph pairs from consecutive kept sentences
        if n >= para_win:
            for pos in range(0, n - para_win + 1, para_stride):
                window = good[pos : pos + para_win]
                para_pairs.append((
                    " ".join(s for s, _ in window),
                    " ".join(t for _, t in window),
                ))
                used_idx.update(range(pos, pos + para_win))

        # Sentence sample from sentences not used in any paragraph window
        sent_pool = [p for i, p in enumerate(good) if i not in used_idx]
        if sent_pool and sent_sample_rate > 0:
            n_take = max(1, int(len(sent_pool) * sent_sample_rate))
            sent_pairs.extend(random.sample(sent_pool, min(n_take, len(sent_pool))))

    all_pairs = para_pairs + sent_pairs
    if not all_pairs:
        return pl.DataFrame({
            "source": pl.Series([], dtype=pl.Utf8),
            "target": pl.Series([], dtype=pl.Utf8),
        })

    print(f"  DocHPLT: {len(para_pairs):,} para + {len(sent_pairs):,} sent  "
          f"(from {len(df):,} docs)")
    return pl.DataFrame({
        "source": [s for s, _ in all_pairs],
        "target": [t for _, t in all_pairs],
    })


def filter_docholt(
    df: "pl.DataFrame",
    lang_pair: str = "en-uk",
    stats_dir: Optional[Path] = None,
) -> "pl.DataFrame":
    """
    DocHPLT: per-sentence quality columns (aligner ≥ 0.70, bicleaner ≥ 0.85,
    bifixer ≥ 0.80).  Builds paragraph pairs + sentence sample from documents,
    then runs pipeline S1 + S2 + S4 + S5 + S7.
    """
    from cleaning_pipeline import run_stages

    df_pairs = _build_docholt_pairs(df)
    return run_stages(
        df_pairs, lang_pair, stages=[1, 2, 4, 5, 7],
        dataset=f"docholt_{lang_pair}", stats_dir=stats_dir,
    )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _unpack_translation(translation_series: "pl.Series") -> "pl.DataFrame":
    """
    Convert a Series of translation dicts / Structs into source + target columns.
    First key → source, second key → target.
    Handles both pl.Struct dtype (auto-detected) and Python-dict objects.
    """
    if hasattr(translation_series.dtype, "fields"):
        fields = translation_series.struct.fields
        return pl.DataFrame({
            "source": translation_series.struct.field(fields[0]),
            "target": translation_series.struct.field(fields[1]),
        })

    rows = translation_series.to_list()
    sources, targets = [], []
    for row in rows:
        if isinstance(row, dict):
            vals = list(row.values())
            sources.append(vals[0] if len(vals) > 0 else None)
            targets.append(vals[1] if len(vals) > 1 else None)
        else:
            sources.append(None)
            targets.append(None)
    return pl.DataFrame({"source": sources, "target": targets})


# =============================================================================
# 3.  SAVE
# =============================================================================

def save_sentence_corpus(df: "pl.DataFrame", out_dir: Path, filename: str) -> None:
    """Save source/target TSV, dropping any remaining null/empty rows."""
    df = df.filter(
        pl.col("source").is_not_null()
        & pl.col("target").is_not_null()
        & (pl.col("source").str.strip_chars() != "")
        & (pl.col("target").str.strip_chars() != "")
    ).select(["source", "target"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.write_csv(out_path, separator="\t")
    print(f"  saved → {out_path}  ({len(df):,} pairs)")


def build_and_save_wmt24pp(
    df: "pl.DataFrame",
    out_dir: Path,
    filename: str,
) -> None:
    """
    Build a mixed sentence+paragraph corpus from WMT24PP.
    Requires columns: document_id, segment_id, domain, source, target.

    Steps
    -----
    a) Sliding-window paragraph blocks per document (domain-specific window/stride).
    b) Exclude paragraph-used segments from sentence pool.
    c) Stratified-sample sentences by domain (SENTENCE_SAMPLE_RATE).
    d) Mix to SENTENCE_RATIO, shuffle, save TSV.
    """
    pdf: pd.DataFrame = df.to_pandas().reset_index(drop=True)

    para_records: list[dict[str, str]] = []
    used_indices: set[int] = set()

    for _doc_id, group in pdf.groupby("document_id"):
        group  = group.sort_values("segment_id")
        domain = group["domain"].iloc[0]
        if domain not in PARAGRAPH_CFG:
            continue
        size, stride = PARAGRAPH_CFG[domain]
        n = len(group)
        if n < size:
            continue
        seen_starts: set[int] = set()
        for pos in range(0, n - size + 1, stride):
            if pos in seen_starts:
                continue
            seen_starts.add(pos)
            window = group.iloc[pos : pos + size]
            para_records.append({
                "source": " ".join(window["source"].tolist()),
                "target": " ".join(window["target"].tolist()),
            })
            used_indices.update(window.index.tolist())

    paragraphs = pd.DataFrame(para_records, columns=["source", "target"])

    eligible   = pdf[~pdf.index.isin(used_indices)]
    sent_parts: list[pd.DataFrame] = []
    for _domain, grp in eligible.groupby("domain"):
        sent_parts.append(
            grp.sample(frac=SENTENCE_SAMPLE_RATE, random_state=SEED)[["source", "target"]]
        )
    sentences = (
        pd.concat(sent_parts, ignore_index=True)
        if sent_parts
        else pd.DataFrame(columns=["source", "target"])
    )

    n_para        = len(paragraphs)
    n_sent_target = int(n_para * SENTENCE_RATIO / (1 - SENTENCE_RATIO))
    n_sent        = min(n_sent_target, len(sentences))
    if n_sent > 0:
        sentences = sentences.sample(n=n_sent, random_state=SEED)

    combined = (
        pd.concat([sentences, paragraphs], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    combined[["source", "target"]].to_csv(out_path, sep="\t", index=False)

    pct = n_sent / len(combined) * 100 if len(combined) else 0
    print(
        f"  saved → {out_path}\n"
        f"  total={len(combined):,}  sent={n_sent:,} ({pct:.1f}%)  "
        f"para={n_para:,} ({100-pct:.1f}%)"
    )


# =============================================================================
# 4.  MAIN  —  load → filter+clean → save
# =============================================================================

if __name__ == "__main__":

    STATS_DIR = DATA_DIR / "stats"

    # ── en-uk training ────────────────────────────────────────────────────────

    print("\n=== MaCoCu uk-en ===")
    save_sentence_corpus(
        filtered_macocu(load_macocu_gz(download_macocu("uk-en")),
                        stats_dir=STATS_DIR / "en-uk"),
        OUT_TRAIN / "en-uk", "macocu.tsv",
    )

    print("\n=== NLLB eng_Latn-ukr_Cyrl ===")
    save_sentence_corpus(
        filter_nllb(load_huggingface("allenai/nllb", "eng_Latn-ukr_Cyrl"),
                    lang_pair="en-uk", stats_dir=STATS_DIR / "en-uk"),
        OUT_TRAIN / "en-uk", "nllb.tsv",
    )

    print("\n=== OpenSubtitles en-uk ===")
    save_sentence_corpus(
        filter_open_subtitles(
            load_opus_moses(
                "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-uk.txt.zip",
                src_lang="en",
            ),
            lang_pair="en-uk", stats_dir=STATS_DIR / "en-uk",
        ),
        OUT_TRAIN / "en-uk", "open_subtitles.tsv",
    )

    print("\n=== ParaCrawl en-uk ===")
    save_sentence_corpus(
        filter_paracrawl(
            load_opus_moses(
                "https://object.pouta.csc.fi/OPUS-ParaCrawl-Bonus/v9/moses/en-uk.txt.zip",
                src_lang="en",
            ),
            lang_pair="en-uk", stats_dir=STATS_DIR / "en-uk",
        ),
        OUT_TRAIN / "en-uk", "paracrawl.tsv",
    )

    print("\n=== DocHPLT en-uk ===")
    save_sentence_corpus(
        filter_docholt(load_huggingface("HPLT/DocHPLT", "en-uk"),
                       lang_pair="en-uk", stats_dir=STATS_DIR / "en-uk"),
        OUT_TRAIN / "en-uk", "docholt.tsv",
    )

    # ── cs-uk training ────────────────────────────────────────────────────────

    print("\n=== NLLB ces_Latn-ukr_Cyrl ===")
    save_sentence_corpus(
        filter_nllb(load_huggingface("allenai/nllb", "ces_Latn-ukr_Cyrl"),
                    lang_pair="cs-uk", stats_dir=STATS_DIR / "cs-uk"),
        OUT_TRAIN / "cs-uk", "nllb.tsv",
    )

    print("\n=== OpenSubtitles cs-uk ===")
    save_sentence_corpus(
        filter_open_subtitles(
            load_opus_moses(
                "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/cs-uk.txt.zip",
                src_lang="cs",
            ),
            lang_pair="cs-uk", stats_dir=STATS_DIR / "cs-uk",
        ),
        OUT_TRAIN / "cs-uk", "open_subtitles.tsv",
    )

    # ── en-uk validation ──────────────────────────────────────────────────────

    print("\n=== WMT24++ en-uk validation ===")
    build_and_save_wmt24pp(
        filter_wmt24pp(load_huggingface("google/wmt24pp", "en-uk")),
        OUT_VAL / "en-uk", "wmt24pp.tsv",
    )

    print("\n=== TildeMODEL en-uk validation ===")
    save_sentence_corpus(
        filter_tilde(
            load_opus_moses(
                "https://object.pouta.csc.fi/OPUS-TildeMODEL/v2018/moses/en-uk.txt.zip",
                src_lang="en",
            ),
            lang_pair="en-uk", stats_dir=STATS_DIR / "en-uk",
        ),
        OUT_VAL / "en-uk", "tilde.tsv",
    )

    print("\n=== OpenSubtitles en-uk validation ===")
    save_sentence_corpus(
        filter_open_subtitles(
            load_huggingface("Helsinki-NLP/OpenSubtitles2024", "en-uk", split="validation"),
            lang_pair="en-uk", stats_dir=STATS_DIR / "en-uk",
        ),
        OUT_VAL / "en-uk", "open_subtitles_en.tsv",
    )

    # ── cs-uk validation ──────────────────────────────────────────────────────

    print("\n=== OpenSubtitles cs-uk validation ===")
    save_sentence_corpus(
        filter_open_subtitles(
            load_huggingface("Helsinki-NLP/OpenSubtitles2024", "cs-uk", split="validation"),
            lang_pair="cs-uk", stats_dir=STATS_DIR / "cs-uk",
        ),
        OUT_VAL / "cs-uk", "open_subtitles_cs.tsv",
    )
