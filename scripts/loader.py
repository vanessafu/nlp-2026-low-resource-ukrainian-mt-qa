"""
loader.py  —  Download raw corpora and build a mixed-granularity parallel corpus.

Run:
    python scripts/loader.py

Output:
    data/training/MT/en-uk/wmt24pp_raw.tsv   (source / target, TSV)

Sections
--------
1. Load raw data      (HuggingFace: google/wmt24pp  |  OPUS via datasets)
2. Filter & select fields
3. Build sentence + paragraph samples, mix, save
"""

from __future__ import annotations

import random
from pathlib import Path
import requests
import zipfile
import io
import polars as pl

import pandas as pd
from datasets import load_dataset

# ── global config ─────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)

SENTENCE_RATIO       = 0.70   # target fraction of sentence samples in final mix
SENTENCE_SAMPLE_RATE = 0.65   # fraction of eligible (non-paragraph) segments to keep

PARAGRAPH_CFG = {
    # domain: (window_size, stride)
    "news":     (6, 3),
    "social":   (3, 2),
    "speech":   (8, 4),
    "literary": (5, 2),
}

OUT_ROOT = Path(__file__).parent.parent / "data" / "training" / "MT"


# =============================================================================
# 1. LOAD RAW DATA
# =============================================================================

def load_huggingface(dataset_name: str, lang_pair: str) -> pl.DataFrame:
    """Download dataset from HuggingFace for the given language pair (e.g. 'en-uk')."""
    ds = load_dataset(
        dataset_name,
        lang_pair,
    )
    return pl.DataFrame(ds.to_pandas())


def load_opus(url: str, source_lang: list[str]) -> list[pl.DataFrame]:
    """
    Download an OPUS dataset from the given URL (expects a zip with .en and .uk files), read into memory, and return a DataFrame with 'en' and 'uk' columns.
    """
    response = requests.get(url)
    response.raise_for_status()
    df = []

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_list = z.namelist()
        for sl in source_lang:
            if not any(f.endswith(f'.{sl}') for f in file_list):
                raise ValueError(f"Error: no file ending with '.{sl}' found in the zip archive.")
            source_filename = next(f for f in file_list if f.endswith(f'.{sl}'))
            target_filename = next(f for f in file_list if f.endswith('.uk'))

            source_lines = z.read(source_filename).decode('utf-8').splitlines()
            target_lines = z.read(target_filename).decode('utf-8').splitlines()
        
            assert len(source_lines) == len(target_lines), "Error: source and target files have different number of lines."
        
            df.append(pl.DataFrame({
                source_lang[0]: source_lines,
                "uk": target_lines
            }))
    
    return df


# =============================================================================
# 2. FILTER & SELECT FIELDS
# =============================================================================

def filter_wmt24pp(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply WMT24PP-specific quality filters and keep only needed columns.
    - Remove domain == 'canary'
    - Remove is_bad_source == True
    - Use 'target' (post-edit) as the reference translation
    - Keep only: document_id, segment_id, domain, source, target
    """
    df = df[df["domain"] != "canary"]
    df = df[~df["is_bad_source"]]
    df = df[["document_id", "segment_id", "domain", "source", "target"]].copy()
    df = df.dropna(subset=["source", "target"])
    df = df[df["source"].str.strip() != ""]
    df = df[df["target"].str.strip() != ""]
    return df.reset_index(drop=True)

def filtered_macocu(df: pl.DataFrame) -> pl.DataFrame:
    """Basic quality filter for MaCoCu dataset (no document structure)."""
    df = df[["source", "target"]].copy()
    df = df.dropna()
    df = df[df["source"].str.strip() != ""]
    df = df[df["target"].str.strip() != ""]
    return df.reset_index(drop=True)

def filter_open_subtitles(df: pl.DataFrame) -> pl.DataFrame:
    """Basic quality filter for OpenSubtitles dataset (no document structure)."""
    df = df[["source", "target"]].copy()
    df = df.dropna()
    df = df[df["source"].str.strip() != ""]
    df = df[df["target"].str.strip() != ""]
    return df.reset_index(drop=True)

def filter_tilde(df: pl.DataFrame) -> pl.DataFrame:
    """Basic quality filter for TildeMODEL dataset (no document structure)."""
    df = df[["source", "target"]].copy()
    df = df.dropna()
    df = df[df["source"].str.strip() != ""]
    df = df[df["target"].str.strip() != ""]
    return df.reset_index(drop=True)

def filter_nllb(df: pl.DataFrame) -> pl.DataFrame:
    """Basic quality filter for NLLB dataset (no document structure)."""
    df = df[["source", "target"]].copy()
    df = df.dropna()
    df = df[df["source"].str.strip() != ""]
    df = df[df["target"].str.strip() != ""]
    return df.reset_index(drop=True)

# =============================================================================
# 3. BUILD SENTENCE + PARAGRAPH SAMPLES  →  MIX  →  SAVE
# =============================================================================

def build_and_save(df: pl.DataFrame, out_dir: Path, filename: str, granularity: bool) -> None:
    """
    Build a mixed-granularity corpus from a filtered WMT24PP DataFrame.

    Steps:
      a) Build paragraph windows per document (domain-specific size + stride).
      b) Exclude paragraph-used segments from the sentence pool.
      c) Stratified-sample sentences by domain.
      d) Mix to reach SENTENCE_RATIO, shuffle, save as TSV with source / target only.
    """
    df = df.reset_index(drop=True)  # ensure 0-based integer index
    if granularity:
        # ── a) paragraphs ──────────────────────────────────────────────────────
        para_records: list[dict] = []
        used_row_indices: set[int] = set()

        for doc_id, group in df.groupby("document_id"):
            group = group.sort_values("segment_id")  # keeps original df index
            domain = group["domain"].iloc[0]

            if domain not in PARAGRAPH_CFG:
                continue

            size, stride = PARAGRAPH_CFG[domain]
            n = len(group)
            if n < size:
                continue

            # sliding window — no cross-document concatenation
            seen_starts: set[int] = set()
            positions = list(range(0, n - size + 1, stride))
            for pos in positions:
                if pos in seen_starts:
                    continue
                seen_starts.add(pos)

                window = group.iloc[pos : pos + size]
                src = " ".join(window["source"].tolist())
                tgt = " ".join(window["target"].tolist())
                para_records.append({"source": src, "target": tgt})
                used_row_indices.update(window.index.tolist())

        paragraphs = pd.DataFrame(para_records, columns=["source", "target"])

        # ── b+c) sentences — exclude rows already used in any paragraph window ─
        eligible = df[~df.index.isin(used_row_indices)]

        sent_parts: list[pd.DataFrame] = []
        for domain, group in eligible.groupby("domain"):
            sampled = group.sample(frac=SENTENCE_SAMPLE_RATE, random_state=SEED)
            sent_parts.append(sampled[["source", "target"]])

        sentences = (
            pd.concat(sent_parts, ignore_index=True)
            if sent_parts
            else pd.DataFrame(columns=["source", "target"])
        )

    # ── d) mix to SENTENCE_RATIO and save ─────────────────────────────────
    n_para = len(paragraphs)
    n_sent_target = int(n_para * SENTENCE_RATIO / (1 - SENTENCE_RATIO))
    n_sent = min(n_sent_target, len(sentences))

    if n_sent > 0:
        sentences = sentences.sample(n=n_sent, random_state=SEED)

    combined = pd.concat(
        [sentences, paragraphs], ignore_index=True
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    combined[["source", "target"]].to_csv(out_path, sep="\t", index=False)

    actual_sent_pct = n_sent / len(combined) * 100 if len(combined) else 0
    print(
        f"  saved → {out_path}\n"
        f"  total: {len(combined)}  |  "
        f"sentence: {n_sent} ({actual_sent_pct:.1f}%)  |  "
        f"paragraph: {n_para} ({100 - actual_sent_pct:.1f}%)"
    )


if __name__ == "__main__":
    #training set:
    #MT data froom OPUS:
    open_subtitles_raw = load_opus("https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-uk.txt.zip",["en","cs"])
    
    #en-uk 
    macocu_raw_en= load_opus("https://object.pouta.csc.fi/OPUS-MaCoCu/v2/moses/en-uk.txt.zip")
    filtered_macocu = filtered_macocu(macocu_raw_en,"en")
    build_and_save(filtered_macocu, OUT_ROOT / "en-uk", "macocu_raw.tsv")
    filtered_open_subtitles_en = filter_open_subtitles(open_subtitles_raw[0],"en")
    build_and_save(filtered_open_subtitles_en, OUT_ROOT / "en-uk", "open_subtitles_raw.tsv")
    tilde_raw_en = load_opus("https://object.pouta.csc.fi/OPUS-TildeMODEL/v2018/moses/en-uk.txt.zip","en")
    filtered_tilde_en = filter_tilde(tilde_raw_en)
    build_and_save(filtered_tilde_en, OUT_ROOT / "en-uk", "tilde_raw.tsv")
    nllb_raw_en = load_huggingface("allenai/nllb", "eng_Latn-ukr_Cyrl")
    filtered_nllb_en = filter_nllb(nllb_raw_en)
    build_and_save(filtered_nllb_en, OUT_ROOT / "en-uk", "nllb_raw.tsv")
    dochplt_raw_en = load_huggingface("HPLT/DocHPLT", "en-uk")
    filtered_dochplt_en = filter_nllb(dochplt_raw_en)
    build_and_save(filtered_dochplt_en, OUT_ROOT / "en-uk", "dochplt_raw.tsv")
    
    #cz-uk
    filtered_open_subtitles_cz = filter_open_subtitles(open_subtitles_raw[1],"cs")
    build_and_save(filtered_open_subtitles_cz, OUT_ROOT / "cz-uk", "open_subtitles_raw.tsv")
    nllb_raw_cz = load_huggingface("allenai/nllb", "ces_Latn-ukr_Cyrl")
    filtered_nllb_cz = filter_nllb(nllb_raw_cz)
    build_and_save(filtered_nllb_cz, OUT_ROOT / "cz-uk", "nllb_raw.tsv")
    
    #qa
    
    
    # validation set:
    
    # MT
    wmt24pp_raw = load_huggingface("google/wmt24pp", "en-uk")
    filtered = filter_wmt24pp(wmt24pp_raw)
    build_and_save(filtered, OUT_ROOT / "en-uk", "wmt24pp_raw.tsv")
    # QA
    