#!/usr/bin/env python3
"""Download English–Ukrainian parallel corpora for MT training.

Sources (OPUS Moses + HuggingFace NLLB), matching the WMT constrained-style
mix used for EN→UK:

  - OPUS: ParaCrawl, WikiMatrix, TED2020, Tatoeba, EUbookshop, CCAligned,
          OpenSubtitles, Wikimedia, XLEnt, QED, Bible, Ubuntu, KDE4, GNOME
  - HuggingFace: allenai/nllb (eng_Latn-ukr_Cyrl), LASER score filtered

Writes merged Moses files:
  data/raw/eng-ukr/train.eng
  data/raw/eng-ukr/train.ukr

Usage
-----
  python scripts/download_mt_en_uk.py
  python scripts/download_mt_en_uk.py --max-per-corpus 200000
  python scripts/download_mt_en_uk.py --skip-nllb
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw" / "eng-ukr"
CACHE_DIR = REPO / "data" / "raw" / "opus_cache"

# Compact OPUS Moses URLs (en-uk). Enough diversity for SDKM retrieval.
OPUS_MOSES: list[tuple[str, str]] = [
    (
        "paracrawl_v9",
        "https://object.pouta.csc.fi/OPUS-ParaCrawl/v9/moses/en-uk.txt.zip",
    ),
    (
        "wikimatrix_v1",
        "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-uk.txt.zip",
    ),
    (
        "ted2020_v1",
        "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-uk.txt.zip",
    ),
    (
        "tatoeba_v2023",
        "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-uk.txt.zip",
    ),
    (
        "eubookshop_v2",
        "https://object.pouta.csc.fi/OPUS-EUbookshop/v2/moses/en-uk.txt.zip",
    ),
    (
        "ccaligned_v1",
        "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/en-uk.txt.zip",
    ),
    (
        "opensubtitles_v2024",
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-uk.txt.zip",
    ),
    (
        "wikimedia_v2023",
        "https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/moses/en-uk.txt.zip",
    ),
    (
        "xlent_v1.2",
        "https://object.pouta.csc.fi/OPUS-XLEnt/v1.2/moses/en-uk.txt.zip",
    ),
    (
        "qed_v2",
        "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-uk.txt.zip",
    ),
    (
        "bible_uedin_v1",
        "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/en-uk.txt.zip",
    ),
    (
        "ubuntu_v14",
        "https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-uk.txt.zip",
    ),
    (
        "kde4_v2",
        "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-uk.txt.zip",
    ),
    (
        "gnome_v1",
        "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en-uk.txt.zip",
    ),
    (
        "macocu_v2",
        "https://object.pouta.csc.fi/OPUS-MaCoCu/v2/moses/en-uk.txt.zip",
    ),
]

NLLB_LASER_MIN = 1.2


def download_bytes(url: str, dest: Path) -> bytes:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cache] {dest.name}")
        return dest.read_bytes()
    print(f"  Downloading {url}")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    data = b"".join(resp.iter_content(chunk_size=1 << 20))
    dest.write_bytes(data)
    print(f"  Saved {dest} ({len(data) / 1e6:.1f} MB)")
    return data


def parse_moses_zip(raw: bytes) -> list[tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        en_name = next((n for n in names if n.endswith(".en")), None)
        uk_name = next((n for n in names if n.endswith(".uk")), None)
        if en_name is None or uk_name is None:
            raise ValueError(f"No .en/.uk Moses files in zip. Found: {names}")
        en_lines = zf.read(en_name).decode("utf-8", errors="replace").splitlines()
        uk_lines = zf.read(uk_name).decode("utf-8", errors="replace").splitlines()
    if len(en_lines) != len(uk_lines):
        n = min(len(en_lines), len(uk_lines))
        en_lines, uk_lines = en_lines[:n], uk_lines[:n]
    pairs = []
    for s, t in zip(en_lines, uk_lines):
        s, t = s.strip(), t.strip()
        if s and t:
            pairs.append((s, t))
    return pairs


def load_opus_corpus(name: str, url: str, max_pairs: int | None) -> list[tuple[str, str]]:
    zip_path = CACHE_DIR / f"{name}.zip"
    try:
        raw = download_bytes(url, zip_path)
        pairs = parse_moses_zip(raw)
    except Exception as exc:
        print(f"  [skip] {name}: {exc}")
        return []
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    print(f"  {name}: {len(pairs):,} pairs")
    return pairs


def load_nllb(max_pairs: int | None) -> list[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(f"  [skip] NLLB (datasets not installed): {exc}")
        return []

    print("  Loading allenai/nllb eng_Latn-ukr_Cyrl …")
    try:
        ds = load_dataset(
            "allenai/nllb",
            "eng_Latn-ukr_Cyrl",
            split="train",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"  [skip] NLLB load failed: {exc}")
        return []

    pairs: list[tuple[str, str]] = []
    for row in ds:
        score = row.get("laser_score")
        if score is not None and float(score) < NLLB_LASER_MIN:
            continue
        translation = row.get("translation") or {}
        s = (translation.get("eng_Latn") or "").strip()
        t = (translation.get("ukr_Cyrl") or "").strip()
        if s and t:
            pairs.append((s, t))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    print(f"  nllb: {len(pairs):,} pairs (laser ≥ {NLLB_LASER_MIN})")
    return pairs


def write_moses(pairs: list[tuple[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    en_path = out_dir / "train.eng"
    uk_path = out_dir / "train.ukr"
    with en_path.open("w", encoding="utf-8") as fe, uk_path.open("w", encoding="utf-8") as fu:
        for s, t in pairs:
            fe.write(s.replace("\n", " ").replace("\t", " ") + "\n")
            fu.write(t.replace("\n", " ").replace("\t", " ") + "\n")
    print(f"Wrote {len(pairs):,} pairs → {en_path} / {uk_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--max-per-corpus", type=int, default=None, help="Cap pairs per OPUS/NLLB corpus")
    parser.add_argument("--skip-nllb", action="store_true")
    parser.add_argument("--corpora", nargs="*", default=None, help="Optional subset of OPUS corpus names")
    args = parser.parse_args()

    selected = {c for c in args.corpora} if args.corpora else None
    all_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for name, url in OPUS_MOSES:
        if selected is not None and name not in selected:
            continue
        for pair in load_opus_corpus(name, url, args.max_per_corpus):
            if pair not in seen:
                seen.add(pair)
                all_pairs.append(pair)

    if not args.skip_nllb:
        for pair in load_nllb(args.max_per_corpus):
            if pair not in seen:
                seen.add(pair)
                all_pairs.append(pair)

    if not all_pairs:
        raise SystemExit("No pairs downloaded. Check network / corpus names.")
    write_moses(all_pairs, args.out_dir)
    print(f"Done. Unique pairs: {len(all_pairs):,}")


if __name__ == "__main__":
    main()
