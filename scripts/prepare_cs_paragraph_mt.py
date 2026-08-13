#!/usr/bin/env python3
"""
Build the full cs-uk paragraph-level MT training set.

Steps
-----
1. Run every cs-uk corpus builder in loader.py (eubookshop, elrc, wikimedia, nllb)
   -> data/training/MT/cs-uk/*.tsv
2. Download UberText fiction, extract 300-700 word paragraphs, back-translate
   uk->cs via LINDAT, clean -> data/training/MT/cs-uk/fiction.tsv
3. Assemble every TSV in data/training/MT/cs-uk/ into train_data/MT/train.cs-uk.jsonl

Usage
-----
  python scripts/prepare_cs_paragraph_mt.py
  python scripts/prepare_cs_paragraph_mt.py --skip-corpora    # fiction + assemble only
  python scripts/prepare_cs_paragraph_mt.py --skip-fiction    # corpora + assemble only
  python scripts/prepare_cs_paragraph_mt.py --no-align
"""
from __future__ import annotations

import argparse
import bz2
import csv
import json
import random
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import requests

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import loader
from converter import _mt_record, _read_tsv

TRAIN_DIR = _REPO / "data" / "training" / "MT" / "cs-uk"
OUT_JSONL = _REPO / "train_data" / "MT" / "train.cs-uk.jsonl"

CS_UK_CAPS = {"nllb.tsv": 58_000, "elrc_acts.tsv": 2_000}
CS_UK_EXCLUDED = {"open_subtitles.tsv", "open_subtitles_para.tsv",
                  "paracrawl.tsv", "paracrawl_para.tsv", "ubertext_para.tsv"}


FICTION_URL = ("https://lang.org.ua/static/downloads/ubertext2.0/fiction/cleansed/"
               "ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2")
RAW_BZ2       = _REPO / "data" / "raw" / "ubertext.fiction.txt.bz2"
RAW_TXT       = _REPO / "data" / "raw" / "fiction.txt"
RAW_FICTION_TSV = _REPO / "data" / "processed" / "fiction_backtrans_raw.tsv"
FICTION_TSV   = TRAIN_DIR / "fiction.tsv"

MIN_WORDS, MAX_WORDS = 300, 700
STANZA_BATCH_CHARS   = 200_000   # cap raw text handed to stanza per call

LINDAT_BASE       = "https://lindat.mff.cuni.cz/services/translation/api/v2"
MODEL_UK_CS       = "uk-cs"
REQUEST_DELAY     = 0.35
MAX_PAYLOAD_BYTES = 30_000

FICTION_STAGES = [1, 2, 3, 6, 7, 8]


def download_fiction() -> Path:
    if RAW_TXT.exists():
        return RAW_TXT
    RAW_BZ2.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_BZ2.exists():
        print(f"Downloading {FICTION_URL} …")
        urllib.request.urlretrieve(FICTION_URL, RAW_BZ2)
    print(f"Decompressing {RAW_BZ2.name} → {RAW_TXT.name} …")
    with bz2.open(RAW_BZ2, "rb") as src, open(RAW_TXT, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return RAW_TXT


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import stanza
        _nlp = stanza.Pipeline("uk", processors="tokenize", use_gpu=False, verbose=False)
    return _nlp


def split_sentences(text: str) -> list[str]:
    return [s.text.strip() for s in _get_nlp()(text).sentences if s.text.strip()]


def wc(text: str) -> int:
    return len(text.split())


def pack_sentences(sentences: list[str]) -> list[list[str]]:
    """Greedily pack sentences into MIN_WORDS-MAX_WORDS chunks, never splitting one."""
    paragraphs: list[list[str]] = []
    buf: list[str] = []
    buf_wc = 0
    for sent in sentences:
        sw = wc(sent)
        if sw > MAX_WORDS:
            if buf:
                paragraphs.append(buf)
                buf, buf_wc = [], 0
            paragraphs.append([sent])
            continue
        if buf_wc + sw > MAX_WORDS and buf_wc >= MIN_WORDS:
            paragraphs.append(buf)
            buf, buf_wc = [sent], sw
        else:
            buf.append(sent)
            buf_wc += sw
    if buf:
        if paragraphs and wc(" ".join(buf)) < MIN_WORDS:
            paragraphs[-1] = paragraphs[-1] + buf
        else:
            paragraphs.append(buf)
    return paragraphs


_RE_WS       = re.compile(r"\s+")
_RE_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_RE_WORD     = re.compile(r"\w")
_RE_DIGIT    = re.compile(r"\d")

# Ukrainian prose marks direct speech with a leading em/en-dash or a fully quoted line.
_DIALOGUE_PREFIX   = ("—", "–", "-")
_QUOTE_PAIRS       = (("«", "»"), ("„", "“"), ("“", "”"), ('"', '"'))
_RE_TRAILING_PUNCT = re.compile(r"[.!?…]+$")


def is_dialogue(sent: str) -> bool:
    s = sent.strip()
    if not s:
        return False
    if s[0] in _DIALOGUE_PREFIX:
        return True
    body = _RE_TRAILING_PUNCT.sub("", s)
    return any(s.startswith(o) and body.endswith(c) for o, c in _QUOTE_PAIRS)


def is_clean(text: str) -> bool:
    total = len(_RE_WORD.findall(text))
    if total == 0:
        return False
    if len(_RE_CYRILLIC.findall(text)) / total < 0.6:
        return False
    if len(_RE_DIGIT.findall(text)) / max(len(text), 1) > 0.15:
        return False
    return True


def index_stories(path: Path) -> list[int]:
    """Byte offset of each story's first line (3+ blank lines separate stories)."""
    offsets: list[int] = []
    blank_run = 0
    in_story = False
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.strip():
                blank_run += 1
                if blank_run >= 3:
                    in_story = False
            else:
                if not in_story:
                    offsets.append(pos)
                    in_story = True
                blank_run = 0
    return offsets


def read_story_at(path: Path, offset: int) -> list[str]:
    story: list[str] = []
    para: list[str] = []
    blank_run = 0
    with open(path, "rb") as f:
        f.seek(offset)
        for raw in f:
            line = raw.decode("utf-8", errors="ignore").rstrip("\n")
            if not line.strip():
                blank_run += 1
                if para:
                    story.append(" ".join(para))
                    para = []
                if blank_run >= 3:
                    break
            else:
                blank_run = 0
                para.append(line)
    if para:
        story.append(" ".join(para))
    return story


def extract_story_paragraphs(story: list[str], max_dialogue_ratio: float) -> list[str]:
    normalized = [p for p in (_RE_WS.sub(" ", p).strip() for p in story) if p]
    if not normalized:
        return []

    sentences: list[str] = []
    batch: list[str] = []
    batch_chars = 0
    for para in normalized:
        batch.append(para)
        batch_chars += len(para)
        if batch_chars >= STANZA_BATCH_CHARS:
            sentences.extend(split_sentences("\n\n".join(batch)))
            batch, batch_chars = [], 0
    if batch:
        sentences.extend(split_sentences("\n\n".join(batch)))

    chunks: list[str] = []
    for sent_group in pack_sentences(sentences):
        text = " ".join(sent_group)
        w = wc(text)
        if w < MIN_WORDS * 0.8 or w > MAX_WORDS * 1.2:
            continue
        if not is_clean(text):
            continue
        if sum(is_dialogue(s) for s in sent_group) / len(sent_group) > max_dialogue_ratio:
            continue
        chunks.append(text)
    return chunks


def iter_paragraphs(txt_path: Path, max_paragraphs: int, seed: int, max_dialogue_ratio: float):
    print(f"Indexing stories in {txt_path} …")
    offsets = index_stories(txt_path)
    random.Random(seed).shuffle(offsets)
    print(f"  {len(offsets):,} stories total")

    n = 0
    for i, offset in enumerate(offsets, 1):
        for para in extract_story_paragraphs(read_story_at(txt_path, offset), max_dialogue_ratio):
            yield para
            n += 1
            if n >= max_paragraphs:
                return
        if i % 200 == 0:
            print(f"  ... {i:,} stories scanned, {n:,} paragraphs so far", flush=True)


_RE_SENT     = re.compile(r"(?<=[.!?])\s+")
_QUOTE_CHARS = '"\'«»‘’“”‹›'


def _split_for_api(text: str) -> list[str]:
    if len(text.encode("utf-8")) <= MAX_PAYLOAD_BYTES:
        return [text]
    sentences = _RE_SENT.split(text)
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        candidate = (buf + " " + sent).strip() if buf else sent
        if len(candidate.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            if buf:
                chunks.append(buf)
            buf = sent
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks or [text]


def translate_chunk(text: str, session: requests.Session) -> str | None:
    url = f"{LINDAT_BASE}/models/{MODEL_UK_CS}"
    for attempt in range(1, 6):
        try:
            r = session.post(url, data={"input_text": text}, params={"src": "uk", "tgt": "cs"}, timeout=90)
            if r.status_code == 200:
                return r.content.decode("utf-8").strip()   # API sends UTF-8 without a charset header
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After") or 2 * attempt)
                print(f"    [warn] 429 rate-limited, waiting {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            print(f"    [warn] attempt {attempt}: HTTP {r.status_code}", flush=True)
        except requests.RequestException as exc:
            print(f"    [warn] attempt {attempt}: {exc}", flush=True)
        time.sleep(2 * attempt)
    return None


def back_translate(text: str, session: requests.Session) -> str | None:
    text = text.strip(_QUOTE_CHARS).strip()
    parts: list[str] = []
    for chunk in _split_for_api(text):
        t = translate_chunk(chunk, session)
        if t is None:
            return None
        parts.append(t)
        time.sleep(REQUEST_DELAY)
    return " ".join(parts)


def build_fiction(max_paragraphs: int, max_dialogue_ratio: float, no_align: bool, seed: int) -> None:
    txt_path = download_fiction()

    n_done = len(_read_tsv(RAW_FICTION_TSV))
    if n_done:
        print(f"Resuming: {n_done:,} paragraphs already translated on disk")

    print("\n── Extracting + back-translating paragraphs ──")
    session = requests.Session()
    n_ok = n_fail = 0
    for i, uk_para in enumerate(
        iter_paragraphs(txt_path, max_paragraphs, seed, max_dialogue_ratio), 1
    ):
        if i <= n_done:
            continue
        cs_para = back_translate(uk_para, session)
        if cs_para is None:
            n_fail += 1
            print(f"  [skip] paragraph {i} — translation failed", flush=True)
            continue

        write_header = not RAW_FICTION_TSV.exists()
        RAW_FICTION_TSV.parent.mkdir(parents=True, exist_ok=True)
        with open(RAW_FICTION_TSV, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            if write_header:
                writer.writerow(["source", "target"])
            writer.writerow([cs_para, uk_para])
        n_ok += 1

        if i % 100 == 0:
            print(f"  [{i:,}/{max_paragraphs:,}]  ok={n_ok:,}  fail={n_fail:,}", flush=True)

    print(f"\nTranslation complete — ok={n_ok:,}  fail={n_fail:,}")

    print("\n── Cleaning + saving fiction.tsv ──")
    stages = [s for s in FICTION_STAGES if not (no_align and s == 8)]
    pipe   = loader.build_pipeline("cs-uk", stages, embed_sim_min=0.87)
    clean  = loader.clean_pairs_in_chunks(_read_tsv(RAW_FICTION_TSV), pipe, flush_size=5_000)
    loader.save_tsv(clean, FICTION_TSV)


def assemble_train_jsonl() -> None:
    pairs: list[tuple[str, str]] = []
    for tsv in sorted(TRAIN_DIR.glob("*.tsv")):
        if tsv.name in CS_UK_EXCLUDED:
            print(f"    [excluded] {tsv.name}")
            continue
        cap  = CS_UK_CAPS.get(tsv.name)
        rows = _read_tsv(tsv, cap=cap)
        print(f"    {tsv.name}: {len(rows):,} pairs" + (f" (cap={cap:,})" if cap else ""))
        pairs.extend(rows)
    print(f"    total: {len(pairs):,}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for s, t in pairs:
            f.write(json.dumps(_mt_record("cs", s, t), ensure_ascii=False) + "\n")
    print(f"    saved → {OUT_JSONL}")


#  Main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-corpora", action="store_true", help="skip eubookshop/elrc/wikimedia/nllb")
    p.add_argument("--skip-fiction", action="store_true", help="skip the fiction back-translation")
    p.add_argument("--no-align", action="store_true", help="skip the semantic-alignment stage")
    p.add_argument("--max-paragraphs", type=int, default=50_000)
    p.add_argument("--max-dialogue-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=loader.SEED)
    args = p.parse_args()

    if not args.skip_corpora:
        print("\n========== eubookshop_cs_uk ==========")
        loader.build_eubookshop_cs_uk(no_align=args.no_align)
        print("\n========== elrc_cs_uk ==========")
        loader.build_elrc_cs_uk(no_align=args.no_align)
        print("\n========== wikimedia_cs_uk ==========")
        loader.build_wikimedia_cs_uk(no_align=args.no_align)
        print("\n========== nllb ==========")
        loader.build_nllb()

    if not args.skip_fiction:
        print("\n========== fiction back-translation ==========")
        build_fiction(args.max_paragraphs, args.max_dialogue_ratio, args.no_align, args.seed)

    print("\n========== assembling train_data/MT/train.cs-uk.jsonl ==========")
    assemble_train_jsonl()

    print("\nDone.")


if __name__ == "__main__":
    main()
