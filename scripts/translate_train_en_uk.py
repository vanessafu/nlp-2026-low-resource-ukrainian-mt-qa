#!/usr/bin/env python3
"""Translate English QA/MR training data to Ukrainian.

Default backend: free Google Translate via deep-translator (no API key).
Optional backend: Google Cloud Translation API (--backend cloud).

Writes *_en / *_uk fields directly into each task's train_data file:
  train_data/MR/gsm8k_train.jsonl
  train_data/MR/competition_math_train.jsonl
  train_data/QA/mmlu_en_train.jsonl
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIALS = ROOT / "config" / "google_cloud_credentials.json"
CACHE_PATH = ROOT / "train_data" / ".translation_cache_en_uk.json"
QA_OUT = ROOT / "train_data" / "QA" / "mmlu_en_train.jsonl"
GSM8K_PATH = ROOT / "train_data" / "MR" / "gsm8k_train.jsonl"
COMP_MATH_PATH = ROOT / "train_data" / "MR" / "competition_math_train.jsonl"
MMLU_TRAIN = ROOT / "train_data" / "QA" / "mmlu_en" / "auxiliary_train" / "train-00000-of-00001.parquet"

MAX_CHUNK_CHARS = 4500


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def cache_key(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def split_long_text(text: str) -> list[str]:
    text = text or ""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        parts.append(text[start:end])
        start = end
    return parts


class FreeGoogleTranslator:
    def __init__(self, sleep_s=0.3, batch_size=25, max_retries=5):
        from deep_translator import GoogleTranslator

        self._translator = GoogleTranslator(source="en", target="uk")
        self.sleep_s = sleep_s
        self.batch_size = batch_size
        self.max_retries = max_retries

    def _translate_chunk(self, chunk: list[str]) -> list[str]:
        for attempt in range(self.max_retries):
            try:
                out = self._translator.translate_batch(chunk)
                if len(out) != len(chunk):
                    raise RuntimeError(f"batch size mismatch: {len(out)} != {len(chunk)}")
                return out
            except Exception as exc:
                wait = min(60, 2 ** attempt)
                print(f"  translate retry {attempt + 1}/{self.max_retries}: {exc}", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"failed to translate batch of {len(chunk)} strings")

    def translate_many(self, texts: list[str], cache: dict) -> list[str]:
        results = [None] * len(texts)
        pending: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            text = text or ""
            key = cache_key(text)
            if key in cache:
                results[i] = cache[key]
            else:
                pending.append((i, text))

        total = len(pending)
        done = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            flat_texts: list[str] = []
            flat_map: list[tuple[int, int, int]] = []

            for orig_idx, text in batch:
                parts = split_long_text(text)
                part_start = len(flat_texts)
                flat_texts.extend(parts)
                flat_map.append((orig_idx, part_start, len(parts)))

            translated_parts = self._translate_chunk(flat_texts)
            for orig_idx, part_start, n_parts in flat_map:
                uk = "".join(translated_parts[part_start : part_start + n_parts])
                src = texts[orig_idx] or ""
                cache[cache_key(src)] = uk
                results[orig_idx] = uk

            done += len(batch)
            save_cache(cache)
            if done % 500 == 0 or done == total:
                print(f"  translated {done}/{total} new strings", flush=True)
            time.sleep(self.sleep_s)

        return results


class CloudGoogleTranslator:
    def __init__(self, api_key=None, batch_size=50, sleep_s=0.05, max_retries=5):
        from google.cloud import translate_v2 as translate

        api_key = api_key or os.environ.get("GOOGLE_TRANSLATE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = translate.Client(api_key=api_key) if api_key else translate.Client()
        self.batch_size = batch_size
        self.sleep_s = sleep_s
        self.max_retries = max_retries

    def _translate_strings(self, chunk: list[str]) -> list[str]:
        for attempt in range(self.max_retries):
            try:
                translated = self._client.translate(
                    chunk, target_language="uk", source_language="en", format_="text"
                )
                if isinstance(translated, dict):
                    translated = [translated]
                return [item["translatedText"] for item in translated]
            except Exception as exc:
                wait = min(30, 2 ** attempt)
                print(f"  cloud retry {attempt + 1}/{self.max_retries}: {exc}", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"cloud translate failed for batch of {len(chunk)}")

    def _translate_one(self, text: str) -> str:
        parts = split_long_text(text)
        if len(parts) == 1:
            return self._translate_strings([parts[0]])[0]
        return "".join(self._translate_strings(parts))

    def translate_many(self, texts: list[str], cache: dict) -> list[str]:
        results = [None] * len(texts)
        pending: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            text = text or ""
            key = cache_key(text)
            if key in cache:
                results[i] = cache[key]
            else:
                pending.append((i, text))

        total = len(pending)
        done = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            chunk = [text for _, text in batch]
            try:
                translated = self._translate_strings(chunk)
            except Exception:
                translated = [self._translate_one(text) for text in chunk]

            for (_, src), uk, (orig_idx, _) in zip(
                [(i, t) for i, t in batch], translated, batch
            ):
                cache[cache_key(src)] = uk
                results[orig_idx] = uk

            done += len(batch)
            save_cache(cache)
            if done % 500 == 0 or done == total:
                print(f"  translated {done}/{total} new strings", flush=True)
            time.sleep(self.sleep_s)

        return results


def setup_credentials(credentials: Path | None) -> Path:
    cred_path = credentials or Path(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", DEFAULT_CREDENTIALS)
    )
    cred_path = cred_path.expanduser().resolve()
    if not cred_path.is_file():
        raise FileNotFoundError(
            f"Google credentials not found: {cred_path}\n"
            "Copy your service account JSON to config/google_cloud_credentials.json"
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
    return cred_path


def get_translator(backend: str, api_key=None, sleep_s=0.3, batch_size=25, credentials=None):
    if backend == "cloud":
        setup_credentials(credentials)
        return CloudGoogleTranslator(api_key=api_key, batch_size=batch_size, sleep_s=sleep_s)
    return FreeGoogleTranslator(sleep_s=sleep_s, batch_size=batch_size)


def translate_batch(translator, texts, cache):
    return translator.translate_many(texts, cache)


def choices_list_to_dict(choices):
    return {str(i): c for i, c in enumerate(choices)}


def mr_en_text(row, field: str) -> str:
    en_key = f"{field}_en"
    return row.get(en_key) or row.get(field, "")


def export_mr_jsonl(translator, cache, rows_in, out_path, limit=None):
    if limit is not None:
        rows_in = rows_in[:limit]

    texts = []
    meta = []
    for i, row in enumerate(rows_in):
        texts.append(mr_en_text(row, "question"))
        meta.append((i, "question"))
        texts.append(mr_en_text(row, "answer"))
        meta.append((i, "answer"))

    translated = translate_batch(translator, texts, cache)
    uk_by_row = {i: {} for i in range(len(rows_in))}
    for (row_i, field), uk in zip(meta, translated):
        uk_by_row[row_i][field] = uk

    out_rows = []
    for i, row in enumerate(rows_in):
        q_en = mr_en_text(row, "question")
        a_en = mr_en_text(row, "answer")
        meta_row = {
            k: v
            for k, v in row.items()
            if k not in {"question", "answer", "question_en", "question_uk", "answer_en", "answer_uk"}
        }
        out_rows.append(
            {
                **meta_row,
                "question_en": q_en,
                "question_uk": uk_by_row[i]["question"],
                "answer_en": a_en,
                "answer_uk": uk_by_row[i]["answer"],
            }
        )

    write_jsonl(out_path, out_rows)
    return len(out_rows)


def export_gsm8k(translator, cache, limit=None):
    return export_mr_jsonl(translator, cache, read_jsonl(GSM8K_PATH), GSM8K_PATH, limit=limit)


def export_competition_math(translator, cache, limit=None):
    return export_mr_jsonl(
        translator, cache, read_jsonl(COMP_MATH_PATH), COMP_MATH_PATH, limit=limit
    )


def export_mmlu(translator, cache, limit=None):
    table = pq.read_table(MMLU_TRAIN)
    records = table.column("train").to_pylist()
    if limit is not None:
        records = records[:limit]

    texts = []
    meta = []
    for i, rec in enumerate(records):
        texts.append(rec["question"])
        meta.append((i, "question", None))
        for j, choice in enumerate(rec["choices"]):
            texts.append(choice)
            meta.append((i, "choice", j))

    translated = translate_batch(translator, texts, cache)
    buckets = {i: {"question_uk": None, "choices_uk": {}} for i in range(len(records))}
    for (row_i, field, j), uk in zip(meta, translated):
        if field == "question":
            buckets[row_i]["question_uk"] = uk
        else:
            buckets[row_i]["choices_uk"][str(j)] = uk

    out_rows = []
    for i, rec in enumerate(records):
        choices_en = choices_list_to_dict(rec["choices"])
        choices_uk = buckets[i]["choices_uk"]
        q_en = rec["question"]
        q_uk = buckets[i]["question_uk"]
        out_rows.append(
            {
                "dataset_id": "mmlu_en_train",
                "id": f"mmlu-en-{i:06d}",
                "subject": rec.get("subject") or "",
                "question_en": q_en,
                "question_uk": q_uk,
                "possible_answers_en": choices_en,
                "possible_answers_uk": choices_uk,
                "correct_answer_num": rec["answer"],
            }
        )

    write_jsonl(QA_OUT, out_rows)
    return len(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["free", "cloud"],
        default="free",
        help="free=deep-translator Google web API; cloud=Google Cloud Translation API",
    )
    parser.add_argument("--api-key", default=None, help="Only for --backend cloud (API key auth)")
    parser.add_argument(
        "--credentials",
        type=Path,
        default=None,
        help=f"Service account JSON (default: {DEFAULT_CREDENTIALS})",
    )
    parser.add_argument("--limit-qa", type=int, default=20000)
    parser.add_argument("--limit-mr", type=int, default=None, help="Limit GSM8K rows")
    parser.add_argument(
        "--limit-comp-math", type=int, default=None, help="Limit competition_math rows"
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.3, help="Pause between batches (free backend)")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--skip-mr", action="store_true")
    parser.add_argument("--skip-comp-math", action="store_true")
    args = parser.parse_args()

    if args.backend == "cloud":
        cred_path = setup_credentials(args.credentials)
        print(f"Credentials: {cred_path}", flush=True)

    translator = get_translator(
        args.backend,
        api_key=args.api_key,
        sleep_s=args.sleep,
        batch_size=args.batch_size,
        credentials=args.credentials,
    )
    cache = load_cache()

    print(f"Backend: {args.backend}, cache entries: {len(cache)}", flush=True)

    if not args.skip_mr:
        print("Translating GSM8K...", flush=True)
        n_gsm8k = export_gsm8k(translator, cache, limit=args.limit_mr)
        print(f"Wrote {n_gsm8k} rows to {GSM8K_PATH}")

    if not args.skip_comp_math:
        print("Translating competition_math...", flush=True)
        n_comp = export_competition_math(translator, cache, limit=args.limit_comp_math)
        print(f"Wrote {n_comp} rows to {COMP_MATH_PATH}")

    if not args.skip_qa:
        print(f"Translating MMLU (limit={args.limit_qa})...", flush=True)
        n_qa = export_mmlu(translator, cache, limit=args.limit_qa)
        print(f"Wrote {n_qa} rows to {QA_OUT}")

    print(f"Cache: {len(cache)} entries at {CACHE_PATH}")


if __name__ == "__main__":
    main()
