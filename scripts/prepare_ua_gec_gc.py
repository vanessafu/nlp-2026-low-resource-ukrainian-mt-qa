#!/usr/bin/env python3
"""Prepare UA-GEC (gec-only) for GC fine-tuning under train_data/GC."""

import argparse
import json
import re
import shutil
from pathlib import Path

from ua_gec import AnnotationLayer, Corpus


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "train_data" / "GC"
RAW_DIR = OUT_DIR / "raw" / "gec-only"
TRAIN_JSONL = OUT_DIR / "ua_gec_train.jsonl"
TRAIN_SINGLE_ERROR_JSONL = OUT_DIR / "ua_gec_train_single_error.jsonl"


def ua_gec_data_dir():
    import ua_gec

    return Path(ua_gec.__file__).resolve().parent / "data"


def copy_raw_data():
    src_root = ua_gec_data_dir()
    metadata_src = src_root / "metadata.csv"
    gec_src = src_root / "gec-only"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)

    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    shutil.copytree(gec_src, RAW_DIR)
    shutil.copy2(metadata_src, OUT_DIR / "metadata.csv")


def extract_error_pair(source: str, target: str):
    if source.strip() == target.strip():
        return "CORRECT", "CORRECT"

    src_words = re.findall(r"[\wА-Яа-яІіЇїЄєҐґ'’\-]+", source, flags=re.UNICODE)
    tgt_words = re.findall(r"[\wА-Яа-яІіЇїЄєҐґ'’\-]+", target, flags=re.UNICODE)

    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, src_words, tgt_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        wrong = " ".join(src_words[i1:i2]).strip()
        correct = " ".join(tgt_words[j1:j2]).strip()
        if wrong or correct:
            return wrong or correct, correct or wrong

    if source != target:
        matcher = SequenceMatcher(None, source, target)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            wrong = source[i1:i2].strip()
            correct = target[j1:j2].strip()
            if wrong or correct:
                return wrong or correct, correct or wrong

    return "CORRECT", "CORRECT"


def count_error_regions(source: str, target: str) -> int:
    if source.strip() == target.strip():
        return 0

    from difflib import SequenceMatcher

    src_words = re.findall(r"[\wА-Яа-яІіЇїЄєҐґ'’\-]+", source, flags=re.UNICODE)
    tgt_words = re.findall(r"[\wА-Яа-яІіЇїЄєҐґ'’\-]+", target, flags=re.UNICODE)
    matcher = SequenceMatcher(None, src_words, tgt_words)
    count = sum(1 for tag, _, _, _, _ in matcher.get_opcodes() if tag != "equal")
    if count:
        return count

    matcher = SequenceMatcher(None, source, target)
    return sum(1 for tag, _, _, _, _ in matcher.get_opcodes() if tag != "equal")


def export_train_jsonl():
    corpus = Corpus(partition="all", annotation_layer=AnnotationLayer.GecOnly)
    rows = []

    for doc in corpus:
        for sent_idx, (src_sent, tgt_sent) in enumerate(
            zip(doc.source_sentences, doc.target_sentences)
        ):
            incorrect_word, correct_word = extract_error_pair(src_sent, tgt_sent)
            num_errors = count_error_regions(src_sent, tgt_sent)
            rows.append(
                {
                    "dataset_id": "ua_gec_gc_train",
                    "id": f"{doc.doc_id}-a{doc.meta.annotator_id}-{sent_idx:05d}",
                    "doc_id": doc.doc_id,
                    "annotator_id": doc.meta.annotator_id,
                    "source_partition": doc.meta.partition,
                    "split": "train",
                    "input_sentence": src_sent,
                    "original_sentence": tgt_sent,
                    "incorrect_word": incorrect_word,
                    "correct_word": correct_word,
                    "num_errors": num_errors,
                }
            )

    single_error_rows = [row for row in rows if row["num_errors"] <= 1]

    with TRAIN_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with TRAIN_SINGLE_ERROR_JSONL.open("w", encoding="utf-8") as f:
        for row in single_error_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows), len(single_error_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-copy", action="store_true")
    args = parser.parse_args()

    if not args.skip_copy:
        copy_raw_data()

    n, n_single = export_train_jsonl()
    print(f"Wrote {n} training sentences to {TRAIN_JSONL}")
    print(f"Wrote {n_single} single-error sentences to {TRAIN_SINGLE_ERROR_JSONL}")
    print(f"Raw UA-GEC (gec-only) copied to {RAW_DIR}")


if __name__ == "__main__":
    main()
