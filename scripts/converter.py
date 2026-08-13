"""
Convert MT corpus TSV files → chat JSONL for fine-tuning.

Handles en-uk and cs-uk language pairs.
QA data is handled separately by scripts/prepare_qa_training.sh.

Usage
─────
  python scripts/converter.py                   # training + validation + test
  python scripts/converter.py --train-only
  python scripts/converter.py --val-only
  python scripts/converter.py --test-only
  python scripts/converter.py --dry-run         # stats only, no writes
"""

from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path

SEED = 42
random.seed(SEED)

_REPO   = Path(__file__).resolve().parent.parent
DATA    = _REPO / "data"
LLMS    = _REPO / "llms-limited-resources2025" / "Ukrainian"


# ══════════════════════════════════════════════════════════════════════════════
# MT — prompts / config
# ══════════════════════════════════════════════════════════════════════════════

_MT_SYS: dict[str, str] = {
    "en": (
        "You are a professional English-to-Ukrainian translator, tasked with providing "
        "translations suitable for use in Ukraine (uk_UA). Your goal is to accurately "
        "convey the meaning and nuances of the original English text while adhering to "
        "Ukrainian grammar, vocabulary, and cultural sensitivities. Produce only the "
        "Ukrainian translation, without any additional explanations or commentary. "
        "Retain the paragraph breaks (double new lines) from the input text. "
        "Please translate the following English text into Ukrainian (uk_UA):"
    ),
    "cs": (
        "You are a professional Czech-to-Ukrainian translator, tasked with providing "
        "translations suitable for use in Ukraine (uk_UA). Your goal is to accurately "
        "convey the meaning and nuances of the original Czech text while adhering to "
        "Ukrainian grammar, vocabulary, and cultural sensitivities. Produce only the "
        "Ukrainian translation, without any additional explanations or commentary. "
        "Retain the paragraph breaks (double new lines) from the input text. "
        "Please translate the following Czech text into Ukrainian (uk_UA):"
    ),
}
_MT_SYS["cz"] = _MT_SYS["cs"]
_MT_SYS_DEFAULT = (
    "You are a professional translator tasked with providing translations into "
    "Ukrainian (uk_UA). Produce only the Ukrainian translation, without any additional "
    "explanations or commentary. "
    "Retain the paragraph breaks (double new lines) from the input text. "
    "Please translate the following text into Ukrainian (uk_UA):"
)

MT_CAPS: dict[str, dict[str, int | None]] = {
    "cs-uk": {"nllb.tsv": 58_000 ,"elrc_acts.tsv": 2_000},
    "en-uk": {"nllb.tsv": 34_000,"dochplt.tsv": 20_000, "macocu.tsv": 64_000},
}

MT_EXCLUDED_VALIDATION: dict[str, set[str]] = {
    "en-uk": {"paracrawl_para01.tsv", "open_subtitles_para.tsv","tilde.tsv"},
    "cz-uk": {"open_subtitles_para.tsv",   "paracrawl_para.tsv","elrc_acts.tsv","nllb.tsv"},
}

MT_EXCLUDED_TRAIN: dict[str, set[str]] = {
    "en-uk": {"paracrawl_para01.tsv","open_subtitles.tsv","open_subtitles_para.tsv","dochplt copy.tsv"},
    "cs-uk": {"open_subtitles.tsv", "open_subtitles_para.tsv", "paracrawl_para.tsv","paracrawl.tsv"},
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

class _Sink:
    def write(self, _): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _read_tsv(path: Path, cap: int | None = None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    try:
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            si = header.index("source") if "source" in header else 0
            ti = header.index("target") if "target" in header else 1
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) > max(si, ti):
                    s, t = cols[si].strip(), cols[ti].strip()
                    if s and t:
                        pairs.append((s, t))
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"    [warn] {path.name}: {exc}")
        return []
    if cap and len(pairs) > cap:
        random.shuffle(pairs)
        pairs = pairs[:cap]
    return pairs


def _read_parallel(src: Path, tgt: Path) -> list[tuple[str, str]]:
    try:
        ss = src.read_text(encoding="utf-8").splitlines()
        ts = tgt.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as e:
        print(f"    [warn] {e}")
        return []
    return [(s.strip(), t.strip()) for s, t in zip(ss, ts) if s.strip() and t.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# MT — record builders
# ══════════════════════════════════════════════════════════════════════════════

def _mt_record(src_lang: str, source: str, target: str) -> dict:
    sys = _MT_SYS.get(src_lang, _MT_SYS_DEFAULT)
    return {"messages": [
        {"role": "system",    "content": sys},
        {"role": "user",      "content": source},
        {"role": "assistant", "content": target},
    ]}


# ══════════════════════════════════════════════════════════════════════════════
# MT — phase runners
# ══════════════════════════════════════════════════════════════════════════════

def _mt_train_dir(lang_dir: Path, dry_run: bool) -> int:
    dir_name = lang_dir.name
    src_lang = dir_name.split("-")[0]
    out_path = lang_dir / "train.jsonl"
    caps     = MT_CAPS.get(dir_name, {})
    excluded = MT_EXCLUDED_TRAIN.get(dir_name, set())
    tsvs     = sorted(lang_dir.glob("*.tsv"))
    if not tsvs:
        print(f"    [{dir_name}]  (no TSV files found)")
        return 0
    print(f"    [{dir_name}]")
    total = 0
    for tsv in tsvs:
        if tsv.name in excluded:
            continue
        pairs = _read_tsv(tsv, cap=caps.get(tsv.name))
        total += len(pairs)
        cap_s = f"cap={caps[tsv.name]:,}" if tsv.name in caps else "all"
        print(f"      {tsv.name:30s} {len(pairs):>8,}  ({cap_s})")
    print(f"      {'─'*40}  {total:>8,}")
    if not dry_run:
        with open(out_path, "w", encoding="utf-8") as f:
            for tsv in tsvs:
                if tsv.name in excluded:
                    continue
                for s, t in _read_tsv(tsv, cap=caps.get(tsv.name)):
                    f.write(json.dumps(_mt_record(src_lang, s, t), ensure_ascii=False) + "\n")
        print(f"      → {out_path}")
    return total


def run_training(dry_run: bool) -> None:
    print("\n═══ TRAINING ═══════════════════════════════════════════════════════")
    mt_dir = DATA / "training" / "MT"
    if not mt_dir.exists():
        print(f"  [skip] {mt_dir} not found")
        return
    grand = sum(_mt_train_dir(d, dry_run) for d in sorted(mt_dir.iterdir()) if d.is_dir())
    print(f"  MT grand total: {grand:,}")


def run_validation(dry_run: bool) -> None:
    print("\n═══ VALIDATION ══════════════════════════════════════════════════════")
    llms_mt = LLMS / "MT"
    configs = [
        ("en-uk", "en", llms_mt / "dev.en-uk.en", llms_mt / "dev.en-uk.uk"),
        ("cz-uk", "cs", llms_mt / "dev.cs-uk.cs", llms_mt / "dev.cs-uk.uk"),
    ]
    for dir_name, src_lang, dev_src, dev_tgt in configs:
        val_dir  = DATA / "validation" / "MT" / dir_name
        out_path = val_dir / "validation.jsonl"
        excluded = MT_EXCLUDED_VALIDATION.get(dir_name, set())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [{dir_name}]")
        total = 0
        with (open(out_path, "w", encoding="utf-8") if not dry_run else _Sink()) as f:
            for tsv in sorted(val_dir.glob("*.tsv")):
                if tsv.name in excluded:
                    continue
                pairs = _read_tsv(tsv)
                total += len(pairs)
                print(f"    {tsv.name:30s} {len(pairs):>6,}")
                if not dry_run:
                    for s, t in pairs:
                        f.write(json.dumps(_mt_record(src_lang, s, t), ensure_ascii=False) + "\n")
            dev_pairs = _read_parallel(dev_src, dev_tgt)
            total += len(dev_pairs)
            lbl = "dev." + dir_name.replace("cz", "cs") + " (llms)"
            print(f"    {lbl:30s} {len(dev_pairs):>6,}")
            if not dry_run:
                for s, t in dev_pairs:
                    f.write(json.dumps(_mt_record(src_lang, s, t), ensure_ascii=False) + "\n")
        print(f"    {'─'*38}  {total:>6,}")
        if not dry_run:
            print(f"    → {out_path}")


def run_test(dry_run: bool) -> None:
    print("\n═══ TEST ════════════════════════════════════════════════════════════")
    llms_mt = LLMS / "MT"
    for src, dest_dir in [
        (llms_mt / "test.en_uk.jsonl", DATA / "test" / "MT" / "en-uk"),
        (llms_mt / "test.cs_uk.jsonl", DATA / "test" / "MT" / "cz-uk"),
    ]:
        if not src.exists():
            print(f"  [skip] {src.name} not found")
            continue
        n = sum(1 for _ in open(src, encoding="utf-8"))
        print(f"  {src.name}  ({n:,} entries)")
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / src.name)
            print(f"    → {dest_dir / src.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Convert MT corpus TSV files → chat JSONL for fine-tuning"
    )
    phase = p.add_mutually_exclusive_group()
    phase.add_argument("--train-only", action="store_true")
    phase.add_argument("--val-only",   action="store_true")
    phase.add_argument("--test-only",  action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print statistics without writing any files")
    args = p.parse_args(argv)

    if args.dry_run:
        print("  [DRY RUN — no files written]")

    do_train = not (args.val_only  or args.test_only)
    do_val   = not (args.train_only or args.test_only)
    do_test  = not (args.train_only or args.val_only)

    if do_train: run_training(args.dry_run)
    if do_val:   run_validation(args.dry_run)
    if do_test:  run_test(args.dry_run)

    print("\n  Done.")


if __name__ == "__main__":
    main()
