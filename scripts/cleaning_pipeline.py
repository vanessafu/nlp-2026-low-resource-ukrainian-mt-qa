"""
9  stages :
  1  Unicode normalization
  2  Advanced deduplication  (Bloom exact-dedup + MinHash near-dedup + Union-Find)
  3  Language identification  (fasttext lid.176)
  4  Ukrainian quality scoring  (Ru contamination, script analysis)
  5  Structural filtering  (length, ratio, punctuation — statistics-calibrated)
  6  Entity / numeric consistency  (numbers, URLs, brackets via regex)
  7  Copy-source / untranslation detection
  8  Semantic alignment scoring  (multilingual-e5-small, ONNX/int8 optional)
  9  QE-style quality scoring  (PyMarian; composite fallback when unavailable)

Usage
  # Full pipeline
  python scripts/cleaning_pipeline.py \\
      --input  data/training/MT/en-uk/macocu.tsv \\
      --output data/training/MT/en-uk/macocu_clean.tsv \\
      --lang-pair en-uk --stages all --workers 8

  # Single stage (Stage 3 only)
  python scripts/cleaning_pipeline.py \\
      --input raw.tsv --output stage3.tsv --lang-pair en-uk --stages 3

  # Calibrate structural thresholds from data, then filter
  python scripts/cleaning_pipeline.py \\
      --input raw.tsv --output filtered.tsv \\
      --lang-pair en-uk --calibrate --stages 5

  # Statistics only (no filtering)
  python scripts/cleaning_pipeline.py \\
      --input raw.tsv --lang-pair en-uk --stats-only
"""

from __future__ import annotations

import argparse
import array
import json
import logging
import math
import multiprocessing
import os
import re
import struct
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Iterator, NamedTuple, Optional

# ── polars required ───────────────────────────────────────────────────────────
try:
    import polars as pl
except ImportError:
    sys.exit("polars is required: pip install polars")

# ── lazy optional imports (checked at stage init time) ────────────────────────
# datasketch, mmh3, fasttext, sentence_transformers, pymarian, optimum
# → imported inside each Stage that needs them

logger = logging.getLogger("cleaning_filtering")

# =============================================================================
# CONSTANTS
# =============================================================================

# Ukrainian-specific Cyrillic (absent from Russian standard orthography)
_UK_ONLY: frozenset[str] = frozenset("іїєґІЇЄҐ")
# Characters that indicate Russian (should NOT appear in standard Ukrainian)
_RU_MARKERS: frozenset[str] = frozenset("ыэъёЫЭЪЁ")

# High-frequency Russian function words that differ from Ukrainian equivalents
_RU_FUNC: frozenset[str] = frozenset({
    "что", "это", "его", "они", "был", "была", "были", "или", "бы",
    "же", "так", "уже", "ещё", "тоже", "тогда", "потому", "вот",
    "всё", "ведь", "даже", "нет", "там", "здесь", "сейчас", "когда",
    "если", "чтобы", "после", "перед", "между", "через", "только",
    "очень", "какой", "такой", "который", "другой", "сам", "себя",
})

# Compiled patterns (module-level for multiprocessing fork efficiency)
_RE_WHITESPACE    = re.compile(r"[ \t\r\f\v]+")
_RE_ZERO_WIDTH    = re.compile(r"[​‌‍\u200E\u200F﻿­]")
_RE_CTRL          = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_NUMBERS       = re.compile(r"\b\d[\d\s,.']*\d|\b\d\b")
_RE_URL           = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL         = re.compile(r"[\w.+-]+@[\w.-]+\.\w{2,}")
_RE_BRACKETS      = re.compile(r"[(){}\[\]<>]")
_RE_PERCENT       = re.compile(r"\d+\s*%")
_RE_DATE          = re.compile(r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b")
_RE_LATIN_RUN     = re.compile(r"[A-Za-z]{2,}")
_RE_CYRILLIC_CHAR = re.compile(r"[А-яҐґЄєІіЇї]")

# Per-language-pair structural defaults (overridden by calibration)
_DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "en-uk": {
        "min_src_chars": 10.0, "max_src_chars": 1000.0,
        "min_tgt_chars": 5.0,  "max_tgt_chars": 1200.0,
        "char_ratio_min": 0.50, "char_ratio_max": 2.20,
        "word_ratio_min": 0.25, "word_ratio_max": 4.00,
    },
    "cs-uk": {
        "min_src_chars": 10.0, "max_src_chars": 1000.0,
        "min_tgt_chars": 5.0,  "max_tgt_chars": 1200.0,
        "char_ratio_min": 0.60, "char_ratio_max": 1.80,
        "word_ratio_min": 0.30, "word_ratio_max": 3.50,
    },
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Pair(NamedTuple):
    source: str
    target: str
    meta:   dict[str, Any] = {}


@dataclass
class StageStats:
    stage_name:  str
    n_in:        int = 0
    n_out:       int = 0
    n_rejected:  int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def retention_rate(self) -> float:
        return self.n_out / self.n_in if self.n_in else 0.0

    def __str__(self) -> str:
        pct = self.retention_rate * 100
        reasons = "  ".join(f"{k}={v}" for k, v in self.reject_reasons.items())
        return (
            f"[{self.stage_name}]  in={self.n_in:,}  out={self.n_out:,}"
            f"  retained={pct:.1f}%  {reasons}"
        )


@dataclass
class StatsCollector:
    """
    Accumulates per-stage statistics, prints minimal step markers to stdout,
    and saves a structured JSON report to data/stats/{lang_pair}/{dataset}.json.
    """

    lang_pair: str
    dataset:   str
    stats_dir: Path
    _stages:   list[StageStats] = field(default_factory=list)

    def record(self, stats: StageStats) -> None:
        self._stages.append(stats)
        pct = stats.retention_rate * 100
        print(f"  → [{stats.stage_name}]  {stats.n_in:,} → {stats.n_out:,}  ({pct:.1f}%)")

    def save(self) -> Path:
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.stats_dir / f"{self.dataset}.json"
        data: dict = {
            "lang_pair": self.lang_pair,
            "dataset":   self.dataset,
            "stages": [
                {
                    "stage":          s.stage_name,
                    "n_in":           s.n_in,
                    "n_out":          s.n_out,
                    "n_rejected":     s.n_rejected,
                    "retention_pct":  round(s.retention_rate * 100, 2),
                    "reject_reasons": s.reject_reasons,
                }
                for s in self._stages
            ],
        }
        if self._stages:
            data["total_in"]  = self._stages[0].n_in
            data["total_out"] = self._stages[-1].n_out
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path


@dataclass
class PipelineConfig:
    lang_pair:      str   = "en-uk"
    stages:         list[int] = field(default_factory=lambda: list(range(1, 10)))
    workers:        int   = 4
    chunk_size:     int   = 100_000
    seed:           int   = 42
    thresholds:     dict  = field(default_factory=dict)  # per-lang overrides
    # Stage 2
    dedup_exact:    bool  = True
    dedup_near:     bool  = True
    minhash_threshold: float = 0.85
    minhash_perms:  int   = 128
    bloom_capacity: int   = 50_000_000
    bloom_fpr:      float = 0.001
    # Stage 3
    fasttext_model: Optional[str] = None   # path to lid.176.bin; auto-downloaded if None
    lid_threshold:  float = 0.70
    # Stage 4
    uk_quality_min: float = 0.0            # reject if score < this; 0 = off
    # Stage 8
    embed_model:    str   = "intfloat/multilingual-e5-small"
    embed_batch:    int   = 512
    embed_sim_min:  float = 0.60
    use_onnx:       bool  = True           # try ONNX/int8; fall back to torch
    # Stage 9
    qe_model:       Optional[str] = None   # path to PyMarian model dir
    composite_min:  float = 0.50
    composite_weights: dict = field(default_factory=lambda: {
        "semantic": 0.35,
        "uk_quality": 0.30,
        "structural": 0.20,
        "qe": 0.15,
    })
    # Audit
    audit_sample_rate: float = 0.001       # fraction of rejected pairs to log
    audit_dir:  Optional[Path] = None

    def get_thresholds(self) -> dict[str, float]:
        base = _DEFAULT_THRESHOLDS.get(self.lang_pair, _DEFAULT_THRESHOLDS["en-uk"]).copy()
        base.update(self.thresholds)
        return base


# =============================================================================
# UTILITIES
# =============================================================================

def print_score_stats(df: pl.DataFrame, cols: list[str], label: str = "") -> None:
    """
    Print mean, median, p25, p75 for numeric columns.
    Used to calibrate filtering thresholds.
    Call this BEFORE applying filters.
    """
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return
    try:
        stats = (
            df.select([pl.col(c).cast(pl.Float64, strict=False) for c in existing])
            .describe()
        )
        def _val(stat_name: str, col: str) -> float:
            row = stats.filter(pl.col("statistic") == stat_name)
            if len(row) == 0 or row[col][0] is None:
                return float("nan")
            return float(row[col][0])

        prefix = f"[{label}] " if label else ""
        print(f"  {prefix}n={len(df):,}")
        print(f"    {'column':<32} {'mean':>10} {'p25':>10} {'p50':>10} {'p75':>10} {'min':>10} {'max':>10}")
        print("    " + "─" * 84)
        for c in existing:
            print(
                f"    {c:<32}"
                f" {_val('mean', c):>10.4f}"
                f" {_val('25%',  c):>10.4f}"
                f" {_val('50%',  c):>10.4f}"
                f" {_val('75%',  c):>10.4f}"
                f" {_val('min',  c):>10.4f}"
                f" {_val('max',  c):>10.4f}"
            )
    except Exception as exc:
        logger.warning("print_score_stats error: %s", exc)


def _chunk_df(df: pl.DataFrame, size: int) -> Generator[pl.DataFrame, None, None]:
    for start in range(0, len(df), size):
        yield df.slice(start, size)


def _read_tsv_chunks(
    path: Path, chunk_size: int
) -> Generator[pl.DataFrame, None, None]:
    """Stream a source/target TSV in chunks without loading it all at once."""
    lf = pl.scan_csv(path, separator="\t", has_header=True, infer_schema_length=1000)
    total = lf.select(pl.len()).collect().item()
    for start in range(0, total, chunk_size):
        yield lf.slice(start, chunk_size).collect()


def _append_tsv(df: pl.DataFrame, path: Path, write_header: bool) -> None:
    mode = "w" if write_header else "a"
    with open(path, mode, encoding="utf-8") as f:
        if write_header:
            f.write("source\ttarget\n")
        for row in df.iter_rows():
            f.write(f"{row[0]}\t{row[1]}\n")


# =============================================================================
# BLOOM FILTER  (exact dedup, no external dependencies)
# =============================================================================

class BloomFilter:
    """
    Simple memory-efficient Bloom filter using mmh3 + stdlib array.

    Uses mmh3 (fast Murmur3 hashing). Install: pip install mmh3
    Falls back to hashlib if mmh3 unavailable (slower).

    Capacity 50M at FPR 0.001 ≈ 860 MB  (reasonable for large-RAM CPU nodes).
    """

    def __init__(self, capacity: int, fpr: float = 0.001) -> None:
        self._n_bits  = self._optimal_bits(capacity, fpr)
        self._n_hash  = self._optimal_hash_count(self._n_bits, capacity)
        self._bits    = array.array("B", [0]) * ((self._n_bits + 7) // 8)
        self._use_mmh3 = True
        try:
            import mmh3  # noqa: F401
            self._mmh3 = mmh3
        except ImportError:
            import hashlib
            self._hashlib = hashlib
            self._use_mmh3 = False
            logger.warning(
                "mmh3 not found; Bloom filter will use hashlib (slower). "
                "Install: pip install mmh3"
            )

    @staticmethod
    def _optimal_bits(n: int, p: float) -> int:
        return max(1, int(-n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        return max(1, round((m / n) * math.log(2)))

    def _hashes(self, item: str) -> list[int]:
        if self._use_mmh3:
            return [
                abs(self._mmh3.hash(item, seed=i)) % self._n_bits
                for i in range(self._n_hash)
            ]
        import hashlib
        h = hashlib.sha256(item.encode()).digest()
        base = int.from_bytes(h[:8], "big")
        inc  = int.from_bytes(h[8:16], "big")
        return [(base + i * inc) % self._n_bits for i in range(self._n_hash)]

    def _set_bit(self, idx: int) -> None:
        self._bits[idx >> 3] |= 1 << (idx & 7)

    def _get_bit(self, idx: int) -> bool:
        return bool(self._bits[idx >> 3] & (1 << (idx & 7)))

    def add(self, item: str) -> None:
        for h in self._hashes(item):
            self._set_bit(h)

    def __contains__(self, item: str) -> bool:
        return all(self._get_bit(h) for h in self._hashes(item))

    def __sizeof__(self) -> int:
        return len(self._bits) + object.__sizeof__(self)


# =============================================================================
# UNION-FIND  (for near-dedup clustering)
# =============================================================================

class UnionFind:
    """Path-compressed, union-by-rank Union-Find."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank:   dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x]   = 0
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            nxt = self._parent[x]
            self._parent[x] = root
            x = nxt
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def clusters(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for node in self._parent:
            result[self.find(node)].append(node)
        return dict(result)


# =============================================================================
# STAGE BASE
# =============================================================================

class FilterStage:
    """
    Abstract base. Each stage receives a pl.DataFrame with 'source'/'target'
    columns and returns a filtered pl.DataFrame plus StageStats.
    """

    name: str = "base"

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def setup(self) -> None:
        """Called once before processing begins (load models, etc.)."""

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        raise NotImplementedError

    def _stats(self, n_in: int, df_out: pl.DataFrame, **reasons: int) -> StageStats:
        s = StageStats(self.name, n_in=n_in, n_out=len(df_out))
        s.n_rejected = n_in - len(df_out)
        s.reject_reasons = {k: v for k, v in reasons.items() if v > 0}
        return s


# =============================================================================
# STAGE 1  —  Unicode normalization
# =============================================================================

def unicode_normalize_pair(src: str, tgt: str) -> tuple[str, str]:
    """
    Normalize a single pair.  Callable standalone.

    Steps:
    1. NFC normalization (canonical decomposition → re-composition)
    2. Remove zero-width / soft-hyphen chars
    3. Remove non-printable control chars (keep \\n\\t)
    4. Collapse whitespace runs to single space, strip
    5. Ukrainian apostrophe normalization: ʼ → '  (U+02BC → U+0027)
    """
    def _clean(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = _RE_ZERO_WIDTH.sub("", text)
        text = _RE_CTRL.sub("", text)
        text = text.replace("ʼ", "'")  # Ukrainian modifier apostrophe
        text = text.replace("’", "'")  # right single quotation mark
        text = _RE_WHITESPACE.sub(" ", text).strip()
        return text

    return _clean(src), _clean(tgt)


class UnicodeNormStage(FilterStage):
    """
    Stage 1: Unicode normalization.

    Why: Raw parallel data contains zero-width characters, inconsistent
    apostrophes, and control characters that cause tokenization failures
    and inflate vocabulary.  NFC normalization also ensures consistent
    representation of composed vs decomposed characters.
    """

    name = "S1_unicode"

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in = len(df)
        srcs, tgts = [], []
        for src, tgt in zip(df["source"].to_list(), df["target"].to_list()):
            s, t = unicode_normalize_pair(src or "", tgt or "")
            srcs.append(s)
            tgts.append(t)

        df = df.with_columns(
            pl.Series("source", srcs),
            pl.Series("target", tgts),
        ).filter(
            (pl.col("source").str.len_chars() > 0)
            & (pl.col("target").str.len_chars() > 0)
        )
        return df, self._stats(n_in, df, empty_after_norm=n_in - len(df))


# =============================================================================
# STAGE 2  —  Deduplication
# =============================================================================

def _char_ngrams(text: str, n: int = 5) -> set[str]:
    t = text.lower()
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def _minhash_signature(ngrams: set[str], n_perms: int = 128) -> list[int]:
    """
    Lightweight MinHash without datasketch — uses double-hashing trick.
    Uses mmh3 if available; falls back to hashlib.
    """
    try:
        import mmh3
        _h = lambda item, seed: abs(mmh3.hash(item, seed=seed))
    except ImportError:
        import hashlib
        def _h(item: str, seed: int) -> int:
            digest = hashlib.md5(f"{seed}:{item}".encode()).digest()
            return int.from_bytes(digest[:8], "big")

    sigs = [min(_h(gram, i) for gram in ngrams) if ngrams else 0
            for i in range(n_perms)]
    return sigs


def _jaccard_from_sigs(a: list[int], b: list[int]) -> float:
    matches = sum(x == y for x, y in zip(a, b))
    return matches / len(a) if a else 0.0


def near_dedup_shard(
    texts: list[str],
    threshold: float = 0.85,
    n_perms: int = 128,
) -> set[int]:
    """
    Return the set of indices to KEEP after near-dedup within a shard.

    Algorithm:
    1. Compute MinHash signatures for all texts.
    2. Build LSH index (b bands × r rows).  b=32, r=4 → threshold ≈ 0.82.
    3. Find candidate duplicate pairs via band matching.
    4. Verify candidate pairs with Jaccard similarity (from signatures).
    5. Union-Find cluster + keep representative (longest text in cluster).

    Tradeoff: shard-local only → cross-shard duplicates need separate pass.
    """
    try:
        from datasketch import MinHash, MinHashLSH  # preferred if installed

        lsh = MinHashLSH(threshold=threshold, num_perm=n_perms)
        minhashes: list[MinHash] = []
        for i, text in enumerate(texts):
            m = MinHash(num_perm=n_perms)
            for gram in _char_ngrams(text):
                m.update(gram.encode())
            minhashes.append(m)
            try:
                lsh.insert(str(i), m)
            except ValueError:
                pass  # duplicate key (exact match already)

        uf = UnionFind()
        for i, m in enumerate(minhashes):
            for key in lsh.query(m):
                j = int(key)
                if j != i:
                    uf.union(i, j)

    except ImportError:
        # Fallback: manual MinHash without datasketch
        sigs = [_minhash_signature(_char_ngrams(t), n_perms) for t in texts]
        # Band-based LSH: b=32 bands of r=4 rows → P(collision) ≈ threshold
        b, r = 32, n_perms // 32
        band_buckets: dict[tuple, list[int]] = defaultdict(list)
        for i, sig in enumerate(sigs):
            for band in range(b):
                key = (band, tuple(sig[band * r : band * r + r]))
                band_buckets[key].append(i)

        uf = UnionFind()
        for bucket in band_buckets.values():
            if len(bucket) < 2:
                continue
            for a in range(len(bucket)):
                for bx in range(a + 1, len(bucket)):
                    i, j = bucket[a], bucket[bx]
                    if _jaccard_from_sigs(sigs[i], sigs[j]) >= threshold:
                        uf.union(i, j)

    # Keep the longest text in each cluster (heuristic: longer = more complete)
    keep: set[int] = set()
    for cluster in uf.clusters().values():
        best = max(cluster, key=lambda idx: len(texts[idx]))
        keep.add(best)
    # Texts not involved in any near-dup are not in UnionFind → keep them
    all_ids = set(range(len(texts)))
    in_cluster = {i for cluster in uf.clusters().values() for i in cluster}
    keep |= all_ids - in_cluster
    return keep


class DeduplicationStage(FilterStage):
    """
    Stage 2: Exact + near deduplication.

    Exact dedup:  Bloom filter on (source + '|||' + target) — sub-millisecond,
                  constant memory, cross-chunk persistent.

    Near dedup:   MinHash LSH within each processing shard.  Uses datasketch
                  (pip install datasketch) when available; falls back to
                  built-in implementation.

    Why near-dedup matters: parallel corpora often contain paraphrastic
    near-duplicates that inflate training signal without adding diversity.
    Training on near-duplicates inflates BLEU but hurts generalization.

    Failure modes: shard-local near-dedup misses cross-shard near-dups.
    For truly exhaustive near-dedup, process all data in one shard (large RAM)
    or use a two-pass sorted approach.
    """

    name = "S2_dedup"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._bloom: Optional[BloomFilter] = None

    def setup(self) -> None:
        if self.cfg.dedup_exact:
            self._bloom = BloomFilter(
                capacity=self.cfg.bloom_capacity,
                fpr=self.cfg.bloom_fpr,
            )
            logger.info(
                "Bloom filter init: capacity=%d  FPR=%.4f  mem≈%.0f MB",
                self.cfg.bloom_capacity,
                self.cfg.bloom_fpr,
                self._bloom.__sizeof__() / 1e6,
            )

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in = len(df)
        srcs = df["source"].to_list()
        tgts = df["target"].to_list()
        n_exact = n_near = 0

        # ── exact dedup via Bloom ─────────────────────────────────────────────
        keep_mask = [True] * len(srcs)
        if self.cfg.dedup_exact and self._bloom is not None:
            for i, (s, t) in enumerate(zip(srcs, tgts)):
                key = f"{s}|||{t}"
                if key in self._bloom:
                    keep_mask[i] = False
                    n_exact += 1
                else:
                    self._bloom.add(key)

        df = df.filter(pl.Series("_bloom_keep", keep_mask, dtype=pl.Boolean))
        srcs = df["source"].to_list()

        # ── near dedup (shard-local) ───────────────────────────────────────────
        if self.cfg.dedup_near and len(srcs) > 1:
            keep_indices = near_dedup_shard(
                srcs,
                threshold=self.cfg.minhash_threshold,
                n_perms=self.cfg.minhash_perms,
            )
            n_near = len(srcs) - len(keep_indices)
            keep_sorted = sorted(keep_indices)
            df = df[keep_sorted]

        return df, self._stats(n_in, df, exact_dup=n_exact, near_dup=n_near)


# =============================================================================
# STAGE 3  —  Language identification
# =============================================================================

def check_language(text: str, expected_lang: str, model: Any, threshold: float = 0.70) -> bool:
    """
    Return True if `text` is detected as `expected_lang` with confidence ≥ threshold.
    `model` is a loaded fasttext model (fasttext.load_model(path)).
    Callable standalone.
    """
    predictions = model.predict(text.replace("\n", " "), k=1)
    label: str = predictions[0][0].replace("__label__", "")
    confidence: float = float(predictions[1][0])
    return label == expected_lang and confidence >= threshold


_FASTTEXT_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)


def _ensure_fasttext_model(path: Optional[str] = None) -> str:
    """Download lid.176.bin if not found; return the local path."""
    if path and Path(path).is_file():
        return path
    default_path = Path.home() / ".cache" / "fasttext" / "lid.176.bin"
    if default_path.is_file():
        return str(default_path)
    default_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading fasttext lid.176.bin → %s", default_path)
    import urllib.request
    urllib.request.urlretrieve(_FASTTEXT_URL, default_path)
    return str(default_path)


class LangIDStage(FilterStage):
    """
    Stage 3: Language identification.

    Uses fasttext lid.176 model (176 languages, ~130 MB, runs fast on CPU).
    Verifies:
      - source is English ('en') or Czech ('cs') depending on lang_pair
      - target is Ukrainian ('uk')

    Why: OPUS and web-crawled corpora frequently contain wrong-language pairs
    (e.g., Belarusian or Bulgarian passed as Ukrainian).  LID catches these
    before expensive semantic scoring.

    CPU performance: ~50k sentences/sec per thread on modern hardware.
    """

    name = "S3_langid"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._src_lang = config.lang_pair.split("-")[0]   # 'en' or 'cs'

    def setup(self) -> None:
        try:
            import fasttext
            fasttext.FastText.eprint = lambda *args, **kwargs: None  # suppress stderr
            model_path = _ensure_fasttext_model(self.cfg.fasttext_model)
            self._model = fasttext.load_model(model_path)
            logger.info("FastText model loaded: %s", model_path)
        except ImportError:
            logger.warning(
                "fasttext not installed. Stage 3 will be skipped. "
                "Install: pip install fasttext-langdetect"
            )

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        if self._model is None:
            logger.warning("Stage 3 skipped (fasttext unavailable).")
            return df, StageStats(self.name, n_in=len(df), n_out=len(df))

        n_in   = len(df)
        thresh = self.cfg.lid_threshold
        src_lang, tgt_lang = self._src_lang, "uk"

        keep = []
        n_bad_src = n_bad_tgt = 0
        for src, tgt in zip(df["source"].to_list(), df["target"].to_list()):
            ok_src = check_language(src, src_lang, self._model, thresh)
            ok_tgt = check_language(tgt, tgt_lang, self._model, thresh)
            if not ok_src:
                n_bad_src += 1
            elif not ok_tgt:
                n_bad_tgt += 1
            keep.append(ok_src and ok_tgt)

        df = df.filter(pl.Series("_keep_lid", keep))
        return df, self._stats(n_in, df, wrong_src_lang=n_bad_src, wrong_tgt_lang=n_bad_tgt)


# =============================================================================
# STAGE 4  —  Ukrainian quality scoring
# =============================================================================

def compute_uk_quality_score(text: str) -> float:
    """
    Score Ukrainian text quality in [0, 1].  Callable standalone.

    Components
    ----------
    1. uk_char_ratio   : fraction of Cyrillic chars that are Ukrainian-specific
                         (і, ї, є, ґ).  Higher = more Ukrainian.
    2. ru_marker_ratio : fraction of Cyrillic chars that are Russian markers
                         (ы, э, ъ, ё).  Higher = more contamination.
    3. ru_func_penalty : fraction of words that are Russian function words.

    Score = uk_char_ratio * 0.5  +  (1 - ru_marker_ratio) * 0.3
            + (1 - ru_func_penalty) * 0.2

    Why weighted scoring vs simple rule: Russian and Ukrainian share 70%+ of
    their vocabulary, so individual word checks are unreliable.  A weighted
    multi-feature approach is more robust.

    Failure modes: short sentences have noisy ratios.  Minimum text length
    of ~20 chars is recommended before applying this score.
    """
    cyrillic = _RE_CYRILLIC_CHAR.findall(text)
    n_cyr = len(cyrillic)
    if n_cyr == 0:
        return 0.5  # no Cyrillic = ambiguous (possibly all-Latin named entities)

    uk_count = sum(1 for c in cyrillic if c in _UK_ONLY)
    ru_count = sum(1 for c in cyrillic if c in _RU_MARKERS)

    uk_char_ratio   = uk_count / n_cyr
    ru_marker_ratio = min(1.0, ru_count / n_cyr * 5)  # amplify: even 1 ы is bad

    words = text.lower().split()
    if words:
        ru_func_count = sum(1 for w in words if w in _RU_FUNC)
        ru_func_penalty = min(1.0, ru_func_count / max(1, len(words)) * 10)
    else:
        ru_func_penalty = 0.0

    score = (
        uk_char_ratio    * 0.50
        + (1.0 - ru_marker_ratio)  * 0.30
        + (1.0 - ru_func_penalty)  * 0.20
    )
    return max(0.0, min(1.0, score))


class UkrainianQualityStage(FilterStage):
    """
    Stage 4: Ukrainian quality filtering.

    Scores the TARGET side for Ukrainian-specific features and penalizes
    Russian contamination.  Filtering is applied only when uk_quality_min > 0.

    Expected score distribution on clean Ukrainian: 0.6 – 0.9.
    Russian text: 0.1 – 0.35.
    Mixed: 0.3 – 0.6.

    Failure modes: proper nouns, code-switching, quoted Russian text can
    lower the score.  Set threshold conservatively (≥ 0.3) for broad corpora.
    """

    name = "S4_uk_quality"

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in = len(df)
        scores = [compute_uk_quality_score(t) for t in df["target"].to_list()]
        df = df.with_columns(pl.Series("_uk_score", scores))

        n_rejected = 0
        min_score = self.cfg.uk_quality_min
        if min_score > 0:
            mask = pl.Series("_uk_ok", [s >= min_score for s in scores])
            n_rejected = int((~mask).sum())
            df = df.filter(mask)

        # Keep score column for composite scoring in Stage 9
        df = df.rename({"_uk_score": "uk_quality_score"}) \
            if "_uk_score" in df.columns else df
        return df, self._stats(n_in, df, low_uk_quality=n_rejected)


# =============================================================================
# STAGE 5  —  Structural filtering
# =============================================================================

def compute_structural_score(
    src: str,
    tgt: str,
    thresholds: dict[str, float],
) -> tuple[bool, str, float]:
    """
    Return (passes, reject_reason, score_0_to_1).  Callable standalone.

    Rules applied (in order):
    1. Character length bounds (src and tgt)
    2. Character ratio src/tgt
    3. Word count ratio
    4. Punctuation consistency (mismatched terminal punctuation)
    5. Bracket balance consistency
    6. Quote balance consistency

    The score is a soft measure of how well the pair passes structural checks.
    """
    src_chars = len(src)
    tgt_chars = len(tgt)
    if src_chars < thresholds["min_src_chars"]:
        return False, "src_too_short", 0.0
    if src_chars > thresholds["max_src_chars"]:
        return False, "src_too_long", 0.0
    if tgt_chars < thresholds["min_tgt_chars"]:
        return False, "tgt_too_short", 0.0
    if tgt_chars > thresholds["max_tgt_chars"]:
        return False, "tgt_too_long", 0.0

    char_ratio = src_chars / tgt_chars
    if not (thresholds["char_ratio_min"] <= char_ratio <= thresholds["char_ratio_max"]):
        return False, "char_ratio", 0.0

    src_words = len(src.split())
    tgt_words = len(tgt.split())
    if tgt_words > 0:
        word_ratio = src_words / tgt_words
        if not (thresholds["word_ratio_min"] <= word_ratio <= thresholds["word_ratio_max"]):
            return False, "word_ratio", 0.0

    # Soft structural score components
    scores: list[float] = []

    # Punctuation consistency: last non-space char
    src_end = src.rstrip()[-1] if src.rstrip() else ""
    tgt_end = tgt.rstrip()[-1] if tgt.rstrip() else ""
    terminal_ok = (
        (src_end in ".!?") == (tgt_end in ".!?")
        or src_end not in ".!?"
    )
    scores.append(1.0 if terminal_ok else 0.5)

    # Bracket balance: count opening vs closing; consistent mismatch
    def _bracket_balance(text: str) -> int:
        return text.count("(") - text.count(")")
    src_bal = abs(_bracket_balance(src))
    tgt_bal = abs(_bracket_balance(tgt))
    scores.append(1.0 if src_bal == 0 and tgt_bal == 0 else
                  0.8 if src_bal == tgt_bal else 0.4)

    # Quote consistency
    def _quote_count(text: str) -> int:
        return text.count('"') + text.count("'") + text.count("«") + text.count("»")
    q_diff = abs(_quote_count(src) - _quote_count(tgt))
    scores.append(1.0 if q_diff == 0 else 0.7 if q_diff <= 1 else 0.3)

    structural_score = sum(scores) / len(scores)
    return True, "", structural_score


def calibrate_structural_thresholds(
    df: pl.DataFrame,
    lang_pair: str,
    lo_pct: float = 2.5,
    hi_pct: float = 97.5,
) -> dict[str, float]:
    """
    Compute statistics-based thresholds from the actual data distribution.

    Keeps the middle (100 - 2*lo_pct)% of each distribution.
    Returns a threshold dict compatible with DEFAULT_THRESHOLDS.

    Why data-driven: corpus-specific length distributions vary significantly
    (subtitle data has very different char lengths than legal/news corpora).
    Fixed thresholds lose too much data or keep too much noise.
    """
    df_stats = df.with_columns(
        pl.col("source").str.len_chars().alias("_src_chars"),
        pl.col("target").str.len_chars().alias("_tgt_chars"),
        (pl.col("source").str.len_chars().cast(pl.Float64)
         / pl.col("target").str.len_chars().cast(pl.Float64).clip(1)).alias("_char_ratio"),
        (pl.col("source").str.split(" ").list.len().cast(pl.Float64)
         / pl.col("target").str.split(" ").list.len().cast(pl.Float64).clip(1)).alias("_word_ratio"),
    )

    def _pct(col: str, p: float) -> float:
        return float(df_stats[col].quantile(p / 100, interpolation="nearest"))

    base = _DEFAULT_THRESHOLDS.get(lang_pair, _DEFAULT_THRESHOLDS["en-uk"]).copy()
    base.update({
        "min_src_chars": _pct("_src_chars", lo_pct),
        "max_src_chars": _pct("_src_chars", hi_pct),
        "min_tgt_chars": _pct("_tgt_chars", lo_pct),
        "max_tgt_chars": _pct("_tgt_chars", hi_pct),
        "char_ratio_min": _pct("_char_ratio", lo_pct),
        "char_ratio_max": _pct("_char_ratio", hi_pct),
        "word_ratio_min": _pct("_word_ratio", lo_pct),
        "word_ratio_max": _pct("_word_ratio", hi_pct),
    })
    logger.info("Calibrated thresholds for %s: %s", lang_pair, base)
    return base


class StructuralFilterStage(FilterStage):
    """
    Stage 5: Structural quality filtering.

    Applies length, ratio, punctuation, and bracket checks.
    Supports calibration mode (compute thresholds from data distribution).

    Performance: fully vectorized via Polars expressions for the length/ratio
    checks; per-pair Python loop only for punctuation/bracket checks.
    Scales to 100M+ pairs on a CPU node.
    """

    name = "S5_structural"

    def __init__(self, config: PipelineConfig, calibrated: Optional[dict] = None) -> None:
        super().__init__(config)
        self._thresh: dict[str, float] = calibrated or config.get_thresholds()

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in = len(df)
        thresh = self._thresh

        # Fast vectorized pre-filter (eliminates most rejects)
        df = (
            df.with_columns(
                pl.col("source").str.len_chars().alias("_sc"),
                pl.col("target").str.len_chars().alias("_tc"),
            )
            .filter(
                pl.col("_sc").is_between(int(thresh["min_src_chars"]),
                                         int(thresh["max_src_chars"]))
                & pl.col("_tc").is_between(int(thresh["min_tgt_chars"]),
                                            int(thresh["max_tgt_chars"]))
                & (pl.col("_sc").cast(pl.Float64) / pl.col("_tc").cast(pl.Float64))
                  .is_between(thresh["char_ratio_min"], thresh["char_ratio_max"])
            )
            .drop(["_sc", "_tc"])
        )

        # Per-pair punctuation / bracket / quote checks + structural score
        keeps, struct_scores = [], []
        reasons: dict[str, int] = defaultdict(int)
        for src, tgt in zip(df["source"].to_list(), df["target"].to_list()):
            ok, reason, score = compute_structural_score(src, tgt, thresh)
            keeps.append(ok)
            struct_scores.append(score)
            if not ok:
                reasons[reason] += 1

        df = df.filter(pl.Series("_sk", keeps))
        # keep structural score for Stage 9 composite
        struct_scores = [s for s, k in zip(struct_scores, keeps) if k]
        df = df.with_columns(pl.Series("structural_score", struct_scores))

        n_pre = n_in - (len(keeps) - sum(keeps))
        total_rejected = n_in - len(df)
        return df, self._stats(n_in, df, **dict(reasons),
                               length_ratio=n_in - n_pre)


# =============================================================================
# STAGE 6  —  Entity / numeric consistency
# =============================================================================

def check_entity_consistency(src: str, tgt: str) -> tuple[bool, float]:
    """
    Verify that numbers, URLs, and key symbols are consistent.
    Returns (passes: bool, score: float).  Callable standalone.

    Checks (lightweight regex only):
    1. Number count equality  (within ±1 tolerance for ordinals etc.)
    2. URL presence match  (src has URL ↔ tgt has URL)
    3. Bracket count approximate match
    4. Percent sign count equality
    5. Specific large number match (numbers > 1000 must appear verbatim in tgt)

    Why regex, not model-based: entity consistency is a structural property
    that regex can check in microseconds without model overhead.

    Failure modes: translated numbers (e.g. "twenty" → "20") create false
    positives; accept ±1 tolerance to reduce false positives.
    """
    score_parts: list[float] = []

    # Numbers
    src_nums = _RE_NUMBERS.findall(src)
    tgt_nums = _RE_NUMBERS.findall(tgt)
    n_diff = abs(len(src_nums) - len(tgt_nums))
    if n_diff > max(1, len(src_nums) // 4):
        return False, 0.0
    score_parts.append(1.0 - min(1.0, n_diff * 0.3))

    # Specific large numbers (> 999) must appear verbatim in target
    large_nums = [n for n in src_nums if len(n.replace(",", "").replace(".", "")) >= 4]
    for num in large_nums:
        if num not in tgt:
            return False, 0.0

    # URL presence
    src_has_url = bool(_RE_URL.search(src))
    tgt_has_url = bool(_RE_URL.search(tgt))
    if src_has_url != tgt_has_url:
        return False, 0.0
    score_parts.append(1.0)

    # URL count match
    if src_has_url:
        src_urls = _RE_URL.findall(src)
        tgt_urls = _RE_URL.findall(tgt)
        if len(src_urls) != len(tgt_urls):
            return False, 0.0

    # Bracket count (allow ±1)
    src_br = sum(1 for c in src if c in "([{") - sum(1 for c in src if c in ")]}")
    tgt_br = sum(1 for c in tgt if c in "([{") - sum(1 for c in tgt if c in ")]}")
    br_ok = abs(src_br - tgt_br) <= 1
    score_parts.append(1.0 if br_ok else 0.5)

    # Percent signs
    src_pct = len(_RE_PERCENT.findall(src))
    tgt_pct = len(_RE_PERCENT.findall(tgt))
    if abs(src_pct - tgt_pct) > 1:
        return False, 0.0
    score_parts.append(1.0 - abs(src_pct - tgt_pct) * 0.2)

    return True, sum(score_parts) / len(score_parts) if score_parts else 0.5


class EntityConsistencyStage(FilterStage):
    """
    Stage 6: Entity and numeric consistency.

    Fast regex-based consistency checks without any model.
    Runs at ~200k pairs/sec per core.
    """

    name = "S6_entity"

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in = len(df)
        keeps: list[bool] = []
        entity_scores: list[float] = []
        n_fail = 0

        for src, tgt in zip(df["source"].to_list(), df["target"].to_list()):
            ok, score = check_entity_consistency(src, tgt)
            keeps.append(ok)
            entity_scores.append(score)
            if not ok:
                n_fail += 1

        df = df.filter(pl.Series("_ek", keeps))
        entity_scores = [s for s, k in zip(entity_scores, keeps) if k]
        df = df.with_columns(pl.Series("entity_score", entity_scores))
        return df, self._stats(n_in, df, entity_inconsistency=n_fail)


# =============================================================================
# STAGE 7  —  Copy-source / untranslated detection
# =============================================================================

def check_copy_source(src: str, tgt: str, lang_pair: str) -> tuple[bool, str]:
    """
    Detect untranslated or copy-pasted segments.  Callable standalone.

    Checks:
    1. char_ratio = len(tgt) / len(src)  — language-pair specific bounds
    2. Latin character ratio in Ukrainian target  — catches transliteration /
       untranslated English words (expected <15% in clean Ukrainian text)
    3. Source-target token overlap ratio  — catches verbatim copies

    Thresholds:
      en→uk:  0.50 < char_ratio < 2.20,  max_latin_in_tgt = 0.20
      cs→uk:  0.60 < char_ratio < 1.80,  max_latin_in_tgt = 0.15

    Preserved intentionally: URLs, code fragments, named entities with Latin.
    The Latin ratio threshold (0.20 / 0.15) is permissive enough to keep those.
    """
    if lang_pair == "cs-uk":
        ratio_min, ratio_max, max_latin = 0.60, 1.80, 0.15
    else:  # en-uk
        ratio_min, ratio_max, max_latin = 0.50, 2.20, 0.20

    src_len = max(1, len(src))
    tgt_len = max(1, len(tgt))
    char_ratio = tgt_len / src_len
    if not (ratio_min < char_ratio < ratio_max):
        return False, "char_ratio_oob"

    # Latin ratio in target (Ukrainian should be mostly Cyrillic)
    tgt_alpha = sum(1 for c in tgt if c.isalpha())
    if tgt_alpha > 0:
        tgt_latin = sum(1 for c in tgt if c.isascii() and c.isalpha())
        latin_ratio = tgt_latin / tgt_alpha
        # Allow URLs / code: if src has URL don't penalise
        if latin_ratio > max_latin and not _RE_URL.search(src):
            return False, "high_latin_in_target"

    # Token overlap (verbatim copy detector)
    src_toks = set(src.lower().split())
    tgt_toks = set(tgt.lower().split())
    if len(src_toks) > 3 and len(tgt_toks) > 3:
        overlap = len(src_toks & tgt_toks) / max(len(src_toks), len(tgt_toks))
        if overlap > 0.70:
            return False, "high_token_overlap"

    return True, ""


class CopySourceStage(FilterStage):
    """Stage 7: Copy-source and untranslated detection."""

    name = "S7_copy_source"

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in = len(df)
        keeps: list[bool] = []
        reasons_ctr: dict[str, int] = defaultdict(int)
        lang = self.cfg.lang_pair

        for src, tgt in zip(df["source"].to_list(), df["target"].to_list()):
            ok, reason = check_copy_source(src, tgt, lang)
            keeps.append(ok)
            if not ok:
                reasons_ctr[reason] += 1

        df = df.filter(pl.Series("_cs_keep", keeps))
        return df, self._stats(n_in, df, **dict(reasons_ctr))


# =============================================================================
# STAGE 8  —  Semantic alignment scoring
# =============================================================================

class SemanticAlignmentStage(FilterStage):
    """
    Stage 8: Multilingual embedding cosine similarity.

    Model: intfloat/multilingual-e5-small
    - 12-layer transformer, 384-dim embeddings, 471M vocab
    - Sentence-level multilingual representations
    - CPU inference time: ~3ms/sentence on modern hardware

    ONNX/int8 optimization:
    - With `use_onnx=True` and `optimum[onnxruntime]` installed, converts to
      ONNX and applies AVX512 int8 quantization automatically.
    - Speedup: ~3–5× over torch CPU on AVX512 nodes.
    - Fallback: regular sentence_transformers if optimum not available.

    Memory: ~300 MB for the FP32 model; ~75 MB for int8 ONNX.

    Failure modes: cross-lingual embeddings can be fooled by shared named
    entities.  Apply after LID and structural stages for best effect.
    """

    name = "S8_semantic"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._encode: Callable = None  # type: ignore[assignment]

    def setup(self) -> None:
        model_name = self.cfg.embed_model
        onnx_dir   = Path.home() / ".cache" / "multilingual_e5_small_int8"

        # Try ONNX/int8 first
        if self.cfg.use_onnx:
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer
                import numpy as np
                from scipy.spatial.distance import cosine

                if not onnx_dir.exists():
                    logger.info("Exporting %s to ONNX int8 → %s", model_name, onnx_dir)
                    from optimum.onnxruntime import ORTQuantizer
                    from optimum.onnxruntime.configuration import AutoQuantizationConfig
                    base_model = ORTModelForFeatureExtraction.from_pretrained(
                        model_name, export=True
                    )
                    quantizer = ORTQuantizer.from_pretrained(base_model)
                    qcfg = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
                    quantizer.quantize(save_dir=str(onnx_dir), quantization_config=qcfg)
                    logger.info("ONNX int8 model saved to %s", onnx_dir)

                tok   = AutoTokenizer.from_pretrained(str(onnx_dir))
                model = ORTModelForFeatureExtraction.from_pretrained(str(onnx_dir))

                def _onnx_encode(texts: list[str]) -> "np.ndarray":
                    inputs = tok(texts, padding=True, truncation=True,
                                 max_length=128, return_tensors="np")
                    out  = model(**inputs)
                    # E5 uses mean pooling over non-padding tokens, not [CLS]
                    hidden = out.last_hidden_state          # [B, S, D]
                    mask   = inputs["attention_mask"][:, :, None].astype(float)
                    embs   = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(1e-9)
                    norms  = np.linalg.norm(embs, axis=1, keepdims=True)
                    return embs / np.clip(norms, 1e-9, None)

                self._encode = _onnx_encode
                logger.info("Semantic stage: ONNX int8 model loaded from %s", onnx_dir)
                return
            except Exception as exc:
                logger.info("ONNX init failed (%s), falling back to torch.", exc)

        # Fallback: sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(model_name)

            def _st_encode(texts: list[str]) -> Any:
                return st_model.encode(
                    texts,
                    batch_size=self.cfg.embed_batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

            self._encode = _st_encode
            logger.info("Semantic stage: SentenceTransformer model loaded (%s)", model_name)
        except ImportError:
            logger.warning(
                "sentence_transformers not installed. Stage 8 will be skipped. "
                "Install: pip install sentence_transformers"
            )

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        if self._encode is None:
            logger.warning("Stage 8 skipped (no embedding model available).")
            return df, StageStats(self.name, n_in=len(df), n_out=len(df))

        import numpy as np
        n_in  = len(df)
        srcs  = df["source"].to_list()
        tgts  = df["target"].to_list()

        # E5 prompt prefix: "query: " for source, "passage: " for target
        src_inputs = [f"query: {s}" for s in srcs]
        tgt_inputs = [f"passage: {t}" for t in tgts]

        all_texts = src_inputs + tgt_inputs
        batch     = self.cfg.embed_batch

        # Batch encode with progress logging
        all_embs_list = []
        for i in range(0, len(all_texts), batch):
            all_embs_list.append(self._encode(all_texts[i : i + batch]))
        all_embs = np.concatenate(all_embs_list, axis=0)
        src_embs = all_embs[: len(srcs)]
        tgt_embs = all_embs[len(srcs) :]

        # Cosine similarity (embeddings already L2-normalized)
        sims = (src_embs * tgt_embs).sum(axis=1).tolist()

        min_sim = self.cfg.embed_sim_min
        keep    = [s >= min_sim for s in sims]
        n_low   = sum(1 for k in keep if not k)

        df = df.filter(pl.Series("_sem_keep", keep))
        sims_kept = [s for s, k in zip(sims, keep) if k]
        df = df.with_columns(pl.Series("semantic_score", sims_kept))

        return df, self._stats(n_in, df, low_semantic_sim=n_low)


# =============================================================================
# STAGE 9  —  QE-style quality scoring
# =============================================================================

def compute_composite_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """
    Weighted composite quality score ∈ [0, 1].  Callable standalone.

    scores:  dict of component scores (any subset of keys in weights)
    weights: dict mapping component name → weight (will be re-normalized)

    Available components: semantic, uk_quality, structural, entity, qe
    """
    total_w = 0.0
    total_s = 0.0
    for key, w in weights.items():
        if key in scores and scores[key] is not None:
            total_s += w * float(scores[key])
            total_w += w
    return total_s / total_w if total_w > 0 else 0.0


class QEStage(FilterStage):
    """
    Stage 9: Quality Estimation.

    Applied only to surviving ~20% of pairs after earlier stages.

    Primary backend: PyMarian (Marian NMT Python bindings).
    - Forward-pass log-prob (teacher forcing): measures how probable the
      target is given the source under a pre-trained MT model.
    - Normalized by target length → per-token log-likelihood.
    - Efficient CPU inference: Marian is highly optimized for CPU (AVX2/AVX512).

    Fallback (when PyMarian unavailable): composite score from Stage 4/5/8
    sub-scores, with configurable weights.

    Setup:
    1. Download a Marian model for your language pair (e.g. Helsinki-NLP/opus-mt-en-uk
       from HuggingFace) or point to an existing Marian model directory.
    2. Set config.qe_model = "/path/to/marian/model/dir"

    Memory: ~300 MB for opus-mt-en-uk.  CPU inference: ~10ms/sentence.
    """

    name = "S9_qe"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._qe_fn: Optional[Callable] = None

    def setup(self) -> None:
        if self.cfg.qe_model is None:
            logger.info("Stage 9: no qe_model configured; will use composite fallback.")
            return
        try:
            import pymarian
            model_dir = self.cfg.qe_model

            # pymarian.Evaluator provides reference-free QE scoring.
            # API: evaluator(sources: list[str], targets: list[str]) → list[float]
            evaluator = pymarian.Evaluator(model=model_dir)

            def _pymarian_qe(srcs: list[str], tgts: list[str]) -> list[float]:
                raw = evaluator(srcs, tgts)
                # Normalize from log-prob domain to [0, 1] via sigmoid
                return [1.0 / (1.0 + math.exp(-v)) for v in raw]

            self._qe_fn = _pymarian_qe
            logger.info("Stage 9: PyMarian QE loaded from %s", model_dir)
        except ImportError:
            logger.info("pymarian not installed. Stage 9 will use composite fallback.")
        except Exception as exc:
            logger.warning("Stage 9 PyMarian init failed: %s.  Using fallback.", exc)

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in    = len(df)
        weights = self.cfg.composite_weights
        srcs    = df["source"].to_list()
        tgts    = df["target"].to_list()

        # ── Collect sub-scores already computed in prior stages ───────────────
        score_cols = ["semantic_score", "uk_quality_score", "structural_score", "entity_score"]
        available  = {c: df[c].to_list() for c in score_cols if c in df.columns}

        # ── Optional PyMarian QE ──────────────────────────────────────────────
        qe_scores: Optional[list[float]] = None
        if self._qe_fn is not None:
            try:
                qe_scores = self._qe_fn(srcs, tgts)
                df = df.with_columns(pl.Series("qe_score", qe_scores))
            except Exception as exc:
                logger.warning("PyMarian scoring failed: %s", exc)

        # ── Composite score ───────────────────────────────────────────────────
        comp_key_map = {
            "semantic":   "semantic_score",
            "uk_quality": "uk_quality_score",
            "structural": "structural_score",
            "qe":         "qe_score",
        }
        composites: list[float] = []
        for i in range(len(df)):
            row_scores: dict[str, float] = {}
            for comp_k, col_k in comp_key_map.items():
                if col_k in df.columns:
                    v = df[col_k][i]
                    if v is not None:
                        row_scores[comp_k] = float(v)
                elif comp_k == "qe" and qe_scores is not None:
                    row_scores[comp_k] = float(qe_scores[i])
            composites.append(compute_composite_score(row_scores, weights))

        df = df.with_columns(pl.Series("composite_score", composites))

        min_comp = self.cfg.composite_min
        keep = [c >= min_comp for c in composites]
        n_low = sum(1 for k in keep if not k)
        df = df.filter(pl.Series("_qe_keep", keep))

        return df, self._stats(n_in, df, low_composite_score=n_low)


# =============================================================================
# PIPELINE
# =============================================================================

_STAGE_REGISTRY: dict[int, type[FilterStage]] = {
    1: UnicodeNormStage,
    2: DeduplicationStage,
    3: LangIDStage,
    4: UkrainianQualityStage,
    5: StructuralFilterStage,
    6: EntityConsistencyStage,
    7: CopySourceStage,
    8: SemanticAlignmentStage,
    9: QEStage,
}


class FilterPipeline:
    """
    Orchestrates the cleaning stages.  Processes input in chunks for
    memory efficiency.  Stages 1–7 are CPU-efficient; Stages 8–9 are
    model-based and will be slower without optimization.

    All score columns from earlier stages are preserved for Stage 9.
    After the pipeline, only 'source' and 'target' are written to output.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg    = config
        self.stages: list[FilterStage] = []
        self._build_stages()

    def _build_stages(self) -> None:
        for stage_num in sorted(self.cfg.stages):
            cls = _STAGE_REGISTRY.get(stage_num)
            if cls is None:
                logger.warning("Unknown stage number: %d — skipped.", stage_num)
                continue
            self.stages.append(cls(self.cfg))

    def setup(self) -> None:
        """Initialize all stages (load models, build Bloom filter, etc.)."""
        for stage in self.stages:
            logger.info("Initializing %s …", stage.name)
            stage.setup()

    def run(
        self,
        input_path: Path,
        output_path: Path,
        calibrate: bool = False,
    ) -> list[StageStats]:
        """
        Process input_path in chunks; write filtered pairs to output_path.

        Returns a list of StageStats, one per stage.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_stats: dict[str, StageStats] = {}

        # Calibration pass (optional): compute per-stage thresholds
        if calibrate:
            logger.info("Calibration pass: reading full dataset for threshold computation …")
            full_df = pl.read_csv(str(input_path), separator="\t", has_header=True)
            for stage in self.stages:
                if isinstance(stage, StructuralFilterStage):
                    calibrated = calibrate_structural_thresholds(
                        full_df, self.cfg.lang_pair
                    )
                    stage._thresh = calibrated
                    logger.info("StructuralFilter calibrated: %s", calibrated)
            del full_df

        first_write = True
        chunk_stats_acc: dict[str, StageStats] = {}

        for chunk_idx, chunk in enumerate(_read_tsv_chunks(input_path, self.cfg.chunk_size)):
            logger.info("Chunk %d: %d pairs", chunk_idx, len(chunk))
            df = chunk

            for stage in self.stages:
                df, stats = stage.apply(df)
                if stats.stage_name not in chunk_stats_acc:
                    chunk_stats_acc[stats.stage_name] = StageStats(stats.stage_name)
                acc = chunk_stats_acc[stats.stage_name]
                acc.n_in      += stats.n_in
                acc.n_out     += stats.n_out
                acc.n_rejected += stats.n_rejected
                for k, v in stats.reject_reasons.items():
                    acc.reject_reasons[k] = acc.reject_reasons.get(k, 0) + v

            if len(df) > 0:
                # Drop score columns — keep only source/target in output
                out_cols = [c for c in ["source", "target"] if c in df.columns]
                _append_tsv(df.select(out_cols), output_path, first_write)
                first_write = False

        final_stats = list(chunk_stats_acc.values())
        for s in final_stats:
            logger.info("%s", s)
        return final_stats

    def run_df(
        self,
        df: pl.DataFrame,
        dataset: str = "dataset",
        stats_dir: Optional[Path] = None,
    ) -> tuple[pl.DataFrame, list[StageStats]]:
        """
        Apply the pipeline to an in-memory DataFrame.
        Prints one minimal line per stage; saves JSON stats if stats_dir is given.
        """
        collector = StatsCollector(
            lang_pair = self.cfg.lang_pair,
            dataset   = dataset,
            stats_dir = stats_dir or (Path("data") / "stats" / self.cfg.lang_pair),
        )
        for stage in self.stages:
            df, stats = stage.apply(df)
            collector.record(stats)
        if stats_dir is not None:
            saved = collector.save()
            print(f"  stats → {saved}")
        return df, collector._stages


# =============================================================================
# PUBLIC API  —  one-call interface
# =============================================================================

def run_stages(
    df: pl.DataFrame,
    lang_pair: str = "en-uk",
    stages: Optional[list[int]] = None,
    dataset: str = "dataset",
    stats_dir: Optional[Path] = None,
    **cfg_overrides: Any,
) -> pl.DataFrame:
    """
    Apply selected pipeline stages to a DataFrame.

    Stage IDs
    ---------
    1  unicode     NFC + zero-width / control char removal
    2  dedup       Bloom exact-dedup + MinHash near-dedup (LSH)
    3  langid      fasttext lid.176 — requires fasttext-langdetect
    4  uk_quality  Ukrainian Cyrillic quality score filter
    5  structural  Length / char-ratio / word-ratio bounds
    6  entity      Number / URL / bracket consistency check
    7  copy_src    Verbatim-copy / high-Latin-ratio detection
    8  semantic    multilingual-e5-small cosine similarity
    9  qe          PyMarian quality estimation

    Parameters
    ----------
    df            DataFrame with 'source' and 'target' string columns
    lang_pair     'en-uk' or 'cs-uk'
    stages        Stage IDs to run  (default: [1, 2, 5, 6, 7])
    dataset       Label written into the JSON stats filename
    stats_dir     Folder where {dataset}.json is saved; None = skip
    **cfg_overrides  Any PipelineConfig field, e.g. minhash_threshold=0.9
    """
    cfg = PipelineConfig(
        lang_pair=lang_pair,
        stages=stages if stages is not None else [1, 2, 5, 6, 7],
        **cfg_overrides,
    )
    pipe = FilterPipeline(cfg)
    pipe.setup()
    result, _ = pipe.run_df(df, dataset=dataset, stats_dir=stats_dir)
    return result


# =============================================================================
# MULTIPROCESSING HELPERS  (CPU-stage parallelism)
# =============================================================================

def _process_shard_worker(
    shard_path: Path,
    output_path: Path,
    config_dict: dict,
    stage_nums: list[int],
) -> dict:
    """
    Worker function: run a subset of stages on one shard file.
    Designed to be called via ProcessPoolExecutor.

    Only CPU-local stages (no shared Bloom filter state) should be run here.
    Stages 2 (Bloom), 3 (model), 8, 9 need coordination.
    """
    cfg = PipelineConfig(**config_dict)
    cfg.stages = stage_nums
    pipeline = FilterPipeline(cfg)
    pipeline.setup()
    stats = pipeline.run(shard_path, output_path)
    return {
        "shard": str(shard_path),
        "stats": [
            {"stage": s.stage_name, "n_in": s.n_in, "n_out": s.n_out}
            for s in stats
        ],
    }


def run_parallel_shards(
    shard_paths: list[Path],
    output_dir: Path,
    config: PipelineConfig,
    stage_nums: list[int],
    workers: int = 4,
) -> list[dict]:
    """
    Process multiple shard files in parallel using ProcessPoolExecutor.

    Best suited for CPU-only stages (1, 4, 5, 6, 7).
    Stages requiring shared state (2) or large models (3, 8, 9) should
    be run sequentially or with distributed coordination.

    Returns list of per-shard result dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    cfg_dict = {
        "lang_pair":   config.lang_pair,
        "stages":      stage_nums,
        "workers":     workers,
        "chunk_size":  config.chunk_size,
        "seed":        config.seed,
        "thresholds":  config.thresholds,
    }

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_shard_worker,
                shard,
                output_dir / shard.name,
                cfg_dict,
                stage_nums,
            ): shard
            for shard in shard_paths
        }
        for future in as_completed(futures):
            shard = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info("Completed shard: %s", shard.name)
            except Exception as exc:
                logger.error("Shard %s failed: %s", shard.name, exc)
    return results


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parallel corpus cleaning pipeline for en→uk / cs→uk MT SFT data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",     type=Path, help="Input TSV (source\\ttarget)")
    p.add_argument("--output",    type=Path, help="Output TSV (filtered)")
    p.add_argument(
        "--lang-pair", default="en-uk", choices=list(_DEFAULT_THRESHOLDS.keys()),
        help="Language pair  (default: en-uk)"
    )
    p.add_argument(
        "--stages", default="all",
        help="Comma-separated stage numbers (e.g. '1,2,3') or 'all'  (default: all)"
    )
    p.add_argument("--workers",    type=int,   default=4,       help="Worker processes")
    p.add_argument("--chunk-size", type=int,   default=100_000, help="Rows per chunk")
    p.add_argument("--calibrate",  action="store_true",
                   help="Calibrate Stage-5 thresholds from data distribution")
    p.add_argument("--stats-only", action="store_true",
                   help="Print score statistics and exit (no filtering)")
    p.add_argument("--embed-sim",  type=float, default=0.60,
                   help="Minimum semantic similarity for Stage 8  (default: 0.60)")
    p.add_argument("--uk-quality-min", type=float, default=0.0,
                   help="Minimum Ukrainian quality score for Stage 4  (default: 0 = off)")
    p.add_argument("--no-onnx",   action="store_true",
                   help="Disable ONNX/int8 optimization for Stage 8")
    p.add_argument("--qe-model",  type=str, default=None,
                   help="Path to PyMarian model directory for Stage 9")
    p.add_argument("--fasttext-model", type=str, default=None,
                   help="Path to fasttext lid.176.bin  (auto-downloaded if omitted)")
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity  (default: INFO)"
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    p   = _build_arg_parser()
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.input is None:
        p.error("--input is required.")

    # ── stats-only mode ───────────────────────────────────────────────────────
    if args.stats_only:
        df = pl.read_csv(str(args.input), separator="\t", has_header=True, n_rows=1_000_000)
        numeric_cols = [c for c in df.columns if c not in ("source", "target")]
        print_score_stats(df, ["source", "target"] + numeric_cols,
                          label=args.input.name)
        return

    if args.output is None:
        p.error("--output is required (unless --stats-only).")

    # ── build config ─────────────────────────────────────────────────────────
    if args.stages == "all":
        stage_nums = list(range(1, 10))
    else:
        stage_nums = [int(s.strip()) for s in args.stages.split(",")]

    cfg = PipelineConfig(
        lang_pair      = args.lang_pair,
        stages         = stage_nums,
        workers        = args.workers,
        chunk_size     = args.chunk_size,
        embed_sim_min  = args.embed_sim,
        uk_quality_min = args.uk_quality_min,
        use_onnx       = not args.no_onnx,
        qe_model       = args.qe_model,
        fasttext_model = args.fasttext_model,
    )

    pipeline = FilterPipeline(cfg)
    pipeline.setup()
    stats = pipeline.run(args.input, args.output, calibrate=args.calibrate)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n── Pipeline Summary ──────────────────────────────────────────────")
    for s in stats:
        print(f"  {s}")
    if stats:
        n_in  = stats[0].n_in
        n_out = stats[-1].n_out
        print(f"\n  Input pairs  : {n_in:,}")
        print(f"  Output pairs : {n_out:,}  ({n_out / max(1, n_in) * 100:.1f}% retained)")
        print(f"  Output file  : {args.output}")


if __name__ == "__main__":
    main()
