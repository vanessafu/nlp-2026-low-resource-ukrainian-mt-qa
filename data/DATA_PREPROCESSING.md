# Data Processing

## Cleaning Pipeline (`cleaning_pipeline.py`)

Eight sequential stages. Each stage operates on `(source, target)` sentence/paragraph pairs.

| Stage | Name | Method / Model | Purpose |
|-------|------|----------------|---------|
| S1 | Unicode + HTML | Regex, `html.unescape`, `unicodedata.normalize("NFC")` | Remove zero-width chars, control chars, HTML tags, garbled glyphs; collapse repeated words/chars and whitespace |
| S2 | Structural filter | Percentile-based thresholds (6.5–93.5 pct) on char length and char/word length ratio; soft score on punctuation symmetry | Drop pairs that are too short,or misaligned in length |
| S3 | Deduplication | Exact: **Bloom filter** (MurmurHash3). Near: **MinHash + LSH** (128 permutations, band-based, Jaccard ≈ 0.90 threshold) | Remove identical and near-duplicate sentence pairs across the full corpus |
| S4 | Language ID | **FastText `lid.176.bin`** (Meta), confidence ≥ 0.87 | Keep only pairs where source is the expected language and target is Ukrainian |
| S5 | Ukrainian quality | Rule-based: ratio of Ukrainian-only letters (і,ї,є,ґ), penalise Russian character markers (ы,э,ъ,ё) and Russian function words | Score target-side Cyrillics; detect and penalise Russian contamination |
| S6 | Entity consistency | Regex matching on numbers, percentages, URLs | Reject pairs where numeric/URL counts diverge between source and target |
| S7 | Semantic alignment | **`intfloat/multilingual-e5-small`** (SentenceTransformers, CUDA fp16), cosine similarity ≥ 0.87 | Remove translation pairs that are semantically unrelated |
| S8 | QE composite | Weighted average of S2 structural score (0.35), S7 semantic score (0.40), S5 Ukrainian quality (0.10); bottom 12.5 percentile removed | Final quality cut based on accumulated per-pair scores |

---

## Per-Corpus Filters Applied

### MT Training — en-uk

| Corpus | Pre-pipeline filter | Pipeline stages |
|--------|-------------------|-----------------|
| **MaCoCu uk-en** | `bicleaner_ai_score ≥ 0.86` (neural bicleaner, built into corpus metadata) | S1 S2 S3 S4 S5 S6 S8 |
| **NLLB en-uk** | `laser_score ≥ 1.2` (LASER2 sentence embedding similarity, built into NLLB metadata) | S1 S2 S3 S4 S5 S6 S7 S8 |
| **ParaCrawl v9 en-uk** | None | S1 S2 S3 S4 S5 S6 S7 S8 |
| **OpenSubtitles en-uk** | None | S1 S2 S3 S4 S5 S6 S7 S8 |
| **DocHPLT en-uk** | URL filter (same domain, path similarity ≥ 0.60); sentence-count ratio ∈ [0.5, 2.0]; per-entry: `aligner ≥ 0.45`, `bicleaner ≥ 0.70`, `bifixer ≥ 4.20`, char-ratio ≥ 0.45; paragraph mean quality ≥ 0.60 | S1 S2 S3 S5 |
| **EUbookshop en-uk** | Paragraphs of 5–MAX sentences only | S1 S2 S3 S5 S6 S7 |

### MT Training — cs-uk

| Corpus | Pre-pipeline filter | Pipeline stages |
|--------|-------------------|-----------------|
| **NLLB cs-uk** | `laser_score ≥ 1.2` | S1 S2 S3 S4 S5 S6 S7 S8 |
| **OpenSubtitles cs-uk** | None | S1 S2 S3 S4 S5 S6 S7 S8 |

### MT Validation

| Corpus | Filter | Pipeline stages |
|--------|--------|-----------------|
| **WMT24++ en-uk** | Remove `domain == "canary"` and `is_bad_source == True` rows | None (human-annotated gold data) |
| **TildeMODEL en-uk** | None | S1 S2 S3 S4 S5 S6 S7 S8 |

### QA (Training & Validation)

No automated cleaning pipeline. Data sourced from structured QA datasets (Belebele, RC-QA); quality is controlled at the source.
