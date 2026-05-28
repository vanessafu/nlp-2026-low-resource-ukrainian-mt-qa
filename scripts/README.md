# scripts/

Two scripts build the full training and validation corpora for en→uk / cs→uk MT.

---

## cleaning_pipeline.py

Generic 9-stage cleaning pipeline.  All algorithms operate on in-memory Polars
DataFrames.  The main entry point is `run_stages()`.

### Stage reference

| ID | Name | Algorithm | Notes |
|----|------|-----------|-------|
| 1 | unicode | NFC + zero-width char removal + apostrophe normalisation | Always run first |
| 2 | dedup | Bloom filter exact-dedup + MinHash/LSH near-dedup | Cross-chunk Bloom; shard-local near-dedup |
| 3 | langid | fasttext lid.176 language detection | Requires `fasttext-langdetect` |
| 4 | uk_quality | Cyrillic quality score: uk_char×0.5 + (1−ru_marker)×0.3 + (1−ru_func)×0.2 | Penalises ы э ъ ё Russian markers |
| 5 | structural | Length bounds, char-ratio, word-ratio, punctuation consistency | Thresholds in `_DEFAULT_THRESHOLDS` |
| 6 | entity | Number / URL / bracket count consistency (regex, no model) | ~200k pairs/sec |
| 7 | copy_src | Token-overlap > 70% and high Latin-in-target ratio | Catches untranslated pairs |
| 8 | semantic | multilingual-e5-small cosine similarity ≥ 0.60 (ONNX/int8 optional) | ~3ms/sentence on CPU |
| 9 | qe | PyMarian teacher-forcing log-prob QE | Disabled by default |

### API

```python
from cleaning_pipeline import run_stages
from pathlib import Path

df_clean = run_stages(
    df,                              # pl.DataFrame with 'source' and 'target' columns
    lang_pair="en-uk",               # "en-uk" or "cs-uk"
    stages=[1, 2, 5, 6, 7],          # stage IDs to run
    dataset="my_dataset",            # label for the JSON stats file
    stats_dir=Path("data/stats/en-uk"),
    # optional PipelineConfig overrides:
    minhash_threshold=0.90,
    embed_sim_min=0.65,
)
```

Stats are saved to `{stats_dir}/{dataset}.json`.

### CLI (file-to-file mode)

```bash
# Full pipeline on a TSV file
python scripts/cleaning_pipeline.py \
    --input  data/raw/my_corpus.tsv \
    --output data/training/MT/en-uk/my_corpus_clean.tsv \
    --lang-pair en-uk --stages all --workers 8

# Lightweight (stages 1–7, skip semantic + QE)
python scripts/cleaning_pipeline.py \
    --input raw.tsv --output clean.tsv \
    --lang-pair en-uk --stages 1,2,3,4,5,6,7

# Calibrate structural thresholds from data, then filter
python scripts/cleaning_pipeline.py \
    --input raw.tsv --output clean.tsv \
    --lang-pair en-uk --calibrate --stages 5
```

---

## loader.py

Downloads, column-filters, and cleans each corpus.  Each `filter_*` function:
1. Applies **dataset-specific column filters** (quality scores unique to that source).
2. Calls `run_stages()` with the appropriate stage subset.

### Dataset filter recipes

| Dataset | Quality columns | Pipeline stages | Rationale |
|---------|----------------|-----------------|-----------|
| **MaCoCu** | bicleaner_ai ≥ 0.86, bifixer ≥ 301.2, bleualign ≥ 0.60, boilerplate ∈ {good/neargood}, biroamer = No | 1, 2, 7 | Column filters already guarantee high quality; only unicode + dedup + copy-source check needed |
| **NLLB** | LASER ≥ 1.10, src_lid ∈ [0.95,1.01], tgt_lid ∈ [0.98,1.01] | 1, 2, 4, 5, 7 | Pre-filtered by cross-lingual similarity; add UK quality + structural + copy-source |
| **OpenSubtitles** | — | 1, 2, 3, 4, 5, 6, 7, 8 | Highly noisy; full pipeline including LangID and semantic alignment |
| **ParaCrawl** | CLD2 LID + neural cleaning upstream | 1, 2, 3, 4, 5, 6, 7, 8 | Neural cleaning helps but lower-confidence pairs still noisy |
| **DocHPLT** | aligner ≥ 0.70, bicleaner ≥ 0.85, bifixer ≥ 0.80 per sentence | 1, 2, 4, 5, 7 | Document-level; builds paragraph + sentence pairs; column scores cover alignment |
| **WMT24++** | human-annotated, post-edited | **none** | Gold-standard validation set; no automated cleaning |
| **TildeMODEL** | basic parallel alignment only | 1, 2, 3, 4, 5, 6, 7, 8 | Moderate noise; full pipeline |

### DocHPLT document structure

Each row in HPLT/DocHPLT is one document — a list of sentence-pair dicts:

```json
[
  {"src": ["sentence 1"], "tgt": ["переклад 1"],
   "aligner-score": 0.858, "bicleaner-score": 0.994, "bifixer-score": 0.926},
  {"src": ["sentence 2"], "tgt": ["переклад 2"],
   "aligner-score": 0.666, "bicleaner-score": 0.884, "bifixer-score": 0.920}
]
```

`filter_docholt` calls `_build_docholt_pairs()` which:
1. Filters each sentence pair by score thresholds (aligner ≥ 0.70, bicleaner ≥ 0.85, bifixer ≥ 0.80).
2. Builds **paragraph pairs** with a sliding window (size=4, stride=2) over consecutive kept sentences.
3. Samples **15% of remaining sentences** (not used in any window) as sentence-level examples.

### Running everything

```bash
# from project root, inside the enroot container
enroot start -m $PWD:/workspace wmt25
python scripts/loader.py
```

Processes all datasets in sequence and writes to `data/`.

### Running a single dataset

```python
from pathlib import Path
from loader import (
    download_macocu, load_macocu_gz, filtered_macocu,
    save_sentence_corpus, DATA_DIR,
)

df = filtered_macocu(
    load_macocu_gz(download_macocu("uk-en")),
    stats_dir=DATA_DIR / "stats" / "en-uk",
)
save_sentence_corpus(df, DATA_DIR / "training" / "MT" / "en-uk", "macocu.tsv")
```

---

## Output structure

```
data/
├── training/MT/
│   ├── en-uk/
│   │   ├── macocu.tsv           sentence pairs  (TSV: source\ttarget)
│   │   ├── nllb.tsv
│   │   ├── open_subtitles.tsv
│   │   ├── paracrawl.tsv
│   │   └── docholt.tsv          paragraph + sentence pairs
│   └── cs-uk/
│       ├── nllb.tsv
│       └── open_subtitles.tsv
├── validation/MT/
│   ├── en-uk/
│   │   ├── wmt24pp.tsv          mixed sentence+paragraph (70/30 ratio)
│   │   ├── tilde.tsv
│   │   └── open_subtitles_en.tsv
│   └── cs-uk/
│       └── open_subtitles_cs.tsv
└── stats/
    ├── en-uk/
    │   ├── macocu.json
    │   ├── nllb_en-uk.json
    │   ├── open_subtitles_en-uk.json
    │   ├── paracrawl_en-uk.json
    │   └── docholt_en-uk.json
    └── cs-uk/
        ├── nllb_cs-uk.json
        └── open_subtitles_cs-uk.json
```

All TSV files have a `source\ttarget` header row.

---

## Stats JSON — reading the output

Each `data/stats/{lang_pair}/{dataset}.json` looks like:

```json
{
  "lang_pair": "en-uk",
  "dataset": "nllb_en-uk",
  "total_in": 3200000,
  "total_out": 2650000,
  "stages": [
    {
      "stage": "S1_unicode",
      "n_in": 3200000, "n_out": 3198000,
      "n_rejected": 2000, "retention_pct": 99.94,
      "reject_reasons": {"empty_after_norm": 2000}
    },
    {
      "stage": "S2_dedup",
      "n_in": 3198000, "n_out": 2900000,
      "n_rejected": 298000, "retention_pct": 90.68,
      "reject_reasons": {"exact_dup": 240000, "near_dup": 58000}
    },
    {
      "stage": "S4_uk_quality",
      "n_in": 2900000, "n_out": 2890000,
      "n_rejected": 10000, "retention_pct": 99.66,
      "reject_reasons": {"low_uk_quality": 10000}
    }
  ]
}
```

### What to look for

| Observation | Likely cause | Action |
|-------------|-------------|--------|
| S2 retention < 60% | Corpus has many near-duplicates (e.g. crawled boilerplate) | Expected for ParaCrawl/OpenSubtitles; check `near_dup` vs `exact_dup` ratio |
| S3 retention < 80% | Many wrong-language pairs in source | Check if `wrong_src_lang` or `wrong_tgt_lang` dominates |
| S4 retention < 90% | Russian contamination in Ukrainian side | Lower `uk_quality_min` threshold or inspect rejected samples |
| S5 retention < 70% | Length distribution very different from defaults | Run with `--calibrate` to use data-driven thresholds |
| S7 rejection spike | Many copy-source pairs | Common in subtitle data; `high_token_overlap` reason will dominate |
| S8 retention < 50% | Semantic misalignment | Reduce `embed_sim_min` or check source/target are correctly assigned |
