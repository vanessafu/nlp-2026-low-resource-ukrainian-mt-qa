# EN→UK MT Training Data Pipeline

This document describes how we obtain and process **English→Ukrainian**
machine-translation training data for Stage‑1 LoRA.

**Motivation.** Official MT *test* inputs are document-level (~hundreds of
words), while raw parallel corpora are sentence-level (~15 tokens). Fine-tuning
only on short sentences transfers poorly to long documents. We therefore:

1. download diverse EN–UK sentence pairs,
2. keep pairs that are **semantically close to the official EN–UK dev**
   (Ukrainian side), and
3. concatenate related sentences into **pseudo-documents**.

Entry point:

```bash
bash scripts/prepare_en_uk_mt.sh
```

Requires the shared-task dev files under
`llms-limited-resources2026/Ukrainian/MT/dev.en-uk.{en,uk}`
(or the 2025 layout). Download them with `bash scripts/download_dev_data.sh`
if needed. Embedding step (filter) prefers a GPU.

---

## 1. Raw data sources

### 1.1 OPUS Moses (primary)

`scripts/download_mt_en_uk.py` downloads a compact OPUS mix (ParaCrawl,
WikiMatrix, TED2020, Tatoeba, EUbookshop, CCAligned, OpenSubtitles, Wikimedia,
XLEnt, QED, Bible, Ubuntu, KDE4, GNOME, MaCoCu) as Moses zip archives and
merges unique `(en, uk)` pairs.

### 1.2 NLLB (optional, default on)

Same script loads HuggingFace `allenai/nllb` (`eng_Latn-ukr_Cyrl`) and keeps
rows with `laser_score ≥ 1.2`.

### 1.3 Full WMT constrained recipe (reference)

`scripts/mtdata.recipes.wmt25-constrained.yml` lists the broader
`wmt25-eng-ukr` inventory. Use with [mtdata](https://github.com/thammegowda/mtdata)
if you want the full constrained dump instead of the OPUS subset above; then
point the filter script at your Moses files.

### Output of download

| File | Content |
|------|---------|
| `data/raw/eng-ukr/train.eng` | English sentences (one per line) |
| `data/raw/eng-ukr/train.ukr` | Aligned Ukrainian sentences |

```bash
python3 scripts/download_mt_en_uk.py
python3 scripts/download_mt_en_uk.py --skip-nllb
python3 scripts/download_mt_en_uk.py --max-per-corpus 200000
```

---

## 2. Filtering + SDKM-style retrieval

Script: `scripts/filter_mt_en_uk.py`

### 2.1 Basic length filter

Drop a pair if either side has **&lt; 3** or **&gt; 100** words, or if the
length ratio is outside **[1/3, 3]**.

### 2.2 Embeddings (Qwen2.5)

Model: `Qwen/Qwen2.5-3B-Instruct`.

For each Ukrainian sentence (train and official **dev**):

1. tokenize with `max_length=128`, padding + truncation,
2. forward pass with `output_hidden_states=True`,
3. **mean-pool** the last hidden state with the attention mask,
4. store `float32` vectors (optionally cached under `data/processed/*.npy`).

### 2.3 Top‑K cosine retrieval

1. **L2-normalize** train and dev Ukrainian embeddings,
2. for **each** official EN–UK **dev Ukrainian** sentence, compute cosine
   similarity against all training Ukrainian embeddings,
3. keep the **top‑75** training indices (configurable `--top-k`),
4. **union + deduplicate** across all queries.

Similarity is computed on the **Ukrainian side only** (domain matching to
dev), then the corresponding EN–UK pairs are written out.

### Output of filter

| File | Content |
|------|---------|
| `data/processed/selected_en_uk.jsonl` | Selected sentence pairs |
| `data/processed/embeddings_en_uk_{train,dev}.npy` | Cached embeddings (optional) |

Each JSONL line:

```json
{"src": "...", "tgt": "...", "word_count": 12, "n_sentences": 1}
```

Typical size after dedup: on the order of **~100k** pairs (depends on raw
volume and how much the top‑75 sets overlap).

```bash
python3 scripts/filter_mt_en_uk.py --top-k 75
```

---

## 3. Pseudo-document construction

Script: `scripts/build_pseudo_docs_en_uk.py`

### 3.1 Training pseudo-docs

1. Embed **English** sides of `selected_en_uk.jsonl` with
   `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
   (`normalize_embeddings=True`).
2. Cluster with `MiniBatchKMeans`
   (`n_clusters = n_sentences // chunk_size`, `batch_size=10000`, `n_init=3`).
3. Within each cluster, sort indices and **concatenate** source/target with
   spaces.
4. Skip clusters with fewer than 2 sentences.
5. Repeat for chunk sizes **3, 5, 8, 12** and concatenate all results.

### 3.2 Dev pseudo-docs (paragraph validation)

Same procedure on official `dev.en-uk.*` with chunk sizes **3, 5, 8**, then
split by English word count:

| Split | Word-count range |
|-------|------------------|
| short | 50–250 |
| medium | 250–800 |

### Outputs

| File | Role |
|------|------|
| `data/pseudo_docs/train_pseudo_en_uk.jsonl` | Training pseudo-docs |
| `data/pseudo_docs/dev_pseudo_en_uk.jsonl` | All dev pseudo-docs |
| `data/pseudo_docs/dev_pseudo_en_uk_short.jsonl` | Short subset |
| `data/pseudo_docs/dev_pseudo_en_uk_medium.jsonl` | Medium subset |
| `train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl` | **Stage‑1 default** (copy) |

Each line has `src`, `tgt`, `n_sentences`, `word_count`.

```bash
python3 scripts/build_pseudo_docs_en_uk.py
python3 scripts/build_pseudo_docs_en_uk.py --train-chunk-sizes 3 5 8 12
```

---

## 4. One-shot wrapper & env knobs

```bash
bash scripts/prepare_en_uk_mt.sh
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `SKIP_DOWNLOAD` | `0` | Reuse existing `data/raw/eng-ukr/` |
| `SKIP_NLLB` | `0` | Skip HuggingFace NLLB download |
| `MAX_PER_CORPUS` | empty | Cap pairs per OPUS/NLLB source |
| `TOP_K` | `75` | Neighbours per dev Ukrainian query |

From `scripts/download_all_data.sh`, set `SKIP_EN_UK_MT=1` to skip this heavy
step in the global data prep.

---

## 5. How Stage‑1 consumes the data

`scripts/finetune_stage1_mt_gc_sc.py` defaults to:

```text
--mt train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl
--mt-format parallel
```

`parallel` format expects `src` / `tgt` (or `en` / `uk`) fields per JSONL
line — exactly what the pseudo-doc builder writes.

---

## 6. Script index

| Script | Purpose |
|--------|---------|
| `scripts/prepare_en_uk_mt.sh` | End-to-end orchestration |
| `scripts/download_mt_en_uk.py` | Download + merge raw EN–UK |
| `scripts/filter_mt_en_uk.py` | Length filter + SDKM top‑K |
| `scripts/build_pseudo_docs_en_uk.py` | Semantic clustering → pseudo-docs |
| `scripts/mtdata.recipes.wmt25-constrained.yml` | Full constrained corpus list (optional) |

Related CS→UK paragraph pipeline (separate): `scripts/prepare_corpus_mt.sh`,
`scripts/prepare_cs_paragraph_mt.py`. Cleaning stages used by several TSV
builders are documented in [`data/DATA_PREPROCESSING.md`](data/DATA_PREPROCESSING.md).
