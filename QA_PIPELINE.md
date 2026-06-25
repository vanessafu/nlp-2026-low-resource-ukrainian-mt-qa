# QA Training Data Pipeline

This document describes how the Ukrainian QA training corpus is assembled,
which scripts are involved, and what each output file contains.

---

## 1. Raw Data Sources

### 1.1 Competition baseline (`llms-limited-resources2025/Ukrainian/QA/`)

Official splits provided by the WMT25 shared task.
All items are single-choice MCQ with Cyrillic option markers (А/Б/В/Г/Д).

| Split | Items | ukrainian-language-and-literature | history-of-ukraine |
|-------|------:|-----------------------------------|--------------------|
| `train.json` | 2,450 | 1,540 | 910 |
| `dev.json`   | 613   | 385   | 228 |
| `test.json`  | 751   | 403   | 348 |

`dev.json` and `test.json` are **evaluation only** — they are never written into training files.

### 1.2 DUMY (`NLPForUA/dumy-zno-ukrainian-math-history-geo-r1-o1`, HuggingFace)

ZNO questions annotated with DeepSeek-R1 and OpenAI-o1 model outputs.
Used as the source for **CoT training data**.

| Subject | Total | o1_score = 1 |
|---------|------:|-------------:|
| Ukrainian | 1,740 | 1,263 (72.6 %) |
| History | 870 | 680 (78.2 %) |
| Math | 105 | 58 (55.2 %) |
| **Total** | **2,715** | **2,001** |

Fields used:
- `o1_answer` — O1 model's full reasoning + answer (used as CoT text)
- `o1_score` — quality gate (must equal 1 for single-choice correctness)
- `answer_options` — option list with numeric labels (`1:`, `2:`, …)

### 1.3 Belebele (`facebook/belebele`, `ukr_Cyrl` split)

900 Ukrainian reading-comprehension questions from the Belebele benchmark.
Each question has a Flores passage embedded in the prompt.
Converted by `scripts/qa/convert_rc.py` and cached at
`data/training/QA/belebele_rc.json`.

### 1.4 QUA-RC (`QUA-RC/quarc.json`)

875 Ukrainian reading-comprehension MCQs with passage context.
Source text is embedded in the question field (same format as Belebele).
Converted by `scripts/qa/convert_rc.py` and cached at
`data/training/QA/quarc_rc.json`.

---

## 2. Scripts

### `scripts/qa/convert_rc.py`

Converts reading-comprehension sources into the standard QA format
(`question`, `answers`, `correct_answers`, `subject`).

```
python scripts/qa/convert_rc.py                        # belebele (embedded) + quarc
python scripts/qa/convert_rc.py --belebele-format classified   # separate passage field
```

Outputs (already generated, **not re-run** unless data changes):

| File | Items | Notes |
|------|------:|-------|
| `data/training/QA/belebele_rc.json` | 900 | passage embedded in `question` |
| `data/training/QA/quarc_rc.json` | 875 | passage embedded in `question` |
| `data/processed/belebele_qa.json` | 900 | separate `passage` field + subject classification |

---

### `scripts/qa/augment.py`

Oversamples a QA JSON file via option shuffle and question-prefix perturbation.
**Not called directly** in the main pipeline — `build_cot.py` imports and calls
`augment()` inline.

Augmentation axes:
- **Option shuffle**: answer options permuted; correct-answer marker remapped.
- **Prefix perturbation**: Ukrainian instruction phrases prepended to the question
  (e.g. `"Оберіть правильну відповідь:\n"`, `"Укажіть правильну відповідь:\n"`, …).

```
python scripts/qa/augment.py \
    --input  llms-limited-resources2025/Ukrainian/QA/train.json \
    --output data/training/QA/train_augmented.jsonl \
    --factor 2
```

Can also be used standalone to pre-generate augmented files.

---

### `scripts/qa/constants.py`

Shared constants imported by all QA scripts.

| Constant | Content |
|----------|---------|
| `MARKERS` | `["А","Б","В","Г","Д"]` |
| `PREFIXES` | 6 Ukrainian instruction prefixes for prompt perturbation |
| `SYSTEM_ZNO` | System prompt for standard ZNO MCQ |
| `SYSTEM_RC` | System prompt for reading-comprehension items |
| `SYSTEM_MATH` | System prompt for math items |
| `SYSTEM_EXPLICIT_COT` | System prompt for explicit CoT format |

---

### `scripts/qa/analyze_dumy.py`

**Analysis-only tool** — prints overlap statistics between DUMY and the local
competition splits. Writes no files.

```
python scripts/qa/analyze_dumy.py [--cache-dir data/.hf_cache]
```

---

### `scripts/qa/build_cot.py` ← main entry point

Assembles the final training corpus from all sources.

```
python scripts/qa/build_cot.py [--cache-dir data/.hf_cache] [--dry-run]
```

`--dry-run` prints item counts without writing any files.

---

## 3. Output Files

### `data/training/QA/train_cot.jsonl` — CoT items only

**1,322 items** (after running `build_cot.py`).

Source: DUMY items where `o1_score == 1`, filtered against dev and test norms.
Format: chat (`system` / `user` / `assistant`).

Two formats mixed in a **deterministic 70 / 30 split** (assigned by
`MD5(question) % 10 < 7`)

#### 70 % — Implicit CoT (reasoning in user prompt)

O1's reasoning is inserted **between the question stem and the options**
so the model learns to associate the reasoning context with the correct answer.
At inference time the reasoning is absent, but the model's hidden representations
carry an implicit reasoning signal.

```
system:    [SYSTEM_ZNO / SYSTEM_MATH]
user:      {question}

           {o1_answer reasoning}

           А: …
           Б: …
           В: …
           Г: …
assistant: Б
```

#### 30 % — Explicit CoT (reasoning in assistant)

The full reasoning chain is placed in the **assistant turn**, ending with a
standardised answer marker so it can be extracted by regex at inference time
(`Відповідь:\s*[А-Д]`).

```
system:    [SYSTEM_EXPLICIT_COT]
user:      {question}

           А: …
           Б: …
           В: …
           Г: …
assistant: {o1_answer reasoning}
           Відповідь: Б
```

**CoT quality gate**: `o1_score == 1` (O1 answered the single-choice question
correctly). No requirement on `r1_score` or `r1_reasoning`.

**CoT deduplication rules**:
- Items whose question norm appears in `dev.json` or `test.json` are excluded (651 items filtered).
- Duplicate questions within DUMY are deduplicated (first occurrence kept).

---

### `data/training/QA/train_merged.jsonl` — full training corpus

**7,482 items** — all sources combined in a single file.

| Source | Items | Format |
|--------|------:|--------|
| DUMY CoT (o1_score=1) | 1,322 | CoT (implicit 70 % + explicit 30 %) |
| ZNO train.json × 2 | 4,496 | Standard, answer letter only |
| Belebele RC | 900 | Standard, answer letter only |
| QUA-RC | 764 | Standard, answer letter only |
| **Total** | **7,482** | |

**ZNO 2× augmentation details**:

`train.json` is oversampled 2× by generating 3 variants per question and
selecting 2 based on whether the question also appears in the CoT set:

- **CoT-overlap questions** (1,191 unique base questions): both selected variants
  are shuffled (identity order skipped) to guarantee the option ordering differs
  from the corresponding CoT item.
- **Regular questions** (1,057 unique base questions): identity variant + a
  prefix-perturbed shuffle variant (maximises prefix diversity).

**Cross-source deduplication**:
- CoT items and ZNO standard items for the same question are **not** deduplicated
  against each other — they coexist as different training formats.
- Belebele and QUA-RC are deduplicated against the ZNO standard set.

---

### `data/training/QA/build_stats.json`

Per-source counts written after each `build_cot.py` run.

```json
{
  "cot_total":    1322,
  "cot_implicit": 936,
  "cot_explicit": 386,
  "zno_2x":       4496,
  "rc_total":     1664,
  "grand_total":  7482
}
```
