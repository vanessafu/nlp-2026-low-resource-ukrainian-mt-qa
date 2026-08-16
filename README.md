# WMT2026 Low-Resource Ukrainian — Two-Stage LoRA Training & Evaluation

Two-stage LoRA fine-tuning pipeline for **Qwen/Qwen3.5-2B** on the WMT2026 Ukrainian shared task (MT, MT_CS, QA, SC, GC, MR).

| | |
|--|--|
| **Team** | Zolint |
| **Institution** | UTokyo & TUM |
| **Base model** | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| **Released LoRA** | [MDK-YLC/Qwen3.5-2B-WMT26-Ukrainian-LoRA](https://huggingface.co/MDK-YLC/Qwen3.5-2B-WMT26-Ukrainian-LoRA) |
| **Dev data** | [TUM-NLP/llms-limited-resources2026](https://github.com/TUM-NLP/llms-limited-resources2026) (Ukrainian track) |

---

## Canonical training recipe (best)

1. **Stage 1** (5 epochs, LoRA r=16): MT (en→uk) + GC + SC  
   → pick best by dev MT+GC+SC (typically `checkpoint-3931`, epoch 1)
2. **Stage 2** (1 epoch): QA + MR + anti-forgetting replay  
   - 5k MT (en→uk)  
   - 5k GC  
   - full CS→UK (`train.cs-uk.jsonl.gz`, 16946 examples)  
   - QA (translated MMLU-en + mmlu_ukr + ZNO train)  
   - MR (gsm8k + competition_math, Ukrainian)
3. **TEST inference** → official `Ukrainian/` submission JSONL layout

### Recommended one-shot (Stage2 from existing Stage1 best + test + submission)

```bash
# Requires Stage-1 best at outputs/stage1_mt_gc_sc_lora/checkpoint-3931
# and train.cs-uk.jsonl.gz in repo root (or set MT_CS_PATH)
bash scripts/run_stage2_csuk_new_then_test_tmux.sh
# monitor:
tail -f outputs/stage2_csuk_new.log
```

Outputs:

| Path | Content |
|------|---------|
| `outputs/stage2_csuk_new_lora/` | Stage-2 LoRA (final adapter) |
| `outputs/stage2_csuk_new_test_preds/` | raw test preds |
| `Ukrainian_csuk_new/` | submission folder |
| `Ukrainian_csuk_new_submission.zip` | zipped submission |

### Full two-stage from scratch

```bash
bash scripts/run_two_stage_tmux.sh
# or: bash scripts/train_two_stage.sh
```

This runs Stage1 → pick best → Stage2 via `run_stage2_ukr_mix_cs.sh` (first 10k CS-UK from version-1 corpus by default).

### Canonical Stage2 only (old CS-UK source)

```bash
STAGE1_LORA=outputs/stage1_mt_gc_sc_lora/checkpoint-3931 \
  bash scripts/run_stage2_ukr_mix_cs.sh
```

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── eval_qwen35_ukrainian_zeroshot.py   # Dev evaluation
├── scripts/
│   ├── lora_train_utils.py             # Shared dataset / LoRA / SFT helpers
│   ├── finetune_stage1_mt_gc_sc.py     # Stage 1: MT + GC + SC
│   ├── finetune_stage2_qa_mr_ukr_mix.py# Stage 2: QA+MR+MT/GC/MT_CS mix
│   ├── pick_best_stage1_checkpoint.py
│   ├── pick_best_stage2_checkpoint.py
│   ├── run_stage2_ukr_mix_cs.sh        # Canonical Stage2 + pick + dev eval
│   ├── run_stage2_csuk_new_then_test.sh# Best Stage2 (full cs-uk) + TEST + submit
│   ├── infer_ukrainian_test.py         # Official test inference
│   ├── export_ukrainian_submission.py  # Preds → Ukrainian/*.jsonl (+ zip)
│   ├── train_two_stage.sh              # Full Stage1→Stage2 pipeline
│   └── ...                             # data prep / ablations
├── train_data/                         # gitignored
├── llms-limited-resources2026/         # gitignored (dev/test)
└── outputs/                            # gitignored
```

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
bash scripts/download_all_data.sh
# Optional: translate QA/MR to Ukrainian
python3 scripts/translate_train_en_uk.py --backend free --limit-qa 20000
```

Place / download:

- `train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl` (MT en→uk)
- `train.cs-uk.jsonl.gz` (CS→UK chat messages; used by best Stage2)
- Dev/test under `llms-limited-resources2026/Ukrainian/`

### 3. Train + evaluate / submit

See **Canonical training recipe** above.

### 4. Standalone dev eval

```bash
python3 eval_qwen35_ukrainian_zeroshot.py \
  --model Qwen/Qwen3.5-2B \
  --lora-path outputs/stage2_csuk_new_lora \
  --output-dir outputs/my_eval \
  --force \
  --tasks MT,MT_CS,QA,SC,GC,MR
```

### 5. Test inference + submission export

```bash
LORA_PATH=outputs/stage2_csuk_new_lora \
  OUT_DIR=outputs/my_test_preds \
  FORCE=1 \
  bash scripts/run_stage2_ukr_test_infer.sh

python3 scripts/export_ukrainian_submission.py \
  --pred-dir outputs/my_test_preds \
  --out-dir Ukrainian \
  --zip-path Ukrainian_submission.zip
```

---

## Training hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| LoRA r / alpha | 16 / 32 (canonical best) |
| Learning rate | 2e-4 |
| Batch × grad accum | 4 × 8 = 32 effective |
| Max sequence length | 3072 |
| Stage 1 epochs | 5 (best ckpt usually epoch 1) |
| Stage 2 epochs | 1 |

Ablations with r=32 / 1+1 or 2+2 epochs: `scripts/train_two_stage_{1,2}ep_r32.sh`.

---

## Reference results (dev, Stage1 best)

Stage-1 best: `outputs/stage1_mt_gc_sc_lora/checkpoint-3931`

| Task | Metric | Zeroshot | Stage1 best |
|------|--------|----------|-------------|
| MT | BLEU | 15.40 | **17.92** |
| GC | det / corr | 34.2% / 32.4% | **42.5% / 42.1%** |
| SC | det / corr | 32.5% / 30.6% | **75.2% / 59.3%** |

After Stage2 (canonical mix+cs, `checkpoint-1843`):

| Task | Metric | Score |
|------|--------|-------|
| MT | BLEU | 18.72 |
| MT_CS | BLEU | 21.45 |
| QA | accuracy | 38.2% |
| SC | det / corr | 76.1% / 61.4% |
| GC | det / corr | 38.2% / 37.8% |
| MR | accuracy | 25.0% |

Released HF LoRA matches the stronger Stage2 run trained with **full** `train.cs-uk.jsonl.gz` (16946 rows).

---

## License

MIT
