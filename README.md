# WMT2026 Low-Resource Ukrainian — Data, Two-Stage LoRA Training & Evaluation

Two-stage LoRA fine-tuning pipeline for **Qwen/Qwen3.5-2B** on the WMT2026 Ukrainian shared task (MT, QA, SC, GC, MR).

Base model: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)

Dev data: [TUM-NLP/llms-limited-resources2026](https://github.com/TUM-NLP/llms-limited-resources2026) (Ukrainian track)

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── eval_qwen35_ukrainian_zeroshot.py   # Dev evaluation (all 5 tasks)
├── config/
│   └── .gitignore                      # Keeps credentials out of git
├── scripts/
│   ├── download_all_data.sh            # Master: download + prepare training data
│   ├── download_dev_data.sh            # Clone Ukrainian dev sets
│   ├── download_qa_mmlu.sh           # MMLU en + ukr from HuggingFace
│   ├── download_competition_math.sh    # Hendrycks MATH + export jsonl
│   ├── prepare_mr_data.py              # GSM8K + competition_math → jsonl
│   ├── prepare_ua_gec_gc.py            # UA-GEC → GC training jsonl
│   ├── prepare_mmlu_ukr_sc.py          # Synthetic SC from mmlu_ukr
│   ├── translate_train_en_uk.py        # Translate QA/MR en→uk (Stage 2)
│   ├── run_translate_en_uk.sh          # Run translation with Google Cloud API
│   ├── install_google_credentials.sh   # Install service-account JSON
│   ├── train_two_stage.sh              # Full pipeline: Stage1 → pick ckpt → Stage2 → eval
│   ├── run_two_stage_tmux.sh           # Run train_two_stage.sh in tmux
│   ├── monitor_two_stage.sh            # Hourly progress log
│   ├── finetune_stage1_mt_gc_sc.py     # Stage 1: MT + GC + SC
│   ├── finetune_stage2_qa_mr.py        # Stage 2: QA + MR (from best Stage-1 ckpt)
│   ├── finetune_mr_only.py             # Optional: MR-only Ukrainian retrain
│   ├── pick_best_stage1_checkpoint.py    # Pick best Stage-1 by dev MT+GC+SC
│   └── lora_train_utils.py             # Shared dataset / LoRA / SFT helpers
├── train_data/                         # Created by download scripts (gitignored)
├── llms-limited-resources2026/         # Dev eval data (gitignored)
└── outputs/                            # Checkpoints & eval results (gitignored)
```

### Expected `train_data/` after preparation

| Path | Task | Source script |
|------|------|---------------|
| `train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl` | MT (Stage 1 default) | Place manually or use version-1 corpus |
| `train_data/GC/ua_gec_train_single_error.jsonl` | GC | `prepare_ua_gec_gc.py` |
| `train_data/SC/mmlu_ukr_sc_train.jsonl` | SC | `prepare_mmlu_ukr_sc.py` |
| `train_data/QA/mmlu_en_train.jsonl` | QA (en→uk) | `translate_train_en_uk.py` |
| `train_data/QA/mmlu_ukr/data/*.parquet` | QA (uk) | `download_qa_mmlu.sh` |
| `train_data/MR/gsm8k_train.jsonl` | MR | `prepare_mr_data.py` |
| `train_data/MR/competition_math_train.jsonl` | MR | `download_competition_math.sh` |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download & prepare data

```bash
bash scripts/download_all_data.sh
```

### 3. (Optional) Translate QA/MR to Ukrainian for Stage 2

Free backend (no API key):

```bash
python3 scripts/translate_train_en_uk.py --backend free --limit-qa 20000
```

Or Google Cloud Translation API:

```bash
bash scripts/install_google_credentials.sh /path/to/service-account.json
bash scripts/run_translate_en_uk.sh
```

### 4. Place MT training data

Default Stage 1 expects English→Ukrainian parallel jsonl at:

`train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl`

Alternatively, use the shared-task version-1 combined corpus:

```bash
python3 scripts/finetune_stage1_mt_gc_sc.py \
  --mt "version 1/train.jsonl.gz" \
  --mt-format version1 \
  --mt-lang-pairs en-uk,cs-uk
```

### 5. Two-stage training + evaluation

```bash
bash scripts/train_two_stage.sh
# or in tmux:
bash scripts/run_two_stage_tmux.sh
```

Pipeline:

1. **Stage 1** (5 epochs): MT + GC + SC interleaved 1:1:1 → `outputs/stage1_mt_gc_sc_lora/`
2. **Pick best checkpoint** by dev MT + GC + SC → `outputs/stage1_best_checkpoint.json`
3. **Stage 2** (5 epochs): QA + MR from best Stage-1 LoRA → `outputs/stage2_qa_mr_lora/`
4. **Final eval** on all 5 dev tasks → `outputs/stage2_final_dev_eval/summary.json`

### 6. Standalone evaluation

```bash
python3 eval_qwen35_ukrainian_zeroshot.py \
  --model Qwen/Qwen3.5-2B \
  --lora-path outputs/stage2_qa_mr_lora \
  --output-dir outputs/my_eval \
  --force \
  --tasks MT,MT_CS,QA,SC,GC,MR
```

---

## Training hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| LoRA r / alpha | 16 / 32 |
| Learning rate | 2e-4 |
| Batch size × grad accum | 4 × 8 = 32 effective |
| Max sequence length | 2048 |
| Epochs per stage | 5 |

---

## Reference results (dev)

Best Stage-1 checkpoint: `checkpoint-3931` (epoch 1, score = BLEU/100 + GC_pair + SC_pair).

| Task | Metric | Stage-2 final |
|------|--------|---------------|
| MT | BLEU / chrF2 | 18.13 / 46.69 |
| QA | accuracy | 30.4% |
| SC | pair accuracy | 50.0% |
| GC | pair accuracy | 48.3% |
| MR | accuracy | 0% (see `finetune_mr_only.py` for Ukrainian MR fix) |

Zeroshot baseline: `outputs/qwen3_5_2b_ukrainian_zeroshot/summary.json` (run eval without `--lora-path`).
