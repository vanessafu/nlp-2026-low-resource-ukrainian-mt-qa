# Zolint at WMT 2026: A Multitask LLM Submission for the Low-Resource Ukrainian Track

**Team:** Zolint · **Institution:** UTokyo & TUM  
**Shared task:** [WMT 2026 — Multitask LLMs with Limited Resources](https://github.com/TUM-NLP/llms-limited-resources2026) (Ukrainian track)

| Resource | Link |
|----------|------|
| **Hugging Face (LoRA)** | [`MDK-YLC/Qwen3.5-2B-WMT26-Ukrainian-LoRA`](https://huggingface.co/MDK-YLC/Qwen3.5-2B-WMT26-Ukrainian-LoRA) |
| **Base model** | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) |

This repository contains the **runnable training / evaluation / submission** code for our unified Ukrainian multitask system (MT, QA, SC, GC, MR).

---

## Official test results (Ukrainian track)

Our unified system is best on **all reported metrics except QA** (where Koshi leads).

| System | MT All chrF++ | CS-UK BLEU | EN-UK BLEU | QA All EM | SC All EM | GC All EM | Math EM |
|--------|--------------:|-----------:|-----------:|----------:|----------:|----------:|--------:|
| Baseline | 18.46 | 5.38 | 0.55 | 36.33 | 8.38 | 2.18 | 20.00 |
| TUMHN | 16.08 | 3.93 | 1.01 | 40.13 | 16.50 | 6.80 | 4.40 |
| Koshi | 16.80 | 6.01 | 0.70 | **46.99** | 52.50 | 30.55 | 0.40 |
| **Ours (Zolint)** | **49.86** | **23.66** | **15.43** | 41.15 | **66.43** | **45.18** | **21.60** |


---

## Method in one page

1. **EN→UK MT data**  
   Download OPUS/NLLB pairs → length filter → **SDKM-style top‑75** retrieval
   against official EN–UK **dev Ukrainian** (Qwen2.5 embeddings + cosine) →
   semantic-cluster **pseudo-documents** for document-length training.
   Details: [`MT_EN_UK_PIPELINE.md`](MT_EN_UK_PIPELINE.md).

2. **Stage 1 — foundation skills**  
   LoRA (`r=16`, `α=32`) on **MT (en→uk) + GC + SC**, up to 5 epochs.  
   Pick best checkpoint by combined **dev MT+GC+SC** (typically **epoch 1**, `checkpoint-3931`).

3. **Stage 2 — QA/MR + anti-forgetting**  
   Continue the Stage‑1 LoRA for **1 epoch** with:
   - QA (translated MMLU-en + mmlu_ukr + ZNO train)
   - MR (GSM8K + competition_math, Ukrainian)
   - replay: 5k MT + 5k GC
   - **full CS→UK** chat data (`train.cs-uk.jsonl.gz`, 16946 examples)

4. **Test inference → official layout**  
   Task-specific prompts → `Ukrainian/*.jsonl` (+ zip) via `export_ukrainian_submission.py`.

---

## Quick start

### Install

```bash
pip install -r requirements.txt
```

### Load the released LoRA

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen3.5-2B"
lora = "MDK-YLC/Qwen3.5-2B-WMT26-Ukrainian-LoRA"

tok = AutoTokenizer.from_pretrained(lora, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, lora).eval()
```

### Best end-to-end run (Stage2 → test → submission)

Requires Stage‑1 best at `outputs/stage1_mt_gc_sc_lora/checkpoint-3931` and `train.cs-uk.jsonl.gz` in the repo root (or set `MT_CS_PATH`).

```bash
bash scripts/run_stage2_csuk_new_then_test_tmux.sh
tail -f outputs/stage2_csuk_new.log
```

| Output | Path |
|--------|------|
| Stage‑2 LoRA | `outputs/stage2_csuk_new_lora/` |
| Test preds | `outputs/stage2_csuk_new_test_preds/` |
| Submission folder | `Ukrainian_csuk_new/` |
| Zip | `Ukrainian_csuk_new_submission.zip` |

### Full pipeline from scratch

```bash
bash scripts/run_two_stage_tmux.sh
```

Stage1 → pick best → Stage2 (`run_stage2_ukr_mix_cs.sh`).

### Dev eval / test export only

```bash
# Dev
python3 eval_qwen35_ukrainian_zeroshot.py \
  --model Qwen/Qwen3.5-2B \
  --lora-path outputs/stage2_csuk_new_lora \
  --output-dir outputs/my_eval --force \
  --tasks MT,MT_CS,QA,SC,GC,MR

# Test preds → official Ukrainian/ folder
LORA_PATH=outputs/stage2_csuk_new_lora OUT_DIR=outputs/my_test_preds FORCE=1 \
  bash scripts/run_stage2_ukr_test_infer.sh
python3 scripts/export_ukrainian_submission.py \
  --pred-dir outputs/my_test_preds \
  --out-dir Ukrainian \
  --zip-path Ukrainian_submission.zip
```

---

## Repository map

```
.
├── README.md
├── MT_EN_UK_PIPELINE.md              # EN→UK download / SDKM / pseudo-docs
├── QA_PIPELINE.md                    # QA corpus assembly
├── eval_qwen35_ukrainian_zeroshot.py # multi-task dev evaluation
├── data/DATA_PREPROCESSING.md        # TSV cleaning stages (loader)
├── scripts/
│   ├── prepare_en_uk_mt.sh           # EN→UK one-shot data prep
│   ├── download_mt_en_uk.py
│   ├── filter_mt_en_uk.py
│   ├── build_pseudo_docs_en_uk.py
│   ├── prepare_corpus_mt.sh          # CS→UK paragraph MT
│   ├── lora_train_utils.py
│   ├── finetune_stage1_mt_gc_sc.py
│   ├── finetune_stage2_qa_mr_ukr_mix.py
│   ├── run_stage2_ukr_mix_cs.sh
│   ├── run_stage2_csuk_new_then_test.sh
│   ├── infer_ukrainian_test.py
│   ├── export_ukrainian_submission.py
│   └── train_two_stage.sh
├── train_data/                       # gitignored
├── llms-limited-resources2026/       # gitignored
└── outputs/                          # gitignored
```

### Data preparation

| Task | Doc / command |
|------|----------------|
| **EN→UK MT** (SDKM + pseudo-docs) | [`MT_EN_UK_PIPELINE.md`](MT_EN_UK_PIPELINE.md) · `bash scripts/prepare_en_uk_mt.sh` |
| **CS→UK MT** (paragraph corpora) | `bash scripts/prepare_corpus_mt.sh` |
| **QA** | [`QA_PIPELINE.md`](QA_PIPELINE.md) · `bash scripts/prepare_qa_training.sh` |
| **All other tasks** | `bash scripts/download_all_data.sh` (set `SKIP_EN_UK_MT=1` to skip heavy EN→UK) |
| TSV cleaning stages | [`data/DATA_PREPROCESSING.md`](data/DATA_PREPROCESSING.md) |

EN→UK Stage‑1 input path after prep:

`train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl`

---

## Defaults

| Hyperparameter | Value |
|----------------|-------|
| LoRA r / α | 16 / 32 |
| LR | 2e-4 |
| Effective batch | 4 × 8 = 32 |
| Max seq length | 3072 |
| Stage 1 / 2 epochs | 5 / 1 (Stage1 best usually ep1) |

---

## Citation

If you use this code or the released LoRA, please cite the WMT 2026 Limited-Resources shared task and refer to:

> **Zolint at WMT 2026: A Multitask LLM Submission for the Low-Resource Ukrainian Track**  
> Hugging Face: [`MDK-YLC/Qwen3.5-2B-WMT26-Ukrainian-LoRA`](https://huggingface.co/MDK-YLC/Qwen3.5-2B-WMT26-Ukrainian-LoRA)

---

## License

MIT
