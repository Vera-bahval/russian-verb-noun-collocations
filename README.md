# Russian Verb-Noun Collocations

This repository provides a dataset and scripts for evaluating and fine-tuning large language models (LLMs) on the task of understanding Russian verb-noun expressions, including both collocations and compositional phrases. The project is designed to enhance automatic processing and deepen linguistic understanding of verb-nominal constructions in Russian.

---

## Dataset

The dataset (`verb-noun-collocations-dataset.csv`) contains **3,264** annotated verb-noun expressions:
- **1,630 collocations** (label `1`)
- **1,634 non-collocations** (label `0`)

### Key Features:
- Covers **479 unique verbs**, each represented with **2 to 8 examples**
- Balanced dataset with equal representation of both classes
- The `source` column specifies the origin of each example
- Designed for three NLP tasks:
  1. **Binary Classification**: Distinguish collocations from non-collocations
  2. **Paraphrasing**: Replace collocations with a synonymous verb
  3. **Gap Filling**: Predict the correct collocation in a masked sentence (semantically and grammatically appropriate)

---

## Model Testing

Scripts for model evaluation are located in the `_runners` directories, one per task. Each model is tested in three prompting regimes:
- **Zero-shot**
- **One-shot**
- **Few-shot**

### Evaluated Models:
1. `Mistral-7B-Instruct`
2. `Qwen2.5-7B-Instruct`
3. `Deepseek-LLM-7B-Base`
4. `LLaMA-3.1-8B`
5. `YandexGPT5-Lite 8B`
6. `YandexGPT5-Pro`
7. `LLaMA-3.1-70B`

- The first three models are accessed via Hugging Face Transformers (`AutoModelForCausalLM`, `AutoTokenizer`)
- The last four models are accessed via the Yandex API:  
  `https://llm.api.cloud.yandex.net/foundationModels/v1/completion`

---
### LLM-as-a-judge
The `llm-as-a-judge` folder contains 2 scripts for evaluating models' performance on the tasks "paraphrasing" and "gap filling". Model `YandexGPT5-Lite 8B` is used as a judge.

## LoRA Fine-Tuning

The `LoRA` folder contains a script for **multi-task fine-tuning** (classification, paraphrasing, gap filling) using **Low-Rank Adaptation (LoRA)**.

### Supported Models:
- `Qwen2.5-7B-Instruct`
- `YandexGPT5-Lite 8B`

The fine-tuned YandexGPT5-Lite 8B model is available on Hugging Face:  
[Yandex Lite LoRA Collocations](https://huggingface.co/VeraKrasnobaeva/yandex_lite_lora_collocations)

---

## Repository Structure
russian-verb-noun-collocations/
├── data/verb-noun-collocations-dataset.csv
├── _runners/ # Scripts for model evaluation
├── llm-as-a-judge # Scripts for model evaluation on paraphrasing and gap filling
├── LoRA/ # LoRA training script and configs
└── README.md

