# russian-verb-noun-collocations
Dataset for complex processing of Russian verb-noun collocations, scripts for testing LLMs and LoRA-training
This project aims at improving automatic processing and general understanding of Russian verb-nominal expessions, which can be either collocial or compositional.

# Data
verb-noun-collocations-dataset.csv contains 3264 examples of verb-nominal expession: 1630 collocations and 1634 non-collocations. In the column "source" you can identify the scource of the example. The dataset includes 479 verbs with 2 to 8 examples for each verbs, classes (1 and 0) are balanced. This data is created specifically for testing models on 3 tasks: binary classification (collocation/non-collocation), paraphrasing (identifying a synonym for a given collocation) and gap filling (a model is supposed to fill a gap in a sentence with appropriate collocations, both semantically and grammatically).

# Model testing
Folders "_runners" contain specific scripts to run testing of models for each task. All the models are tested in 3 generation regimes: zero-shot, one-shot and few-shot.
All in all, 7 basic LLMs are tested:
1. Mistral-7B-Instruct
2. Qwen2.5-7B-Instruct
3. Deepseek-llm-7b-base
4. Llama-3.1-8B
5. YandexGPT5-Lite 8B
6. YandexGPT5-Pro
7. Llama-3.1-70B
The first three models are tested via AutoModelForCausalLM and AutoTokenizer, the last four are reached through API: "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

# LoRA adapting of models
Folder "LoRA" contains a script for training models on the created dataset on 3 tasks simalteneously. This script realizes training of 2 models: Qwen2.5-7B-Instruct and YandexGPT5-Lite 8B, but the choice of models can be expanded.
YandexGPT5-Lite 8B adapted using LoRA on the given dataset can be found on the following page:
https://huggingface.co/VeraKrasnobaeva/yandex_lite_lora_collocations
