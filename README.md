# Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability
This repository contains the official implementation for the paper **Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability**. [https://arxiv.org/abs/2505.11924](https://arxiv.org/abs/2505.11924)

Warning: Some data, prompts, and model outputs may contain toxic or offensive language.

## Run experiments
If you use gated Hugging Face models, export your token first:
```sh
export HF_TOKEN="<YOUR_TOKEN_HERE>"
```

Setup venv:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Run all the experiments:

```sh
bash run.sh
```
