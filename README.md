# Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability
This repository contains the official implementation for the paper: **Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability**. [https://arxiv.org/abs/2505.11924](https://arxiv.org/abs/2505.11924)

**TL;DR:** We provide a mechanisic explanation of intrinsic self-correction as representation steering along interpretable latent directions.

<br>
<p align="center">
  <img src="./images/isc_pipeline.png" width="100%">
  <br>
  <em>Figure 1: Overview of intrinsic self-correction as representation steering.</em>
</p>


## Run experiments
Warning: Some data, prompts, and model outputs may contain toxic or offensive language.

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
