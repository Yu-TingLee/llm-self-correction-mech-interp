# Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability
This repository contains the official implementation for the paper **Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability**. [https://arxiv.org/abs/2505.11924](https://arxiv.org/abs/2505.11924)

Warning: Some data, prompts, and model outputs may contain toxic or offensive language.

## Experimental Results
**TL;DR:** We propose a mechanisic analysis that links intrinsic self-correction to representation steering along interpretable latent directions.

Intrinsic self-correction can steer a model's response via prompting.
We are interested in how this steering functions mechanistically.
To understand this phenomenon, we construct contrastive steering vectors and measure their alignment with prompt-induced representation shifts. Our results support the hypothesis that intrinsic self-correction functions as representation steering along interpretable latent directions.

<br>
<p align="center">
  <img src="./images/isc_pipeline.png" width="100%">
  <br>
  <em>Figure 1: Overview of intrinsic self-correction as representation steering.</em>
</p>


<details>
<summary><b>Toxicity evolution under self-correction prompting.</b></summary><br>

<br>
<p align="center">
  <img src="./images/text_exp_res.png" width="100%">
  <br>
  <em>Figure 2: We test whether self-correction prompting can induce toxicity changes.
Results show that the tested models are consistently steered.</em>
</p>
</details>

<details>
<summary><b>Alignment between prompt-induced shifts and steering vectors.</b></summary><br>

<br>
<p align="center">
  <img src="./images/cossim_res.png" width="100%">
  <br>
  <em>Figure 3: We measure cosine similarity between prompt-induced shifts and steering vectors (interpretable latent directions).<br>
Results support the view that intrinsic self-correction functions as representation steering along an interpretable latent direction.</em>
</p>
</details>


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
