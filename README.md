# Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability
This repository contains the official implementation for the paper **Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability**. [https://arxiv.org/abs/2505.11924](https://arxiv.org/abs/2505.11924)

Warning: Some data, prompts, and model outputs may contain toxic or offensive language.

## Experiments
If you use gated Hugging Face models, export your token first:
```sh
export HF_TOKEN="<YOUR_TOKEN_HERE>"
```
### Intrinsic Self-Correction
- isc0_create_split.py: Create stratified train/test splits.
- isc1_text_exp.py: Run multi-round text experiments (toxification + detoxification) and sample prompt-induced shifts.
- isc2_plotting.py: Plots figures of toxicity trajectories across rounds.
  
### Steering
- isc0_create_split.py: Create stratified train/test splits.
- steering0_preprocess.py: Pre-process the dataset and sort the dataset in increasing/decreasing toxicity.
- steering1_build.py: Build the steering vectors from training datasets with difference-in-means over hidden states.
- steering2_injection.py: Inject steering vectors during generation.
- steering2_plotting_inj.sh: Produces figures of toxicity change across layers.
- steering3_cossim.py: Plots the cosine similarity between steering vectors and prompt-induced shifts.

Then, run all the experiments with:
```sh
# ISC exps
bash isc0_create_splits.sh
bash isc1_text_exp.sh
bash isc2_plotting.sh
# Steering exps
bash steering0_preprocess.sh
bash steering1_build.sh
bash steering2_injection.sh
bash steering2_plotting_delta.sh
bash steering3_cossim.sh
```
