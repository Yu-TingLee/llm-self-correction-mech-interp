import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import seed_everything


def dataset_preprocessing(args):
    t_prompts = []
    t_scores = []
    nt_prompts = []
    nt_scores = []
    for data_tag in args.data_tags:
        data_toxic = os.path.join(args.data_processed_dir, data_tag, "t_prompt_score.jsonl")
        data_nontoxic = os.path.join(args.data_processed_dir, data_tag, "nt_prompt_score.jsonl")
        df_t = pd.read_json(data_toxic, lines=True)
        df_nt = pd.read_json(data_nontoxic, lines=True)
        for (i_t, row_t), (i_nt, row_nt) in zip(df_t.iterrows(), df_nt.iterrows()):
            if i_t / len(df_t) < args.data_ratio:
                t_prompts.append(row_t["prompt"])
                t_scores.append(row_t["toxicity"])
            if i_nt / len(df_nt) < args.data_ratio:
                nt_prompts.append(row_nt["prompt"])
                nt_scores.append(row_nt["toxicity"])
    return t_prompts, nt_prompts


def gen_hook_func_hidden_states(layer_idx, hidden_states):
    def get_hidden_states(module, input, output):
        if any(x in str(module) for x in ["Qwen2", "Qwen3", "Llama", "Mistral", "zephyr", "Phi3", "Lfm2"]):
            hidden_states[layer_idx].append(output[0])
        elif "Gemma3" in str(module):
            hidden_states[layer_idx].append(output[0][0])
        else:
            assert 0
    return get_hidden_states


def init_model_hook_hidden_states(model):
    hidden_states = defaultdict(list)
    if any(x in str(model) for x in ["Qwen2", "Qwen3", "Llama", "Mistral", "zephyr", "Phi3", "Lfm2"]):
        for k, model_layer_k in enumerate(list(model.model.layers)):
            model_layer_k.register_forward_hook(
                gen_hook_func_hidden_states(layer_idx=k, hidden_states=hidden_states)
            )
    elif "Gemma3" in str(model):
        for k, model_layer_k in enumerate(list(model.language_model.layers)):
            model_layer_k.register_forward_hook(
                gen_hook_func_hidden_states(layer_idx=k, hidden_states=hidden_states)
            )
        assert 0, f"Hook is not implemented for model {str(model)}"
    return hidden_states


def init_model(args, hook_hidden=True):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=False, token=args.access_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
        token=args.access_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hidden_states = init_model_hook_hidden_states(model) if hook_hidden else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, hidden_states


def get_embed_steering(args, model, tokenizer, hidden_states):
    dataset_tags = "".join(args.data_tags)
    dataset_ratio = int(args.data_ratio * 100)
    output_dir = os.path.join(
        "outputs",
        os.path.basename(args.model_dir),
        f"steering_d{dataset_tags}_t{dataset_ratio}",
        "1_vector",
    )
    os.makedirs(output_dir, exist_ok=True)

    if any(x in str(model) for x in ["Qwen2", "Qwen3", "Llama", "Mistral", "zephyr", "Phi3", "Lfm2"]):
        selected_layers = list(range(model.model.config.num_hidden_layers))
    elif "Gemma3" in str(model):
        selected_layers = list(range(len(model.language_model.layers)))
    else:
        assert 0

    t_prompts, nt_prompts = dataset_preprocessing(args)

    sep = 5
    embed_toxic = [defaultdict(int) for _ in range(sep)]
    embed_nontoxic = [defaultdict(int) for _ in range(sep)]
    counts = 0
    max_len = 0

    for prompt_toxic, prompt_nontoxic in zip(t_prompts, nt_prompts):
        if counts >= args.limit:
            break
        ids_toxic = tokenizer.encode(prompt_toxic)
        ids_nontoxic = tokenizer.encode(prompt_nontoxic)
        max_len = max(max_len, len(ids_toxic), len(ids_nontoxic))

    total_line = min([args.limit, len(t_prompts), len(nt_prompts)])
    for prompt_toxic, prompt_nontoxic in zip(t_prompts, nt_prompts):
        print(f"forward: {counts}/{total_line}")
        if counts >= args.limit:
            break

        ids_toxic = tokenizer.encode(prompt_toxic)
        ids_nontoxic = tokenizer.encode(prompt_nontoxic)
        ids_toxic = [[tokenizer.pad_token_id] * (max_len - len(ids_toxic)) + ids_toxic]
        ids_nontoxic = [[tokenizer.pad_token_id] * (max_len - len(ids_nontoxic)) + ids_nontoxic]

        model(torch.tensor(ids_toxic, dtype=torch.long, device=model.device), use_cache=False)
        for layer, ascore in hidden_states.items():
            if layer in selected_layers:
                x = ascore[-1].detach().float().cpu()
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                embed_toxic[counts % sep][layer] += x.numpy()
            ascore.clear()

        model(torch.tensor(ids_nontoxic, dtype=torch.long, device=model.device), use_cache=False)
        for layer, ascore in hidden_states.items():
            if layer in selected_layers:
                x = ascore[-1].detach().float().cpu()
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                embed_nontoxic[counts % sep][layer] += x.numpy()
            ascore.clear()

        counts += 1

    steering_vec = defaultdict(int)
    steering_vec_sep = [defaultdict(int) for _ in range(sep)]

    for i in range(sep):
        for layer in embed_toxic[i].keys():
            embed_toxic[i][layer] = embed_toxic[i][layer] / counts
            embed_nontoxic[i][layer] = embed_nontoxic[i][layer] / counts
            steering_vec_layer = embed_nontoxic[i][layer] - embed_toxic[i][layer]
            steering_vec[layer] += steering_vec_layer
            steering_vec_sep[i][layer] = steering_vec_layer.tolist()

    steering_vec = {x: y.tolist() for x, y in steering_vec.items()}

    json.dump(steering_vec, open(os.path.join(output_dir, "steer.json"), "w"), indent=2)
    json.dump(steering_vec_sep, open(os.path.join(output_dir, "steer_sep.json"), "w"), indent=2)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--access_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--data_tags", type=str, default=["1"], nargs="+")
    parser.add_argument("--data_ratio", type=float, default=1)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=20000000)
    parser.add_argument("--data_processed_dir", type=str, default="./data_processed")
    args = parser.parse_args()

    seed_everything(0, benchmark=True)
    model, tokenizer, hidden_states = init_model(args)
    get_embed_steering(args, model, tokenizer, hidden_states)


if __name__ == "__main__":
    run()
