import argparse
import os

import pandas as pd
from tqdm import tqdm
from transformers import logging

from src.utils import load_roberta_toxicity_classifier

logging.set_verbosity_error()


def post_process(t_prompt_score, nt_prompt_score):
    t_prompt_score = sorted(t_prompt_score, key=lambda x: x[1], reverse=True)
    nt_prompt_score = sorted(nt_prompt_score, key=lambda x: x[1])
    t_prompt_score = t_prompt_score[: int(len(t_prompt_score))]
    nt_prompt_score = nt_prompt_score[: int(len(nt_prompt_score))]
    return t_prompt_score, nt_prompt_score


def run_and_save_json(data_id, run_function, output_dir):
    limit = 2000000
    t_prompt_score, nt_prompt_score = run_function(limit)
    t_prompt_score = [{"prompt": x[0], "toxicity": x[1]} for x in t_prompt_score]
    nt_prompt_score = [{"prompt": x[0], "toxicity": x[1]} for x in nt_prompt_score]

    out = os.path.join(output_dir, str(data_id))
    os.makedirs(out, exist_ok=True)

    t_df = pd.DataFrame(data=t_prompt_score, columns=["prompt", "toxicity"])
    nt_df = pd.DataFrame(data=nt_prompt_score, columns=["prompt", "toxicity"])

    t_df.to_json(os.path.join(out, "t_prompt_score.jsonl"), orient="records", lines=True)
    nt_df.to_json(os.path.join(out, "nt_prompt_score.jsonl"), orient="records", lines=True)


def run_realtoxicity(limit, data_dir, roberta_tokenizer, roberta_model):
    data_toxic = pd.read_json(os.path.join(data_dir, "toxic_train_4k.jsonl"), lines=True)
    data_notoxic = pd.read_json(os.path.join(data_dir, "non_toxic_train_4k.jsonl"), lines=True)

    t_prompt_score = []
    nt_prompt_score = []
    counts = 0

    for t_row, nt_row in tqdm(
        zip(data_toxic.iterrows(), data_notoxic.iterrows()), total=len(data_toxic)
    ):
        if counts >= limit:
            break
        toxic_prompt = t_row[1]["prompt"]["text"]
        notoxic_prompt = nt_row[1]["prompt"]["text"]
        toxic_score = t_row[1]["prompt"]["toxicity"]
        notoxic_score = nt_row[1]["prompt"]["toxicity"]

        t_prompt_score.append((toxic_prompt, toxic_score))
        nt_prompt_score.append((notoxic_prompt, notoxic_score))
        counts += 1

    return post_process(t_prompt_score, nt_prompt_score)


def run_1dataset(data_dir, output_dir):
    roberta_tokenizer, roberta_model = load_roberta_toxicity_classifier("cuda")

    def _run(limit):
        return run_realtoxicity(limit, data_dir, roberta_tokenizer, roberta_model)

    run_and_save_json(1, _run, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process dataset and sort by toxicity score."
    )
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the train split JSONL files.")
    parser.add_argument("--output_dir", type=str, default="./data_processed",
                        help="Root directory for processed output files.")
    args = parser.parse_args()
    run_1dataset(args.data_dir, args.output_dir)
