import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(toxic_data, non_toxic_data, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    tox_df = pd.read_json(toxic_data, lines=True)
    non_tox_df = pd.read_json(non_toxic_data, lines=True)

    tox_df["toxicity"] = tox_df["prompt"].apply(lambda d: d.get("toxicity") if isinstance(d, dict) else None)
    non_tox_df["toxicity"] = non_tox_df["prompt"].apply(lambda d: d.get("toxicity") if isinstance(d, dict) else None)

    tox_df["tox_bin"] = pd.qcut(tox_df["toxicity"], q=10, duplicates="drop")
    toxic_train, toxic_test = train_test_split(
        tox_df, test_size=1000, stratify=tox_df["tox_bin"], random_state=87
    )
    toxic_train = toxic_train.drop(columns=["tox_bin"])
    toxic_test = toxic_test.drop(columns=["tox_bin"])
    toxic_train.to_json(os.path.join(data_dir, "toxic_train_4k.jsonl"), orient="records", lines=True, force_ascii=False)
    toxic_test.to_json(os.path.join(data_dir, "toxic_test_1k.jsonl"), orient="records", lines=True, force_ascii=False)

    plt.figure(figsize=(9, 5.5))
    plt.hist(toxic_train["toxicity"], bins=10, alpha=0.55, label="Train")
    plt.hist(toxic_test["toxicity"], bins=10, alpha=0.55, label="Test")
    plt.title("Toxicity Distribution of toxic train and test splits")
    plt.xlabel("Toxicity")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "tox_splits_distribution.png"), dpi=160)
    plt.close()

    print("\nToxic splits toxicity stats:")
    print(f"Train split. mean = {toxic_train['toxicity'].mean()}, std = {toxic_train['toxicity'].std()}")
    print(f"Test  split. mean = {toxic_test['toxicity'].mean()}, std = {toxic_test['toxicity'].std()}")

    non_tox_df["tox_bin"] = pd.qcut(non_tox_df["toxicity"], q=10, duplicates="drop")
    non_toxic_train, non_toxic_test = train_test_split(non_tox_df, test_size=1000, stratify=non_tox_df["tox_bin"], random_state=87)
    non_toxic_train = non_toxic_train.drop(columns=["tox_bin"])
    non_toxic_test = non_toxic_test.drop(columns=["tox_bin"])
    non_toxic_train.to_json(os.path.join(data_dir, "non_toxic_train_4k.jsonl"), orient="records", lines=True, force_ascii=False)
    non_toxic_test.to_json(os.path.join(data_dir, "non_toxic_test_1k.jsonl"), orient="records", lines=True, force_ascii=False)

    plt.figure(figsize=(9, 5.5))
    plt.hist(non_toxic_train["toxicity"], bins=10, alpha=0.55, label="Train")
    plt.hist(non_toxic_test["toxicity"], bins=10, alpha=0.55, label="Test")
    plt.title("Toxicity Distribution of non-toxic train and test splits")
    plt.xlabel("Toxicity")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "non_tox_splits_distribution.png"), dpi=160)
    plt.close()

    print("\nNon-toxic splits toxicity stats:")
    print(f"Train split. mean = {non_toxic_train['toxicity'].mean()}, std = {non_toxic_train['toxicity'].std()}")
    print(f"Test  split. mean = {non_toxic_test['toxicity'].mean()}, std = {non_toxic_test['toxicity'].std()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toxic_data", type=str,
                        default="./data/unprocessed_toxic_5k.jsonl",
                        help="Path to the raw toxic JSONL file.")
    parser.add_argument("--non_toxic_data", type=str,
                        default="./data/unprocessed_non_toxic_5k.jsonl",
                        help="Path to the raw non-toxic JSONL file.")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory where split files and plots are saved.")
    args = parser.parse_args()
    create_splits(args.toxic_data, args.non_toxic_data, args.data_dir)
