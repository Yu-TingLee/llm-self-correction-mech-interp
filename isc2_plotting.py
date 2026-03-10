import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    d = json.load(path.open("r", encoding="utf-8"))["per_round"]
    d = sorted(d, key=lambda x: x["round"])
    round_ = np.array([r["round"] for r in d])
    mean = np.array([r["mean"] for r in d])
    var = np.sqrt(np.array([r["var"] for r in d]))
    return round_, mean, var


def main(model_name: str, strength: str):
    model_basename = os.path.basename(model_name)
    output_dir = os.path.join("outputs", model_basename)
    os.makedirs(output_dir, exist_ok=True)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(3.44, 4.2), dpi=300)

    series = [
        ("RoBERTa",  "detox", f"{model_basename}_RoBERTa_{strength}_detox_stats.json"),
        ("Detoxify", "detox", f"{model_basename}_Detoxify_{strength}_detox_stats.json"),
        ("RoBERTa",  "tox",   f"{model_basename}_RoBERTa_{strength}_tox_stats.json"),
        ("Detoxify", "tox",   f"{model_basename}_Detoxify_{strength}_tox_stats.json"),
    ]
    x_ticks = None
    colors = {
        ("RoBERTa",  "detox"): "tab:blue",
        ("Detoxify", "detox"): "tab:orange",
        ("RoBERTa",  "tox"):   "tab:green",
        ("Detoxify", "tox"):   "tab:red",
    }
    band_alpha = 0.25

    for classifier, cond, fname in series:
        p = Path("outputs") / model_basename / f"{classifier}_{strength}_text_results" / fname
        x, mean, std = load(p)
        x_ticks = x

        c = colors[(classifier, cond)]
        label = f"{classifier} ({cond})"

        ax.fill_between(x, mean - std, mean + std, color=c, alpha=band_alpha, linewidth=0, zorder=1)
        ax.plot(x, mean, color=c, linewidth=2.2, label=label, zorder=2)

    ax.set_title(f"{model_basename}", fontsize=16, weight="bold", pad=6)
    ax.set_xlabel("round", fontsize=14, labelpad=4)
    ax.set_xlim(min(x_ticks), max(x_ticks))
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.legend(loc="best", frameon=True, fontsize=10)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(output_dir, f"{model_basename}_{strength}_new.png"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--strength", required=True, choices=["weak", "strong"])
    args = parser.parse_args()
    main(args.model_name, args.strength)
