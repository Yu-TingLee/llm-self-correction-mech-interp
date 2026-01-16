import argparse, json
import os
from zipfile import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def nice_notation(a: float) -> str:
    s = f"{abs(a):g}"
    return ("−" + s) if a < 0 else s  # Unicode minus

def load_curves(seq_dir: str, targets: np.ndarray, tol: float = 1e-9) -> dict[str, tuple[list[int], np.ndarray]]:
    curves: dict[str, tuple[list[int], np.ndarray]] = {}

    for d in sorted(p for p in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, p))):
        
        _, coef_str = d.rsplit("_", 1)
        
        alpha = float(coef_str)

        if not np.isclose(abs(alpha), targets, atol=tol, rtol=0.0).any():
            continue

        files = sorted(os.path.join(seq_dir, d, f) for f in os.listdir(os.path.join(seq_dir, d)) if f.endswith(".json"))

        rows = [json.loads(open(f).read()) for f in files]
        if not rows or "baseline_tox" not in rows[0]:
            continue

        layers = []
        prefix, suffix = "steered_L", "_tox"
        for k in rows[0].keys():
            if k.startswith(prefix) and k.endswith(suffix):
                mid = k[len(prefix):-len(suffix)]
                if mid.isdigit():
                    layers.append(int(mid))
        layers.sort()

        baseline = np.array([r["baseline_tox"] for r in rows], float)
        steered = np.array([[r[f"steered_L{L}_tox"] for L in layers] for r in rows], float)

        curves[coef_str] = ([L + 1 for L in layers],
                            (100.0 * (steered - baseline[:, None])).mean(axis=0))

    return curves


def plot_panel(ax, curves_group, title: str):
    if not curves_group:
        ax.axis("off")
        return

    curves_group = sorted(curves_group, key=lambda t: (abs(float(t[0])), float(t[0])))

    for coef_str, xs, ys in curves_group:
        a = float(coef_str)
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.0,
                label=rf"$\alpha={nice_notation(a)}$")

    ax.set_title(title, fontsize=10, pad=2)
    ax.axhline(0, ls="--", linewidth=0.8, alpha=0.8)

    ymin, ymax = ax.get_ylim()
    yr = max(1e-9, ymax - ymin)
    nice_steps = np.array([0.5, 1, 2, 5, 10, 20, 40, 60], float)
    step = nice_steps[np.argmin(np.abs((yr / 4) - nice_steps))]
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.legend(loc="best", fontsize=6, frameon=False, handlelength=1.6, borderaxespad=0.2)
    ax.grid(True, axis="y", which="major", linewidth=0.5, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_basename")
    parser.add_argument("--alpha-abs", type=float, nargs="+", required=True)
    args = parser.parse_args()

    out_dir = os.path.join("outputs", args.model_basename, "steering_d1_t100", "2_injection")
    seq_dir = os.path.join(out_dir, "sequential")
    os.makedirs(out_dir, exist_ok=True)

    targets = np.asarray(args.alpha_abs, float)
    curves = load_curves(seq_dir, targets)

    pos = [(c, *curves[c]) for c in curves if float(c) > 0]
    neg = [(c, *curves[c]) for c in curves if float(c) < 0]
    if not (pos or neg):
        raise SystemExit("No matching nonzero alpha curves found (check --alpha-abs).")

    plt.close("all")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.44, 3.10), constrained_layout=True)

    plot_panel(axes[0], pos, "Detox. direction")
    plot_panel(axes[1], neg, "Tox. direction")

    max_layer = max(max(xs) for _, xs, _ in (pos + neg))
    axes[1].set_xlim(0, max_layer + 1.5)
    axes[1].xaxis.set_major_locator(MultipleLocator(5))
    axes[1].set_xlabel("layer", fontsize=10, labelpad=4)

    fig.align_ylabels(axes)
    fig.supylabel("toxicity change (%pt)", fontsize=10, x=-0.04)
    fig.canvas.draw()

    pos_axes = [ax.get_position() for ax in axes if ax.axison]
    x_center = 0.5 * (min(p.x0 for p in pos_axes) + max(p.x1 for p in pos_axes))
    fig.suptitle(args.model_basename, x=x_center, ha="center", fontsize=10, fontweight="bold")

    suffix = "_abs" + "_".join(f"{x:g}".replace(".", "p") for x in args.alpha_abs)
    out_path = os.path.join(out_dir, f"{args.model_basename}_500{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    main()