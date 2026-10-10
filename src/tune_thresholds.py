import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import fbeta_score

from dataset import DEFECTS, WaferDataset
from evaluate import predict
from model import WaferCNN

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
MIN_POSITIVES = 500  # do not tune thresholds for defects with too few validation examples


def error_counts(y, pred):
    missed = ((y == 1) & (pred == 0)).sum(axis=0)
    false_alarm = ((y == 0) & (pred == 1)).sum(axis=0)
    return missed, false_alarm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wafers_clean.npz")
    parser.add_argument("--ckpt", default="runs/exp01/best.pt")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--beta", type=float, default=2.0, help="F-beta: beta > 1 favors recall")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaferCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    val_probs, val_y = predict(model, WaferDataset(args.data, "val"), device)
    test_probs, test_y = predict(model, WaferDataset(args.data, "test"), device)

    # --- 1. Choose per-defect thresholds on the VALIDATION set only ---
    thresholds = np.full(len(DEFECTS), 0.5)
    for i in range(len(DEFECTS)):
        if val_y[:, i].sum() < MIN_POSITIVES:
            continue  # too few examples to tune reliably, keep 0.5
        scores = np.array(
            [fbeta_score(val_y[:, i], val_probs[:, i] >= t, beta=args.beta) for t in GRID]
        )
        best = GRID[np.isclose(scores, scores.max())]
        thresholds[i] = best[np.argmin(np.abs(best - 0.5))]  # on ties stay close to 0.5

    # --- 2. Compare default vs tuned thresholds on the TEST set ---
    base_pred = (test_probs >= 0.5).astype(int)
    tuned_pred = (test_probs >= thresholds).astype(int)
    base_missed, base_fa = error_counts(test_y, base_pred)
    tuned_missed, tuned_fa = error_counts(test_y, tuned_pred)

    print(f"{'Defect':<10} {'Thr':>5} {'Missed':>12} {'False alarms':>14}")
    for i, name in enumerate(DEFECTS):
        print(
            f"{name:<10} {thresholds[i]:>5.2f} "
            f"{base_missed[i]:>5} -> {tuned_missed[i]:<4} {base_fa[i]:>6} -> {tuned_fa[i]:<4}"
        )
    base_exact = (base_pred == test_y).all(axis=1).mean()
    tuned_exact = (tuned_pred == test_y).all(axis=1).mean()
    print(f"\nTotal missed:       {base_missed.sum()} -> {tuned_missed.sum()}")
    print(f"Total false alarms: {base_fa.sum()} -> {tuned_fa.sum()}")
    print(f"Exact match:        {base_exact:.4f} -> {tuned_exact:.4f}")

    # --- 3. Trade-off curves for the three hardest defects (validation set) ---
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, ["Edge-Loc", "Loc", "Scratch"]):
        i = DEFECTS.index(name)
        missed = [((val_y[:, i] == 1) & (val_probs[:, i] < t)).sum() for t in GRID]
        false_alarm = [((val_y[:, i] == 0) & (val_probs[:, i] >= t)).sum() for t in GRID]
        ax.plot(GRID, missed, marker="o", label="Missed defects")
        ax.plot(GRID, false_alarm, marker="s", label="False alarms")
        ax.axvline(thresholds[i], color="gray", linestyle="--", label=f"Chosen: {thresholds[i]:.2f}")
        ax.set_title(f"{name} (validation)")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Wafers")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out / "threshold_tradeoff.png", dpi=150)

    json.dump(dict(zip(DEFECTS, thresholds.tolist())), open(out / "thresholds.json", "w"), indent=2)
    print(f"\nSaved {out / 'threshold_tradeoff.png'} and {out / 'thresholds.json'}")


if __name__ == "__main__":
    main()