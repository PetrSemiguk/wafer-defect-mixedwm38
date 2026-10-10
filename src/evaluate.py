import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from dataset import DEFECTS, WaferDataset
from model import WaferCNN


def join_names(mask, empty="Normal"):
    names = [d for d, flag in zip(DEFECTS, mask) if flag]
    return "+".join(names) if names else empty


@torch.no_grad()
def predict(model, dataset, device):
    probs, targets = [], []
    for x, y in DataLoader(dataset, batch_size=512):
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy().astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wafers_clean.npz")
    parser.add_argument("--ckpt", default="runs/exp01/best.pt")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaferCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    test_ds = WaferDataset(args.data, "test")
    probs, y = predict(model, test_ds, device)
    pred = (probs >= args.threshold).astype(int)
    exact = (pred == y).all(axis=1)

    # --- 1. Per-defect metrics: misses (escapes) vs false alarms ---
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, average=None, zero_division=0
    )
    missed = ((y == 1) & (pred == 0)).sum(axis=0)
    false_alarm = ((y == 0) & (pred == 1)).sum(axis=0)
    negatives = (y == 0).sum(axis=0)
    print(f"Test set: {len(y)} wafers, threshold {args.threshold}\n")
    print(f"{'Defect':<10} {'N':>5} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Missed':>7} {'FalseAl':>8} {'FA rate':>8}")
    for i, name in enumerate(DEFECTS):
        print(
            f"{name:<10} {support[i]:>5} {precision[i]:>7.4f} {recall[i]:>7.4f} {f1[i]:>7.4f} "
            f"{missed[i]:>7} {false_alarm[i]:>8} {false_alarm[i] / negatives[i]:>8.2%}"
        )
    print(f"\nMacro-F1: {f1.mean():.4f} | Exact match: {exact.mean():.4f} "
          f"({(~exact).sum()} wafers with at least one error)")

    # --- 2. Exact match by number of defects on the wafer ---
    n_defects = y.sum(axis=1)
    print("\nExact match by number of defects:")
    by_count = {}
    for k in range(5):
        mask = n_defects == k
        by_count[k] = float(exact[mask].mean())
        print(f"  {k} defects: {exact[mask].mean():.4f}  (n={mask.sum()}, errors={(~exact[mask]).sum()})")

    # --- 3. Hardest label combinations ---
    combos = np.array([join_names(v) for v in y])
    combo_stats = sorted(
        ((exact[combos == c].mean(), (combos == c).sum(), c) for c in np.unique(combos))
    )
    print("\nHardest combinations:")
    for acc, n, name in combo_stats[:8]:
        print(f"  {name:<38} {acc:.4f}  (n={n})")

    # --- 4. Most frequent error patterns ---
    patterns = Counter()
    for i in np.where(~exact)[0]:
        parts = []
        if ((y[i] == 1) & (pred[i] == 0)).any():
            parts.append("missed " + join_names((y[i] == 1) & (pred[i] == 0)))
        if ((y[i] == 0) & (pred[i] == 1)).any():
            parts.append("false " + join_names((y[i] == 0) & (pred[i] == 1)))
        patterns["; ".join(parts)] += 1
    print("\nMost frequent error patterns:")
    for pattern, n in patterns.most_common(8):
        print(f"  {n:>3}x  {pattern}")

    # --- 5. Save a gallery of misclassified wafers ---
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    error_idx = np.where(~exact)[0][:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), error_idx):
        ax.imshow(test_ds.maps[i].numpy(), cmap="viridis", vmin=0, vmax=2)
        ax.set_title(f"True: {join_names(y[i])}\nPred: {join_names(pred[i])}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "test_errors.png", dpi=150)

    # --- 6. Save metrics for the README ---
    metrics = {
        "threshold": args.threshold,
        "macro_f1": float(f1.mean()),
        "exact_match": float(exact.mean()),
        "per_defect": {
            name: {
                "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                "support": int(support[i]), "missed": int(missed[i]), "false_alarms": int(false_alarm[i]),
            }
            for i, name in enumerate(DEFECTS)
        },
        "exact_match_by_defect_count": by_count,
    }
    json.dump(metrics, open(out / "test_metrics.json", "w"), indent=2)
    print(f"\nSaved {out / 'test_errors.png'} and {out / 'test_metrics.json'}")


if __name__ == "__main__":
    main()