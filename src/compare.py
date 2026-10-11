import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import DEFECTS, WaferDataset, combo_name
from model import WaferCNN


@torch.no_grad()
def predict_labels(ckpt_path, dataset, device):
    """Predictions in the 8-label space for a multi-label or multi-class checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    multiclass = ckpt.get("mode", "multilabel") == "multiclass"
    n_outputs = len(dataset.class_vectors) if multiclass else len(DEFECTS)
    model = WaferCNN(n_outputs=n_outputs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds = []
    for x, _ in DataLoader(dataset, batch_size=512):
        logits = model(x.to(device)).cpu()
        if multiclass:
            preds.append(dataset.class_vectors[logits.argmax(dim=1)])
        else:
            preds.append((torch.sigmoid(logits) >= 0.5).float())
    return torch.cat(preds).numpy().astype(int), ckpt.get("holdout", [])


def summarize(y, pred):
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "exact_match": float((pred == y).all(axis=1).mean()),
        "missed": int(((y == 1) & (pred == 0)).sum()),
        "false_alarms": int(((y == 0) & (pred == 1)).sum()),
        "per_defect_f1": f1_score(y, pred, average=None, zero_division=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wafers_clean.npz")
    parser.add_argument("--multilabel", default="runs/exp01/best.pt")
    parser.add_argument("--multiclass", default="runs/exp02_multiclass/best.pt")
    parser.add_argument("--holdout", default="runs/exp03_holdout/best.pt")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_ds = WaferDataset(args.data, "test")
    y = test_ds.labels.numpy().astype(int)
    names = np.array([combo_name(v) for v in y])

    pred_ml, _ = predict_labels(args.multilabel, test_ds, device)
    pred_mc, _ = predict_labels(args.multiclass, test_ds, device)
    pred_ho, held_out = predict_labels(args.holdout, test_ds, device)

    # --- Experiment 1: both formulations trained on all 38 combinations ---
    ml, mc = summarize(y, pred_ml), summarize(y, pred_mc)
    print(f"EXPERIMENT 1: full training data, test set of {len(y)} wafers\n")
    print(f"{'Model':<22} {'Macro-F1':>9} {'Exact':>8} {'Missed':>7} {'FalseAl':>8}")
    for label, m in [("Multi-label (8 out)", ml), ("Multi-class (38 out)", mc)]:
        print(f"{label:<22} {m['macro_f1']:>9.4f} {m['exact_match']:>8.4f} {m['missed']:>7} {m['false_alarms']:>8}")
    print(f"\n{'Per-defect F1':<12} {'Multi-label':>12} {'Multi-class':>12}")
    for i, name in enumerate(DEFECTS):
        print(f"{name:<12} {ml['per_defect_f1'][i]:>12.4f} {mc['per_defect_f1'][i]:>12.4f}")

    # --- Experiment 2: combinations never seen during training ---
    unseen = np.isin(names, held_out)
    exact_ho = (pred_ho == y).all(axis=1)
    exact_ml = (pred_ml == y).all(axis=1)
    print(f"\nEXPERIMENT 2: {len(held_out)} combinations removed from training\n")
    print(f"{'Combination':<32} {'N':>4} {'Unseen':>8} {'Seen (ref)':>11}")
    rows = {}
    for combo in held_out:
        mask = names == combo
        rows[combo] = {"n": int(mask.sum()), "unseen": float(exact_ho[mask].mean()),
                       "seen_reference": float(exact_ml[mask].mean())}
        print(f"{combo:<32} {mask.sum():>4} {exact_ho[mask].mean():>8.4f} {exact_ml[mask].mean():>11.4f}")
    print(f"{'All unseen combinations':<32} {unseen.sum():>4} {exact_ho[unseen].mean():>8.4f} {exact_ml[unseen].mean():>11.4f}")
    print(f"{'All other combinations':<32} {(~unseen).sum():>4} {exact_ho[~unseen].mean():>8.4f} {exact_ml[~unseen].mean():>11.4f}")
    print("\nUnseen:     multi-label model trained WITHOUT these combinations")
    print("Seen (ref): multi-label model trained on all combinations")
    print("A 38-class model scores 0 on unseen combinations by construction (no such output).")

    # --- What goes wrong on unseen combinations ---
    patterns = Counter()
    for i in np.where(unseen & ~exact_ho)[0]:
        missed = "+".join(d for d, t, p in zip(DEFECTS, y[i], pred_ho[i]) if t and not p)
        extra = "+".join(d for d, t, p in zip(DEFECTS, y[i], pred_ho[i]) if p and not t)
        parts = ([f"missed {missed}"] if missed else []) + ([f"false {extra}"] if extra else [])
        patterns[f"{names[i]}: " + "; ".join(parts)] += 1
    print("\nMost frequent errors on unseen combinations:")
    for pattern, n in patterns.most_common(6):
        print(f"  {n:>3}x  {pattern}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    json.dump(
        {"full_data": {"multilabel": ml, "multiclass": mc},
         "held_out": {"combinations": rows,
                      "all_unseen": float(exact_ho[unseen].mean()),
                      "all_other": float(exact_ho[~unseen].mean())}},
        open(out / "comparison.json", "w"), indent=2,
    )
    print(f"\nSaved {out / 'comparison.json'}")


if __name__ == "__main__":
    main()