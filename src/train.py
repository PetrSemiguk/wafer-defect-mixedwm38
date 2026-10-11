import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from dataset import DEFECTS, WaferDataset
from model import WaferCNN


@torch.no_grad()
def evaluate(model, loader, device, class_vectors=None):
    """Metrics in the 8-label space, so both formulations are directly comparable."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        logits = model(x.to(device)).cpu()
        if class_vectors is None:  # multi-label: threshold each of the 8 outputs
            preds.append((torch.sigmoid(logits) >= 0.5).int())
            targets.append(y.int())
        else:  # multi-class: decode the predicted class into its 8 labels
            preds.append(class_vectors[logits.argmax(dim=1)].int())
            targets.append(class_vectors[y].int())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "exact_match": float((preds == targets).all(axis=1).mean()),  # all 8 labels correct
        "per_defect_f1": f1_score(targets, preds, average=None, zero_division=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wafers_clean.npz")
    parser.add_argument("--out", default="runs/debug")
    parser.add_argument("--mode", choices=["multilabel", "multiclass"], default="multilabel")
    parser.add_argument("--holdout", default="", help="comma-separated combinations removed from train/val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0, help="use only N training samples (smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    multiclass = args.mode == "multiclass"
    target = "class" if multiclass else "labels"
    holdout = [s.strip() for s in args.holdout.split(",") if s.strip()]

    train_ds = WaferDataset(args.data, "train", augment=True, target=target, exclude=holdout)
    val_ds = WaferDataset(args.data, "val", target=target, exclude=holdout)
    class_vectors = train_ds.class_vectors if multiclass else None
    n_outputs = len(train_ds.class_vectors) if multiclass else len(DEFECTS)
    if args.limit:
        train_ds = Subset(train_ds, range(args.limit))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    print(f"Mode: {args.mode} ({n_outputs} outputs) | held out: {holdout or 'none'}")
    print(f"Device: {device} | train: {len(train_ds)} | val: {len(val_ds)}")

    model = WaferCNN(n_outputs=n_outputs).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    # Multi-class: pick one of 38 combinations. Multi-label: 8 independent yes/no decisions.
    criterion = nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader)
    )

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running_loss = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * len(x)

        val = evaluate(model, val_loader, device, class_vectors)
        train_loss = running_loss / len(train_ds)
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        print(
            f"Epoch {epoch:>2}/{args.epochs} | loss {train_loss:.4f} | "
            f"val macro-F1 {val['macro_f1']:.4f} | exact match {val['exact_match']:.4f} | "
            f"{time.time() - start:.0f}s"
        )

        if val["macro_f1"] > best_f1:  # keep the best checkpoint by validation macro-F1
            best_f1 = val["macro_f1"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val": val,
                 "mode": args.mode, "holdout": holdout},
                out / "best.pt",
            )
        json.dump(history, open(out / "history.json", "w"), indent=2)

    best = max(history, key=lambda h: h["macro_f1"])
    print(f"\nBest epoch: {best['epoch']} (val macro-F1 {best['macro_f1']:.4f})")
    for name, f1 in zip(DEFECTS, best["per_defect_f1"]):
        print(f"  {name:<10} F1 {f1:.4f}")


if __name__ == "__main__":
    main()