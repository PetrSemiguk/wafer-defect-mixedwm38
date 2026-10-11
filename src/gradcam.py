import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import DEFECTS, WaferDataset, combo_name
from model import WaferCNN

SCRATCH = DEFECTS.index("Scratch")


def load_model(path):
    model = WaferCNN()
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model.eval()


@torch.no_grad()
def predict_all(model, ds):
    x = F.one_hot(ds.maps, num_classes=3).permute(0, 3, 1, 2).float()
    return torch.cat([torch.sigmoid(model(x[i:i + 512])) for i in range(0, len(x), 512)]).numpy()


def grad_cam(model, x, target):
    """Heatmap (52x52, scaled to [0, 1]) for one output, plus that output's probability."""
    features = model.features(x)                      # (1, 256, 13, 13)
    logit = model.head(features)[0, target]
    grads = torch.autograd.grad(logit, features)[0]   # how each feature map affects the logit
    weights = grads.mean(dim=(2, 3), keepdim=True)    # one importance weight per feature map
    cam = F.relu((weights * features).sum(dim=1, keepdim=True))  # keep positive evidence only
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), torch.sigmoid(logit).item()


def show(ax, wafer, cam=None, title=""):
    ax.imshow(wafer.numpy(), cmap="gray", vmin=0, vmax=2)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wafers_clean.npz")
    parser.add_argument("--ckpt", default="runs/exp01/best.pt")
    parser.add_argument("--holdout-ckpt", default="runs/exp03_holdout/best.pt")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ds = WaferDataset(args.data, "test")
    y = ds.labels.numpy().astype(int)
    names = np.array([combo_name(v) for v in y])

    model = load_model(args.ckpt)
    pred = (predict_all(model, ds) >= 0.5).astype(int)
    correct = (pred == y).all(axis=1)

    # --- Figure 1: sanity check on single defects ---
    fig, axes = plt.subplots(2, 8, figsize=(22, 6))
    for col, name in enumerate(DEFECTS):
        i = np.where((names == name) & correct)[0][0]
        cam, p = grad_cam(model, ds[i][0][None], col)
        show(axes[0, col], ds.maps[i], title=name)
        show(axes[1, col], ds.maps[i], cam, f"Grad-CAM (p={p:.2f})")
    plt.tight_layout()
    plt.savefig(out / "gradcam_single.png", dpi=130)
    plt.close()

    # --- Figure 2: mixed wafers, one heatmap per defect present ---
    chosen, used = [], set()
    for k in [2, 3, 3, 4]:
        for i in np.where((y.sum(axis=1) == k) & correct)[0]:
            if names[i] not in used:
                chosen.append(i)
                used.add(names[i])
                break
    fig, axes = plt.subplots(len(chosen), 5, figsize=(16, 3.4 * len(chosen)))
    for ax in axes.ravel():
        ax.axis("off")
    for row, i in enumerate(chosen):
        show(axes[row, 0], ds.maps[i], title=names[i])
        for col, d in enumerate(np.where(y[i] == 1)[0], start=1):
            cam, p = grad_cam(model, ds[i][0][None], int(d))
            show(axes[row, col], ds.maps[i], cam, f"{DEFECTS[d]} (p={p:.2f})")
    plt.tight_layout()
    plt.savefig(out / "gradcam_mixed.png", dpi=130)
    plt.close()

    # --- Figure 3: Scratch errors ---
    # Row 1: scratches the main model missed.
    # Row 2: false scratches from the held-out model on Edge-Ring+Loc (never seen in training).
    rows = [("Missed Scratch", model, np.where((y[:, SCRATCH] == 1) & (pred[:, SCRATCH] == 0))[0])]
    if Path(args.holdout_ckpt).exists():
        model_ho = load_model(args.holdout_ckpt)
        probs_ho = predict_all(model_ho, ds)
        false_idx = np.where((names == "Edge-Ring+Loc") & (probs_ho[:, SCRATCH] >= 0.5))[0]
        rows.append(("False Scratch", model_ho, false_idx))
    fig, axes = plt.subplots(len(rows), 8, figsize=(22, 3.3 * len(rows)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, (label, m, indices) in enumerate(rows):
        print(f"{label}: {len(indices)} wafers, showing {min(4, len(indices))}")
        for j, i in enumerate(indices[:4]):
            cam, p = grad_cam(m, ds[i][0][None], SCRATCH)
            show(axes[r, 2 * j], ds.maps[i], title=f"{label}\nTrue: {names[i]}")
            show(axes[r, 2 * j + 1], ds.maps[i], cam, f"Scratch Grad-CAM (p={p:.2f})")
    plt.tight_layout()
    plt.savefig(out / "gradcam_errors.png", dpi=130)
    plt.close()

    print(f"Saved gradcam_single.png, gradcam_mixed.png, gradcam_errors.png to {out}")


if __name__ == "__main__":
    main()