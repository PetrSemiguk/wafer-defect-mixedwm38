import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset import WaferDataset, combo_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wafers_clean.npz")
    parser.add_argument("--ckpt", default="runs/exp01/best.pt")
    parser.add_argument("--thresholds", default="reports/thresholds.json")
    parser.add_argument("--out", default="demo")
    parser.add_argument("--per-combo", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # --- 1. A small, balanced sample of test wafers: N per label combination ---
    ds = WaferDataset(args.data, "test")
    maps = ds.maps.numpy().astype(np.uint8)
    labels = ds.labels.numpy().astype(np.uint8)
    names = np.array([combo_name(v) for v in labels])
    rng = np.random.default_rng(args.seed)
    chosen = np.concatenate([
        rng.choice(np.where(names == n)[0], size=min(args.per_combo, (names == n).sum()), replace=False)
        for n in sorted(set(names))
    ])
    np.savez_compressed(out / "samples.npz", maps=maps[chosen], labels=labels[chosen])

    # --- 2. Model weights only (no training metadata) ---
    state = torch.load(args.ckpt, map_location="cpu")["model"]
    torch.save(state, out / "wafer_cnn_weights.pth")

    # --- 3. Files the app needs next to it: thresholds and the model definition ---
    shutil.copy(args.thresholds, out / "thresholds.json")
    shutil.copy(Path(__file__).parent / "model.py", out / "model.py")

    print(f"Demo bundle in {out}/:")
    print(f"  samples.npz            {len(chosen)} wafers, {len(set(names[chosen]))} combinations")
    for f in ["wafer_cnn_weights.pth", "thresholds.json", "model.py"]:
        print(f"  {f:<22} {(out / f).stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()