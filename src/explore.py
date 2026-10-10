from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("data/Wafer_Map_Datasets.npz")
# Assumed label order from the dataset authors -- verified visually by the plot below
DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]

# --- 1. Load raw arrays ---
data = np.load(DATA_PATH)
print("Keys:", data.files)
maps, labels = data["arr_0"], data["arr_1"]
print("Maps:  ", maps.shape, maps.dtype)
print("Labels:", labels.shape, labels.dtype)

# --- 2. Pixel values (expected: 0 = no die, 1 = good die, 2 = failed die) ---
values, counts = np.unique(maps, return_counts=True)
print("\nPixel values:", dict(zip(values.tolist(), counts.tolist())))

# --- 3. Per-defect frequency ---
print("\nWafers containing each defect:")
for name, n in zip(DEFECTS, labels.sum(axis=0)):
    print(f"  {name:<10} {int(n):>6}  ({n / len(labels):.1%})")

# --- 4. Number of defects per wafer and number of unique combinations ---
n_defects = labels.sum(axis=1).astype(int)
print("\nDefects per wafer:", dict(sorted(Counter(n_defects.tolist()).items())))
combos = Counter(map(tuple, labels.astype(int).tolist()))
print("Unique label combinations:", len(combos))
print("Smallest class size:", min(combos.values()), "| largest:", max(combos.values()))

# --- 5. Exact duplicates (matter for train/test leakage later) ---
unique_maps = len({m.tobytes() for m in maps})
print(f"\nExact duplicate maps: {len(maps) - unique_maps}")

# --- 6. One example of normal + each single defect ---
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
targets = [np.zeros(8, dtype=int)] + [np.eye(8, dtype=int)[i] for i in range(8)]
titles = ["Normal"] + DEFECTS
for ax, target, title in zip(axes.ravel(), targets, titles):
    idx = np.where((labels == target).all(axis=1))[0]
    if len(idx):
        ax.imshow(maps[idx[0]], cmap="viridis", vmin=0, vmax=2)
    ax.set_title(f"{title} (n={len(idx)})")
    ax.axis("off")
plt.tight_layout()
Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/single_defect_examples.png", dpi=150)
print("\nSaved reports/single_defect_examples.png")