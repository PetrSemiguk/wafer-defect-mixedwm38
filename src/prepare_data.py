import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]
SEED = 42


def combo_name(label):
    """Human-readable name of a label vector, e.g. 'Center+Scratch'."""
    names = [d for d, flag in zip(DEFECTS, label) if flag]
    return "+".join(names) if names else "Normal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/Wafer_Map_Datasets.npz")
    parser.add_argument("--out", default="data/wafers_clean.npz")
    args = parser.parse_args()

    data = np.load(args.raw)
    maps, labels = data["arr_0"], data["arr_1"].astype(np.uint8)
    print(f"Raw: {len(maps)} wafers")

    # --- 1. Unexpected pixel value 3: report where it occurs, then treat as failed die (2) ---
    has3 = (maps == 3).any(axis=(1, 2))
    print(f"\nWafers containing value 3: {has3.sum()}")
    for name, n in Counter(combo_name(l) for l in labels[has3]).most_common():
        print(f"  {name:<35} {n}")
    maps = np.where(maps == 3, 2, maps).astype(np.uint8)

    # --- 2. Remove exact duplicates, keeping the first occurrence ---
    first_seen, keep = {}, []
    dup_classes, conflicts = Counter(), 0
    for i, m in enumerate(maps):
        key = m.tobytes()
        if key in first_seen:
            dup_classes[combo_name(labels[i])] += 1
            if not np.array_equal(labels[i], labels[first_seen[key]]):
                conflicts += 1  # same map, different labels
        else:
            first_seen[key] = i
            keep.append(i)
    print(f"\nDuplicates removed: {len(maps) - len(keep)} (with conflicting labels: {conflicts})")
    for name, n in dup_classes.most_common(10):
        print(f"  {name:<35} {n}")
    maps, labels = maps[keep], labels[keep]

    # --- 3. Stratified 70/15/15 split by the full label combination (38 classes) ---
    combo_id = labels @ (1 << np.arange(8))  # unique integer per combination
    idx = np.arange(len(maps))
    train_idx, rest_idx = train_test_split(
        idx, test_size=0.30, stratify=combo_id, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=combo_id[rest_idx], random_state=SEED
    )

    # --- 4. Report and save ---
    print(f"\nClean: {len(maps)} wafers, {len(np.unique(combo_id))} combinations")
    print(f"{'Split':<6} {'Size':>6}  " + "  ".join(f"{d[:6]:>6}" for d in DEFECTS))
    for name, part in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        per_defect = labels[part].sum(axis=0)
        print(f"{name:<6} {len(part):>6}  " + "  ".join(f"{n:>6}" for n in per_defect))

    np.savez_compressed(
        args.out, maps=maps, labels=labels,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nSaved {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()