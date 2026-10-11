import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]


def combo_name(label):
    """Human-readable name of a label vector, e.g. 'Center+Scratch'."""
    names = [d for d, flag in zip(DEFECTS, label) if flag]
    return "+".join(names) if names else "Normal"


class WaferDataset(Dataset):
    """Wafer maps as 3-channel one-hot tensors.

    target="labels": 8-dim multi-label vector (default).
    target="class":  index of the label combination (38 classes).
    exclude: combination names to drop from this split (held-out experiment).
    """

    def __init__(self, path, split, augment=False, target="labels", exclude=()):
        data = np.load(path)
        all_labels = data["labels"]
        weights = 1 << np.arange(8)

        # Fixed class list: every label combination in the dataset, in a stable order
        combo_ids = np.unique(all_labels @ weights)
        vectors = ((combo_ids[:, None] >> np.arange(8)) & 1).astype(np.float32)
        self.class_vectors = torch.from_numpy(vectors)  # (38, 8): labels of each class
        self.class_names = [combo_name(v) for v in vectors]

        idx = data[f"{split}_idx"]
        if exclude:
            unknown = set(exclude) - set(self.class_names)
            if unknown:
                raise ValueError(f"Unknown combinations: {sorted(unknown)}")
            names = np.array([combo_name(v) for v in all_labels[idx]])
            idx = idx[~np.isin(names, list(exclude))]

        self.maps = torch.from_numpy(data["maps"][idx]).long()
        self.labels = torch.from_numpy(all_labels[idx]).float()
        self.classes = torch.from_numpy(np.searchsorted(combo_ids, all_labels[idx] @ weights)).long()
        self.augment = augment
        self.target = target

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, i):
        x = self.maps[i]
        if self.augment:
            # Rotations by 90 degrees and flips keep every defect type valid
            x = torch.rot90(x, k=int(torch.randint(4, (1,))), dims=(0, 1))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(1,))
        # Pixel values are categories (no die / good / failed), so one-hot encode them
        x = F.one_hot(x, num_classes=3).permute(2, 0, 1).float()
        y = self.labels[i] if self.target == "labels" else self.classes[i]
        return x, y