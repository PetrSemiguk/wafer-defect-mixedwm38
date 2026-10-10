import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]


class WaferDataset(Dataset):
    """Wafer maps as 3-channel one-hot tensors with 8-dim multi-label targets."""

    def __init__(self, path, split, augment=False):
        data = np.load(path)
        idx = data[f"{split}_idx"]
        self.maps = torch.from_numpy(data["maps"][idx]).long()
        self.labels = torch.from_numpy(data["labels"][idx]).float()
        self.augment = augment

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
        return x, self.labels[i]