import torch.nn as nn


def conv_block(c_in, c_out, pool):
    layers = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class WaferCNN(nn.Module):
    """Compact CNN: 52x52 input -> 13x13 feature map -> 8 independent logits."""

    def __init__(self, n_outputs=8):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, pool=True),     # 52 -> 26
            conv_block(32, 64, pool=True),    # 26 -> 13
            conv_block(64, 128, pool=False),  # keep 13x13 for sharper Grad-CAM
            conv_block(128, 256, pool=False),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        return self.head(self.features(x))