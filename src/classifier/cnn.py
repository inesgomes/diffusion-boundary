"""Class with a simple binary CNN. Testing purposes."""

import math

from torch import nn


def pool_out(in_size, kernel, dilation=1, padding=0, stride=None):
    """Calculate the output size after a pooling layer."""
    stride = kernel if stride is None else stride
    out_size = (in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(math.floor(out_size))


class CNN(nn.Module):
    """Simple CNN with 2 conv layers and 2 max pool layers."""

    def __init__(self, img_size, num_classes, nf):
        """Init CNN with 2 conv layers and 2 max pool layers."""
        super().__init__()
        nc, nh, nw = img_size

        self.blocks = nn.ModuleList()
        block_1 = nn.Sequential(
            nn.Conv2d(nc, nf, 3, padding="same"),
        )
        self.blocks.append(block_1)
        block_2 = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.blocks.append(block_2)

        nh = pool_out(nh, 2)
        nw = pool_out(nw, 2)

        block_3 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 3, padding="same"),
        )
        self.blocks.append(block_3)
        block_4 = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.blocks.append(block_4)

        nh = pool_out(nh, 2)
        nw = pool_out(nw, 2)

        self.blocks.append(nn.Flatten())

        predictor = nn.Sequential(
            nn.Linear(nh * nw * nf * 2, 1 if num_classes == 2 else num_classes),
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1),
        )
        self.blocks.append(predictor)

    def forward(self, x, output_feature_maps=False):
        """Forward pass for training the CNN."""
        intermediate_outputs = []

        for block in self.blocks:
            x = block(x)
            intermediate_outputs.append(x)

        if intermediate_outputs[-1].shape[1] == 1:
            intermediate_outputs[-1] = intermediate_outputs[-1].flatten()

        return intermediate_outputs if output_feature_maps else intermediate_outputs[-1]
