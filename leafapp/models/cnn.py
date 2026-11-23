from torch import nn

DEFAULT_DROPOUT_P = 0.2
DEFAULT_POOLING = "flatten"


def conv_block(in_ch, out_ch, *, regularisation=None, dropout_p=DEFAULT_DROPOUT_P):
    if regularisation not in (None, "dropout", "batchnorm"):
        raise ValueError("regularisation must be None, 'dropout' or 'batchnorm'")

    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
    if regularisation == "batchnorm":
        layers.append(nn.BatchNorm2d(out_ch))

    layers.append(nn.ReLU())
    if regularisation == "dropout":
        layers.append(nn.Dropout2d(p=dropout_p))

    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class BasicCNN(nn.Module):
    """
    Baseline CNN: a stack of conv blocks followed by a linear classifier.

    Args:
        num_classes: number of output classes.
        channels: output channels of each conv block; one block per entry,
            e.g. (16, 32, 64) builds 3 blocks.
        image_size: height/width of the (square) input images.
        pooling: "flatten" (default) flattens the final feature maps, while
            "avg" uses global average pooling for a much smaller final layer.
        
    Input [B, 3, image_size, image_size] -> output [B, num_classes] logits.
    """
    def __init__(self, num_classes=39, channels=(16, 32, 64), image_size=128, pooling=DEFAULT_POOLING):
        super().__init__()
        # Chain the blocks: each block's input channels = the previous block's output
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(self.make_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        
        # Each block halves height and width, so the final maps are
        # image_size // 2**num_blocks on each side
        if pooling == "flatten":
            final_size = image_size // 2**len(blocks)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels[-1] * final_size**2, num_classes)
            )
        elif pooling == "avg":
            # Average each feature map to one value, so the final layer gets one number per channel
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[-1], num_classes)
            )
        else:
            raise ValueError("pooling must be 'flatten' or 'avg'")

    def make_block(self, in_ch, out_ch):
        """Build one block with the baseline's default layers."""
        return conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
