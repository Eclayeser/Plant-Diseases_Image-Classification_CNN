from torch import nn


def conv_block(in_ch, out_ch, regularisation="none", dropout_p=0.2):
    """Build one convolution block with optional regularisation."""
    if regularisation not in {"none", "dropout", "batchnorm"}:
        raise ValueError(
            "regularisation must be 'none', 'dropout' or 'batchnorm'"
        )

    if regularisation == "dropout" and not 0 <= dropout_p < 1:
        raise ValueError("dropout_p must be at least 0 and less than 1")

    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
    ]

    # Batch normalisation acts on the convolution's output channels.
    if regularisation == "batchnorm":
        layers.append(nn.BatchNorm2d(out_ch))

    layers.append(nn.ReLU())

    # Dropout2d drops entire feature maps during training.
    if regularisation == "dropout":
        layers.append(nn.Dropout2d(p=dropout_p))

    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class BasicCNN(nn.Module):
    """CNN with baseline, dropout and batch normalisation variants.

    regularisation selects 'none', 'dropout' or 'batchnorm'.
    dropout_p is used only by the dropout variant.
    Input: [batch, 3, image_size, image_size].
    Output: [batch, num_classes] raw class scores.
    """

    def __init__(
        self,
        num_classes=39,
        channels=(16, 32, 64),
        image_size=128,
        regularisation="none",
        dropout_p=0.2,
    ):
        super().__init__()

        blocks = []
        in_ch = 3

        for out_ch in channels:
            blocks.append(
                conv_block(
                    in_ch,
                    out_ch,
                    regularisation=regularisation,
                    dropout_p=dropout_p,
                )
            )
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)

        # Each pooling layer halves the image height and width.
        final_size = image_size // 2**len(blocks)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * final_size**2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)