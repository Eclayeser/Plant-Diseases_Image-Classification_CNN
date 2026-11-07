from torch import nn

from leafapp.models.cnn import BasicCNN


class RegularisedCNN(BasicCNN):
    """Reuse the baseline CNN with one regularisation option."""

    def __init__(
        self,
        num_classes=39,
        channels=(16, 32, 64),
        image_size=128,
        *,
        regularisation,
        dropout_p=0.2,
    ):
        if regularisation not in {"dropout", "batchnorm"}:
            raise ValueError("regularisation must be 'dropout' or 'batchnorm'")
        if regularisation == "dropout" and not 0 <= dropout_p < 1:
            raise ValueError("dropout_p must be at least 0 and less than 1")

        super().__init__(
            num_classes=num_classes,
            channels=channels,
            image_size=image_size,
        )

        for i, block in enumerate(self.features):
            conv, relu, pool = block
            if regularisation == "dropout":
                layers = (conv, relu, nn.Dropout2d(p=dropout_p), pool)
            else:
                layers = (conv, nn.BatchNorm2d(conv.out_channels), relu, pool)
            self.features[i] = nn.Sequential(*layers)