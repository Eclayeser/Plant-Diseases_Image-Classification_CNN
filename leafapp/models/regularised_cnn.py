from leafapp.models import cnn


class RegularisedCNN(cnn.BasicCNN):
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

        self._regularisation = regularisation
        self._dropout_p = dropout_p
        super().__init__(
            num_classes=num_classes,
            channels=channels,
            image_size=image_size,
        )

    def make_block(self, in_ch, out_ch):
        return cnn.conv_block(
            in_ch,
            out_ch,
            regularisation=self._regularisation,
            dropout_p=self._dropout_p,
        )
