from torch import nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class BasicCNN(nn.Module):
    """
    Baseline 3-block CNN. Expects [B, 3, 128, 128] input, returns [B, num_classes] logits."""
    def __init__(self, num_classes=39):
        super().__init__()
        self.features = nn.Sequential(
        #input channels, output channels
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            #  channels * height * width
            nn.Linear(64 * 16 * 16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x