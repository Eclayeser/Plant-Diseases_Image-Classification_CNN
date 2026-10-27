import torch
from leafapp.models.cnn import BasicCNN

def test_basiccnn_output_shape():
    model = BasicCNN()
    x = torch.randn(8, 3, 128, 128)
    out = model(x)
    assert out.shape == (8, 39), "BasicCNN should output [8, 39] logits"