import torch
from leafapp.models.cnn import BasicCNN
from leafapp.models.resnet import build_resnet18

def test_basiccnn_output_shape():
    model = BasicCNN()
    x = torch.randn(8, 3, 128, 128)
    out = model(x)
    assert out.shape == (8, 39), "BasicCNN should output [8, 39] logits"
    
def test_resnet_output_shape():
    model = build_resnet18()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 39), "ResNet18 should output [2, 39] logits"
    