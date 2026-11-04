from torch import nn
from torchvision import models

def build_resnet18(num_classes=39):
    """ResNet18 pretrained on ImageNet, with a new last layer for num_classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model