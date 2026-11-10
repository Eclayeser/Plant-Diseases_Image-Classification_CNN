from torch import nn
from torchvision import models

def build_resnet18(num_classes=39):
    """Pretrained ResNet18 with the last layer replaced for plant_village 
    classes.

    fc is Linear(512, 1000) for the 1000 ImageNet classes, which is
    replaced with Linear(512, num_classes). The new fc starts with random
    weights and the other layers keep the ImageNet ones.
    """
    # Downloads the ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model