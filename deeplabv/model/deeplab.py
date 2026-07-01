import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.net = deeplabv3_resnet50(weights=weights)
        self.net.classifier[4] = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        return self.net(x)["out"]
