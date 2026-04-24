import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights = weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    


