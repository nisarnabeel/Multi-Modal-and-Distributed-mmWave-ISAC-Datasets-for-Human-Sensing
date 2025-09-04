import torch.nn as nn
import torchvision.models as models

class GenericResNet(nn.Module):
    """ResNet18 for classification or regression."""
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        return self.model(x)
