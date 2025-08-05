import torch
import torch.nn as nn
from timm import create_model


class SwinOnlyNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x_img):
        x = self.backbone(x_img)
        return self.head(x)
