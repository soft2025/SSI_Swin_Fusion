import torch
from src.models.swin import SSI_SwinFusionNet

model = SSI_SwinFusionNet(num_classes=4, ssi_input_dim=10, pretrained=False)
img = torch.randn(2, 3, 224, 224)
ssi = torch.randn(2, 10)
print("Output shape:", model(img, ssi).shape)
