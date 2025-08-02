import torch
from src.models import SSI_SwinFusionNet

def test_forward_shape():
    model = SSI_SwinFusionNet(num_classes=4, ssi_input_dim=10, pretrained=False)
    imgs = torch.rand(2, 3, 224, 224)
    ssis = torch.rand(2, 10)
    out = model(imgs, ssis)
    assert out.shape == (2, 4)

