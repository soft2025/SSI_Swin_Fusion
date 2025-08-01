# SSI_Swin_Fusion
Projet de classification FBG avec SSI + Swin Transformer + CBAM

## Usage example

```python
from src.models import SSI_SwinFusionNet
import torch

model = SSI_SwinFusionNet(num_classes=4, ssi_input_dim=64, pretrained=False)
img = torch.randn(2, 3, 224, 224)
ssi = torch.randn(2, 64)
print(model(img, ssi).shape)  # torch.Size([2, 4])
```
