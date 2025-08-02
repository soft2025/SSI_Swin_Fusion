"""Swin Transformer based fusion model."""

from __future__ import annotations
import torch
from torch import nn

try:
    import timm
except Exception:
    timm = None

from .cbam import CBAM

class SSI_SwinFusionNet(nn.Module):
    """Late-fusion network combining Swin Transformer features and SSI vectors."""

    def __init__(self, num_classes: int = 4, ssi_input_dim: int = 10, pretrained: bool = True) -> None:
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required to instantiate the Swin backbone")

        # Backbone Swin Transformer
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True
        )
        
        # Nombre de canaux de sortie
        self.num_features = self.backbone.feature_info[-1]['num_chs']
        
        # CBAM
        self.cbam = CBAM(self.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP pour SSI
        self.ssi_mlp = nn.Sequential(
            nn.Linear(ssi_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Fusion et classification
        fusion_dim = self.num_features + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img: torch.Tensor, ssi: torch.Tensor) -> torch.Tensor:
        # 1. Features Swin
        features = self.backbone(img)  # Liste de 4 features
        x = features[-1]  # [B, 768, 7, 7]

        # 2. CBAM + Pool
        x = self.cbam(x)
        x = self.pool(x).flatten(1)  # [B, 768]

        # 3. SSI
        ssi_feat = self.ssi_mlp(ssi)

        # 4. Fusion
        feat = torch.cat((x, ssi_feat), dim=1)

        # 5. Classification
        return self.classifier(feat)

__all__ = ["SSI_SwinFusionNet"]
