"""Swin Transformer based fusion model."""
from __future__ import annotations

import torch
from torch import nn

try:
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None

from .cbam import CBAM


class SSI_SwinFusionNet(nn.Module):
    """Late-fusion network combining Swin Transformer features and SSI vectors."""

    def __init__(self, num_classes: int = 4, ssi_input_dim: int = 64, pretrained: bool = True) -> None:
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required to instantiate the Swin backbone")

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.cbam = CBAM(self.backbone.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.ssi_mlp = nn.Sequential(
            nn.Linear(ssi_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        fusion_dim = self.backbone.num_features + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, img: torch.Tensor, ssi: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(img)  # [B, 768, 7, 7]
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # [B, 768]

        ssi_feat = self.ssi_mlp(ssi)  # [B, 32]

        fused = torch.cat([x, ssi_feat], dim=1)  # [B, 800]
        out = self.classifier(fused)  # [B, num_classes]
        return out


__all__ = ["SSI_SwinFusionNet"]
