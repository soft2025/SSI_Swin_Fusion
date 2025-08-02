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
        self.num_features = self.backbone.feature_info[-1]["num_chs"]

        # CBAM attention module
        self.cbam = CBAM(self.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP for SSI features
        self.ssi_mlp = nn.Sequential(
            nn.Linear(ssi_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Fusion and classifier
        fusion_dim = self.num_features + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img: torch.Tensor, ssi: torch.Tensor) -> torch.Tensor:
        # 1. Extract features from Swin Transformer
        x = self.backbone.forward_features(img)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # Convert to [B, C, H, W]
        assert x.shape[1] == self.num_features, \
            f"CBAM input mismatch: got {x.shape}, expected [B, {self.num_features}, H, W]"

        # 2. Apply CBAM and Global Pooling
        x = self.cbam(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, C]

        # 3. Process SSI vector
        ssi_feat = self.ssi_mlp(ssi)  # [B, 32]

        # 4. Concatenate and classify
        feat = torch.cat((x, ssi_feat), dim=1)  # [B, C+32]
        out = self.classifier(feat)
        return out


__all__ = ["SSI_SwinFusionNet"]
