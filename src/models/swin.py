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
        print("DEBUG ➤ Input shape:", img.shape)
        
        # Passage dans le backbone Swin Transformer
        features = self.backbone(img)
        x = features[-1]  # [B, 7, 7, 768]
        print("DEBUG ➤ After Swin features:", x.shape)

        # Rearrangement en [B, C, H, W] pour CBAM
        if x.shape[-1] == self.num_features:
            x = x.permute(0, 3, 1, 2)  # [B, 768, 7, 7]
        print("DEBUG ➤ After permute:", x.shape)

        # Passage dans CBAM
        x = self.cbam(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # Global pooling -> [B, 768]
        print("DEBUG ➤ After CBAM+Pool:", x.shape)

        # Partie SSI
        ssi_feat = self.ssi_mlp(ssi)
        print("DEBUG ➤ SSI features:", ssi_feat.shape)

        # Fusion des deux
        feat = torch.cat((x, ssi_feat), dim=1)
        print("DEBUG ➤ After Fusion:", feat.shape)

        # Classification finale
        out = self.classifier(feat)
        print("DEBUG ➤ Output:", out.shape)
        return out


__all__ = ["SSI_SwinFusionNet"]
