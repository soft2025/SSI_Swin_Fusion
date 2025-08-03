"""Swin Transformer based fusion model with SSI vector integration."""

from __future__ import annotations
import torch
from torch import nn

try:
    import timm
except Exception:
    timm = None

from .cbam import CBAM


class SSI_SwinFusionNet(nn.Module):
    """
    Late-fusion network combining:
    - Swin Transformer image features
    - SSI feature vectors
    """

    def __init__(self, num_classes: int = 4, ssi_input_dim: int = 10,
                 pretrained: bool = True, debug: bool = True) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("The 'timm' library is required for SSI_SwinFusionNet. "
                              "Install it via 'pip install timm'.")

        self.debug = debug

        # Backbone Swin Transformer
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True
        )

        # Nombre de canaux de sortie
        self.num_features = self.backbone.feature_info[-1]['num_chs']

        # CBAM module
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
        """
        img: [B, 3, 224, 224]
        ssi: [B, ssi_input_dim]
        """
        if self.debug:
            print(f"DEBUG ➤ Input image shape: {img.shape}")

        # 1. Extraction de features Swin
        features = self.backbone(img)
        x = features[-1]  # [B,H,W,C] ou [B,C,H,W]
        if self.debug:
            print(f"DEBUG ➤ Backbone raw output shape: {x.shape}")

        # ✅ Permutation si nécessaire
        if x.dim() == 4 and x.shape[1] != self.num_features:
            if self.debug:
                print("DEBUG ➤ Detected [B,H,W,C] format -> permuting to [B,C,H,W]")
            x = x.permute(0, 3, 1, 2).contiguous()
            if self.debug:
                print(f"DEBUG ➤ Permuted feature shape: {x.shape}")

        # 2. CBAM + Pooling
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        if self.debug:
            print(f"DEBUG ➤ Image feature shape after CBAM+pool: {x.shape}")

        # 3. SSI vector
        ssi = ssi.view(ssi.size(0), -1)
        expected_dim = self.ssi_mlp[0].in_features
        assert ssi.shape[1] == expected_dim, (
            f"Expected SSI feature dimension {expected_dim}, got {ssi.shape[1]}"
        )
        if self.debug:
            print(f"DEBUG ➤ Reshaped SSI shape: {ssi.shape}")
        ssi_feat = self.ssi_mlp(ssi)

        # 4. Fusion
        feat = torch.cat((x, ssi_feat), dim=1)
        if self.debug:
            print(f"DEBUG ➤ Fused feature shape: {feat.shape}")

        # 5. Classification
        out = self.classifier(feat)
        expected_out = self.classifier[-1].out_features
        assert out.shape[1] == expected_out, (
            f"Expected classifier output features {expected_out}, got {out.shape}"
        )
        if self.debug:
            print(f"DEBUG ➤ Output shape: {out.shape}")

        return out


__all__ = ["SSI_SwinFusionNet"]
