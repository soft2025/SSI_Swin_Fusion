"""
swin.py
Implémentation du modèle fusion tardive :
- Branche image : Swin Tiny + CBAM
- Branche SSI : MLP
- Fusion tardive → Classification (4 classes par défaut)

Auteur : BRAHIM RIZOUG ZEGHLACHE
Date : 2025-08-01
"""

import timm
import torch
from torch import nn
from .cbam import CBAM


class SSI_SwinFusionNet(nn.Module):
    """
    Modèle de classification combinant :
    - Images CWT (Swin Tiny + CBAM)
    - Vecteurs SSI (MLP)
    - Fusion tardive avant la couche finale
    """
    def __init__(self, num_classes: int = 4, ssi_input_dim: int = 64, pretrained: bool = True):
        super(SSI_SwinFusionNet, self).__init__()

        # === Branche image (Swin + CBAM) ===
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,        # pas de tête FC
            global_pool=""
        )
        self.cbam = CBAM(self.backbone.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # === Branche SSI (MLP) ===
        self.ssi_mlp = nn.Sequential(
            nn.Linear(ssi_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # === Fusion tardive (Concat) ===
        fusion_dim = self.backbone.num_features + 32  # 768 + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, img, ssi):
        # Branche image
        x = self.backbone.forward_features(img)   # [B, 768, 7, 7]
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)                   # [B, 768]

        # Branche SSI
        ssi_feat = self.ssi_mlp(ssi)              # [B, 32]

        # Fusion tardive
        fused = torch.cat([x, ssi_feat], dim=1)   # [B, 800]
        out = self.classifier(fused)              # [B, num_classes]
        return out
