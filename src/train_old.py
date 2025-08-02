import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm

from src.data import FusionDataset


class FusionModel(nn.Module):
    """Simple fusion model using a Swin Transformer backbone and an SSI branch."""

    def __init__(self, ssi_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )
        self.fc_ssi = nn.Linear(ssi_dim, 128)
        self.classifier = nn.Linear(self.backbone.num_features + 128, num_classes)

    def forward(self, images: torch.Tensor, ssi_vecs: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(images)
        ssi_feat = F.relu(self.fc_ssi(ssi_vecs))
        feat = torch.cat((img_feat, ssi_feat), dim=1)
        return self.classifier(feat)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, vecs, labels in loader:
        imgs = imgs.to(device)
        vecs = vecs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, vecs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, vecs, labels in loader:
            imgs = imgs.to(device)
            vecs = vecs.to(device)
            labels = labels.to(device)
            outputs = model(imgs, vecs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    train_ds = FusionDataset(args.csv_path, split="train")
    val_ds = FusionDataset(
        args.csv_path, split="val", label_map=train_ds.label_map
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    ssi_dim = int(np.load(train_ds.data.iloc[0]["ssi_path"]).shape[0])
    num_classes = len(train_ds.label_map)
    model = FusionModel(ssi_dim=ssi_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    ckpt_path = os.path.join(args.output_dir, "fusion_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument("--csv-path", type=str, required=True, help="Fusion CSV file")
    parser.add_argument("--output-dir", type=str, default="runs", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
