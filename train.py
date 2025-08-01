import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import FusionDataset
from src.models import SSI_SwinFusionNet


def train(
    csv_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cpu",
) -> None:
    """Simple training loop for :class:`SSI_SwinFusionNet`."""

    transform = transforms.ToTensor()
    train_ds = FusionDataset(csv_path, split="train", transform=transform)
    val_ds = FusionDataset(csv_path, split="val", transform=transform, label_map=train_ds.label_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SSI_SwinFusionNet(num_classes=len(train_ds.label_map), ssi_input_dim=train_ds[0][1].shape[0])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, ssis, labels in train_loader:
            imgs = imgs.to(device)
            ssis = ssis.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, ssis)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for imgs, ssis, labels in val_loader:
                imgs = imgs.to(device)
                ssis = ssis.to(device)
                labels = labels.to(device)
                outputs = model(imgs, ssis)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {running_loss / len(train_ds):.4f} "
            f"val_loss: {val_loss / len(val_ds):.4f} "
            f"val_acc: {correct / total:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))


def cli() -> None:
    parser = argparse.ArgumentParser(description="Train SSI Swin fusion model")
    parser.add_argument("--csv-path", type=str, required=True, help="Fusion CSV file")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    args = parser.parse_args()
    train(args.csv_path, args.output_dir, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    cli()

