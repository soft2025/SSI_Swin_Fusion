import argparse
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.data.fusion_dataset import FusionDataset
from src.models.swin_only import SwinOnlyNet


def train(csv_path, output_dir, epochs=5, batch_size=8, lr=1e-4, device="cpu"):
    print("📂 Chargement du dataset...")
    train_ds = FusionDataset(csv_path, split="train", use_ssi=False)
    val_ds = FusionDataset(
        csv_path, split="val", label_map=train_ds.label_map, use_ssi=False
    )

    print(f"✅ Taille train: {len(train_ds)} | Taille val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    model = SwinOnlyNet(num_classes=len(train_ds.label_map), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
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
    torch.save(model.state_dict(), os.path.join(output_dir, "model_swin_only.pth"))
    print("✅ Modèle sauvegardé dans:", output_dir)


def cli():
    parser = argparse.ArgumentParser(description="Train Swin-only model")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    train(args.csv_path, args.output_dir, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    cli()
