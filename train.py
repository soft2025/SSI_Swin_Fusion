import argparse
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

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
    """Training loop for SSI_SwinFusionNet with improvements."""

    print(" Chargement du dataset...")
    train_ds = FusionDataset(csv_path, split="train")
    val_ds = FusionDataset(csv_path, split="val", label_map=train_ds.label_map)

    print(f" Taille train: {len(train_ds)} | Taille val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    model = SSI_SwinFusionNet(num_classes=len(train_ds.label_map),
                              ssi_input_dim=train_ds[0][1].shape[0],
                              pretrained=True,
                              debug=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Utilisation de AMP pour accélérer l'entraînement
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (imgs, ssis, labels) in enumerate(train_loader, start=1):
            imgs, ssis, labels = imgs.to(device), ssis.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs, ssis)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)

            #if batch_idx % 20 == 0:
                #print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # --- Validation ---
        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for imgs, ssis, labels in val_loader:
                imgs, ssis, labels = imgs.to(device), ssis.to(device), labels.to(device)
                outputs = model(imgs, ssis)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_time = time.time() - epoch_start
        print(
            f" Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {running_loss / len(train_ds):.4f} "
            f"val_loss: {val_loss / len(val_ds):.4f} "
            f"val_acc: {correct / total:.4f} "
            f"(⏱ {epoch_time:.2f} sec)"
        )

    total_time = time.time() - start_time
    print(f"\n Entraînement terminé en {total_time:.2f} secondes.")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    print(" Modèle sauvegardé dans:", output_dir)


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
