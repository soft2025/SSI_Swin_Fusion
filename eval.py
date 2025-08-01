import argparse

import torch
from torch.utils.data import DataLoader

from src.data import FusionDataset
from src.models import SSI_SwinFusionNet


def evaluate(
    csv_path: str,
    checkpoint: str,
    batch_size: int = 8,
    device: str = "cpu",
) -> float:
    """Evaluate a trained :class:`SSI_SwinFusionNet` on the test split."""

    dataset = FusionDataset(csv_path, split="test")
    loader = DataLoader(dataset, batch_size=batch_size)

    model = SSI_SwinFusionNet(num_classes=len(dataset.label_map), ssi_input_dim=dataset[0][1].shape[0])
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, ssis, labels in loader:
            imgs = imgs.to(device)
            ssis = ssis.to(device)
            labels = labels.to(device)
            outputs = model(imgs, ssis)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total else 0.0
    print(f"Test accuracy: {acc:.4f}")
    return acc


def cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SSI Swin fusion model")
    parser.add_argument("--csv-path", type=str, required=True, help="Fusion CSV file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    args = parser.parse_args()
    evaluate(args.csv_path, args.checkpoint, args.batch_size, args.device)


if __name__ == "__main__":
    cli()

