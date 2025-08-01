import argparse
import os
import pickle
from typing import Optional

import torch

from src.data import generate_cwt_images


def load_ffdnet(model_path: Optional[str], device: torch.device) -> torch.nn.Module:
    """Load FFDNet from timm or local weights."""
    try:
        import timm
    except Exception:
        timm = None

    if model_path and os.path.isfile(model_path):
        if timm is None:
            raise RuntimeError("timm is required to instantiate FFDNet")
        model = timm.create_model("ffdnet", pretrained=False)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    else:
        if timm is None:
            raise RuntimeError("timm is required to load pretrained FFDNet")
        model = timm.create_model("ffdnet", pretrained=True)

    model.to(device)
    model.eval()
    return model


def main(
    segments_path: str,
    split_path: str,
    output_dir: str,
    model_path: Optional[str],
    device_str: str,
) -> None:
    device = torch.device(device_str)
    with open(segments_path, "rb") as f:
        segments_fbg = pickle.load(f)
    with open(split_path, "rb") as f:
        split_tags = pickle.load(f)

    model = load_ffdnet(model_path, device)
    generate_cwt_images(segments_fbg, split_tags, output_dir, model, device)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate CWT images")
    parser.add_argument(
        "--segments-path",
        type=str,
        default="/content/drive/MyDrive/segments_fbg.pkl",
        help="Path to segments_fbg.pkl",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="/content/drive/MyDrive/split_segments.pkl",
        help="Path to split_segments.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/CWT_images",
        help="Directory to store generated images",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional path to FFDNet weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    args = parser.parse_args()
    main(args.segments_path, args.split_path, args.output_dir, args.model_path, args.device)


if __name__ == "__main__":
    cli()
