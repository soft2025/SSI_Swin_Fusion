import argparse
import os
import pandas as pd

from src.data import load_weights, apply_weights_to_directory


def build_csv(cwt_dir: str, ssi_dir: str) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        img_split_dir = os.path.join(cwt_dir, split)
        if not os.path.isdir(img_split_dir):
            continue
        for root, _, files in os.walk(img_split_dir):
            for file in files:
                if file.endswith(".png"):
                    image_path = os.path.join(root, file)
                    rel = os.path.relpath(image_path, cwt_dir)
                    parts = rel.split(os.sep)
                    if len(parts) < 5:
                        continue
                    _, classe, test_id, capteur, name = parts
                    ssi_name = name.replace(".png", "_weighted.npy")
                    ssi_path = os.path.join(ssi_dir, split, classe, test_id, capteur, ssi_name)
                    rows.append({
                        "image_path": image_path,
                        "ssi_path": ssi_path,
                        "label": classe,
                        "split": split,
                    })
    return pd.DataFrame(rows)


def main(cwt_dir: str, ssi_dir: str, weights_csv: str, output_csv: str) -> None:
    if weights_csv:
        weights = load_weights(weights_csv)
        apply_weights_to_directory(ssi_dir, weights)
    df = build_csv(cwt_dir, ssi_dir)
    df.to_csv(output_csv, index=False)
    print(f"Saved fusion CSV to {output_csv}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Create fusion dataset CSV")
    parser.add_argument(
        "--cwt-dir",
        type=str,
        default="/content/drive/MyDrive/CWT_images",
        help="Directory containing CWT images",
    )
    parser.add_argument(
        "--ssi-dir",
        type=str,
        default="/content/drive/MyDrive/SSI_vectors",
        help="Directory containing SSI vectors",
    )
    parser.add_argument(
        "--weights-csv",
        type=str,
        default="",
        help="Optional weighting table to apply",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/content/drive/MyDrive/dataset_fusion_weighted.csv",
        help="Path to output CSV",
    )
    args = parser.parse_args()
    main(args.cwt_dir, args.ssi_dir, args.weights_csv, args.output_csv)


if __name__ == "__main__":
    cli()
