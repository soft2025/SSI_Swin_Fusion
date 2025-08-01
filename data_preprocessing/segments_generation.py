import argparse
import os
import pickle

from hawk_data import FST
from src.data import segment_fbg, split_segments


def main(data_dir: str, output_dir: str) -> None:
    fst = FST(data_dir)
    data = getattr(fst, "data", None)
    if data is None and hasattr(fst, "load"):
        data = fst.load()

    tests_per_class = getattr(fst, "tests_per_class", None)
    if tests_per_class is None and hasattr(fst, "get_tests_per_class"):
        tests_per_class = fst.get_tests_per_class()
    if tests_per_class is None:
        raise RuntimeError("Could not obtain test/class mapping from FST")

    segments_fbg = segment_fbg(data, tests_per_class)
    split_tags = split_segments(segments_fbg)

    os.makedirs(output_dir, exist_ok=True)
    seg_path = os.path.join(output_dir, "segments_fbg.pkl")
    split_path = os.path.join(output_dir, "split_segments.pkl")
    with open(seg_path, "wb") as f:
        pickle.dump(segments_fbg, f)
    with open(split_path, "wb") as f:
        pickle.dump(split_tags, f)
    print(f"Saved segments to {seg_path}")
    print(f"Saved split tags to {split_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate FBG segments and split")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/content/drive/MyDrive/FST",
        help="Directory containing the raw FBG dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive",
        help="Directory where the pickle files will be stored",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)


if __name__ == "__main__":
    cli()
