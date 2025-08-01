import argparse
import pickle

from src.data import generate_ssi_vector


def main(
    segments_path: str,
    split_path: str,
    output_dir: str,
    decim: int,
    lags: int,
    order: int,
    top_k: int,
) -> None:
    with open(segments_path, "rb") as f:
        segments_fbg = pickle.load(f)
    with open(split_path, "rb") as f:
        split_tags = pickle.load(f)

    for classe, tests in segments_fbg.items():
        for test_path, capteurs in tests.items():
            for capteur, segments in capteurs.items():
                tags = split_tags[classe][test_path][capteur]
                for idx, segment in enumerate(segments):
                    tag = tags[idx]
                    if tag not in {"train", "val", "test"}:
                        continue
                    generate_ssi_vector(
                        segment=segment,
                        split=tag,
                        classe=classe,
                        test_path=test_path,
                        capteur=capteur,
                        index=idx,
                        output_base=output_dir,
                        decim=decim,
                        lags=lags,
                        ordre=order,
                        top_k=top_k,
                    )


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate SSI vectors")
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
        default="/content/drive/MyDrive/SSI_vectors",
        help="Directory to store SSI vectors",
    )
    parser.add_argument("--decim", type=int, default=4, help="Decimation factor")
    parser.add_argument("--lags", type=int, default=40, help="Number of lags")
    parser.add_argument("--order", type=int, default=20, help="SSI order")
    parser.add_argument("--top-k", type=int, default=10, help="Top frequencies")
    args = parser.parse_args()

    main(
        args.segments_path,
        args.split_path,
        args.output_dir,
        args.decim,
        args.lags,
        args.order,
        args.top_k,
    )


if __name__ == "__main__":
    cli()
