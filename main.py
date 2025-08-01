import argparse

from train import train
from eval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate the model")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument("csv_path", type=str, help="Fusion CSV file")
    train_p.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")

    eval_p = sub.add_parser("eval", help="Evaluate the model")
    eval_p.add_argument("csv_path", type=str, help="Fusion CSV file")
    eval_p.add_argument("checkpoint", type=str, help="Model weights")
    eval_p.add_argument("--batch-size", type=int, default=8)
    eval_p.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.command == "train":
        train(args.csv_path, args.output_dir, args.epochs, args.batch_size, args.lr, args.device)
    else:
        evaluate(args.csv_path, args.checkpoint, args.batch_size, args.device)


if __name__ == "__main__":
    main()

