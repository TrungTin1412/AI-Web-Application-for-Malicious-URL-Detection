import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Hybrid URL model training configuration")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum tokenized URL length",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for the hybrid classifier head",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the hybrid classifier head",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay applied during optimization",
    )
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def args_parser():
    return parse_args()
