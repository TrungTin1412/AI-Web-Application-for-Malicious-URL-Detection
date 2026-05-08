import argparse
from pathlib import Path

import pandas as pd

from src.utils.url_normalization import normalize_url


def load_and_prepare_benign(benign_path: Path) -> pd.DataFrame:
    benign_df = pd.read_csv(benign_path)
    benign_df["label"] = "benign"
    benign_df = benign_df[["url", "label"]].dropna()
    benign_df["url"] = benign_df["url"].astype(str).str.strip()
    benign_df = benign_df[benign_df["url"] != ""].copy()
    benign_df["normalized_url"] = benign_df["url"].map(normalize_url)
    benign_df = benign_df.drop_duplicates(subset=["normalized_url"])
    return benign_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Base dataset to augment")
    parser.add_argument("--benign_csv", required=True, help="Additional benign URLs")
    parser.add_argument("--output_csv", required=True, help="Where to save the augmented dataset")
    parser.add_argument(
        "--max_benign",
        type=int,
        default=1000,
        help="Maximum number of benign URLs to add after deduplication",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1337,
        help="Random seed for benign sampling when limiting rows",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    benign_path = Path(args.benign_csv)
    output_path = Path(args.output_csv)

    base_df = pd.read_csv(input_path)
    base_df = base_df[["url", "label"]].dropna().copy()
    base_df["url"] = base_df["url"].astype(str).str.strip()
    base_df = base_df[base_df["url"] != ""]
    base_df["normalized_url"] = base_df["url"].map(normalize_url)
    base_df = base_df.drop_duplicates(subset=["normalized_url"])

    benign_df = load_and_prepare_benign(benign_path)
    benign_df = benign_df[~benign_df["normalized_url"].isin(base_df["normalized_url"])]

    benign_available = len(benign_df)
    if args.max_benign > 0 and benign_available > args.max_benign:
        benign_df = benign_df.sample(n=args.max_benign, random_state=args.shuffle_seed)

    augmented_df = pd.concat([base_df, benign_df], ignore_index=True)
    augmented_df = augmented_df.drop_duplicates(subset=["normalized_url"])
    augmented_df = augmented_df[["url", "label"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)

    print("Base rows:", len(base_df))
    print("Benign candidates available:", benign_available)
    print("Benign rows added:", len(benign_df))
    print("Augmented rows:", len(augmented_df))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
