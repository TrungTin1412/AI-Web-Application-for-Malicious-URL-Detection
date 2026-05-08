from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/train.csv")
OUTPUT_DIR = Path("data/sample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Original dataset shape:", df.shape)
    print("Label distribution:")
    print(df["label"].value_counts())

    target_per_class = 12500

    sample_50k = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(
            n=min(len(x), target_per_class),
            random_state=42
        ))
        .reset_index(drop=True)
    )

    output_path = OUTPUT_DIR / "sample_50k.csv"
    sample_50k.to_csv(output_path, index=False)

    print("\nSaved sample_50k.csv")
    print("Shape:", sample_50k.shape)
    print("\nNew label distribution:")
    print(sample_50k["label"].value_counts())


if __name__ == "__main__":
    main()