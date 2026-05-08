from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/train.csv")
OUTPUT_DIR = Path("data/sample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_PATH)

    sample_20k = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 5000), random_state=42))
        .reset_index(drop=True)
    )

    output_path = OUTPUT_DIR / "sample_20k.csv"
    sample_20k.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("Shape:", sample_20k.shape)
    print(sample_20k["label"].value_counts())


if __name__ == "__main__":
    main()