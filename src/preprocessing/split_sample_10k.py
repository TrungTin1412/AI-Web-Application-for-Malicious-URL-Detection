import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT_PATH = Path("data/sample/sample_10k.csv")
OUTPUT_DIR = Path("data/sample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)
    print("Label distribution:")
    print(df["label"].value_counts())

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train_path = OUTPUT_DIR / "sample_10k_train.csv"
    val_path = OUTPUT_DIR / "sample_10k_val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("\nTrain shape:", train_df.shape)
    print("Val shape:", val_df.shape)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    print("\nVal label distribution:")
    print(val_df["label"].value_counts())

    print("\nSaved files:")
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()