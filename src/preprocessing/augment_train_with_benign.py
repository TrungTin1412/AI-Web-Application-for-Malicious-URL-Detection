from pathlib import Path
import pandas as pd

TRAIN_PATH = Path("data/sample/sample_50k_train.csv")
BENIGN_PATH = Path("data/extra/benign_urls.csv")
OUTPUT_PATH = Path("data/sample/sample_50k_train_augmented.csv")


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    benign_df = pd.read_csv(BENIGN_PATH)

    benign_df["label"] = "benign"
    benign_df = benign_df[["url", "label"]].dropna()
    benign_df["url"] = benign_df["url"].astype(str).str.strip()
    benign_df = benign_df[benign_df["url"] != ""]
    benign_df = benign_df.drop_duplicates(subset=["url"])

    before = len(benign_df)
    benign_df = benign_df[~benign_df["url"].isin(train_df["url"])]
    after = len(benign_df)

    augmented_df = pd.concat([train_df, benign_df], ignore_index=True)
    augmented_df = augmented_df.drop_duplicates(subset=["url"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(OUTPUT_PATH, index=False)

    print("Original train shape:", train_df.shape)
    print("Benign candidates:", before)
    print("Benign kept after dedup:", after)
    print("Augmented train shape:", augmented_df.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()