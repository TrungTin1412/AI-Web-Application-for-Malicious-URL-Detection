import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "malicious_phish.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_url(url: str) -> str:
    if pd.isna(url):
        return ""
    return str(url).strip()


def clean_label(label: str) -> str:
    if pd.isna(label):
        return ""
    return str(label).strip().lower()


def main():
    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Rename columns if needed
    if "type" in df.columns:
        df = df.rename(columns={"type": "label"})
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'url' and 'label' columns.")

    # Keep only necessary columns
    df = df[["url", "label"]].copy()

    # Clean
    df["url"] = df["url"].apply(clean_url)
    df["label"] = df["label"].apply(clean_label)

    # Remove empty rows
    df = df[(df["url"] != "") & (df["label"] != "")]

    # Keep only expected labels
    valid_labels = {"benign", "phishing", "malware", "defacement"}
    df = df[df["label"].isin(valid_labels)]

    print("After cleaning:", df.shape)
    print("Label distribution before dedup:")
    print(df["label"].value_counts())

    # Remove exact duplicates
    df = df.drop_duplicates(subset=["url", "label"])

    # Remove conflicting labels for the same URL
    label_counts = df.groupby("url")["label"].nunique()
    conflicting_urls = label_counts[label_counts > 1].index
    if len(conflicting_urls) > 0:
        print(f"Removing {len(conflicting_urls)} conflicting URLs")
        df = df[~df["url"].isin(conflicting_urls)]

    # Remove duplicate URLs after conflict removal
    df = df.drop_duplicates(subset=["url"])

    print("After deduplication:", df.shape)
    print("Final label distribution:")
    print(df["label"].value_counts())

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=42
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42
    )

    print("\nSplit sizes:")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation label distribution:")
    print(val_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nSaved cleaned datasets to data/processed/")


if __name__ == "__main__":
    main()
