from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(INPUT_PATH)

    sample_1k = (
        df.groupby("label", group_keys=False)[["url", "label"]]
        .apply(lambda x: x.sample(min(len(x), 250), random_state=42))
        .reset_index(drop=True)
    )

    sample_5k = (
        df.groupby("label", group_keys=False)[["url", "label"]]
        .apply(lambda x: x.sample(min(len(x), 1250), random_state=42))
        .reset_index(drop=True)
    )

    sample_1k.to_csv(OUTPUT_DIR / "sample_1k.csv", index=False)
    sample_5k.to_csv(OUTPUT_DIR / "sample_5k.csv", index=False)

    print("Saved:")
    print(" - data/sample/sample_1k.csv:", sample_1k.shape)
    print(" - data/sample/sample_5k.csv:", sample_5k.shape)

if __name__ == "__main__":
    main()
