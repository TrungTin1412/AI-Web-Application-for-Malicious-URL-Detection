from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_url
from src.inference.predict_baseline import predict_url_baseline
from src.utils.url_normalization import normalize_url


def format_prediction(result: dict) -> str:
    return f"{result['predicted_label']} ({result['confidence']:.1%})"


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "extra" / "real_life_test_urls.csv"
    df = pd.read_csv(input_path)

    rows = []
    for _, row in df.iterrows():
        original_url = row["url"]
        normalized_url = normalize_url(original_url)
        baseline = predict_url_baseline(normalized_url)
        hybrid = predict_url(normalized_url)

        rows.append(
            {
                "group": row["group"],
                "expected_label": row["expected_label"],
                "original_url": original_url,
                "normalized_url": normalized_url,
                "baseline_prediction": baseline["predicted_label"],
                "baseline_confidence": baseline["confidence"],
                "hybrid_prediction": hybrid["predicted_label"],
                "hybrid_confidence": hybrid["confidence"],
                "baseline_match": baseline["predicted_label"] == row["expected_label"],
                "hybrid_match": hybrid["predicted_label"] == row["expected_label"],
                "notes": row["notes"],
            }
        )

    results = pd.DataFrame(rows)

    print("\n=== REAL-LIFE URL EVALUATION ===")
    print(f"Input file: {input_path}")
    print(f"Total URLs: {len(results)}")
    print(
        "Baseline accuracy on this list: "
        f"{results['baseline_match'].mean():.1%}"
    )
    print(
        "Hybrid accuracy on this list: "
        f"{results['hybrid_match'].mean():.1%}"
    )

    for group, group_df in results.groupby("group", sort=False):
        print(f"\n--- Group: {group} ---")
        for _, item in group_df.iterrows():
            print(f"\nURL                : {item['original_url']}")
            print(f"Normalized URL     : {item['normalized_url']}")
            print(f"Expected label     : {item['expected_label']}")
            print(
                "Baseline prediction: "
                f"{format_prediction({'predicted_label': item['baseline_prediction'], 'confidence': item['baseline_confidence']})}"
                f" | Match: {item['baseline_match']}"
            )
            print(
                "Hybrid prediction  : "
                f"{format_prediction({'predicted_label': item['hybrid_prediction'], 'confidence': item['hybrid_confidence']})}"
                f" | Match: {item['hybrid_match']}"
            )
            print(f"Notes              : {item['notes']}")


if __name__ == "__main__":
    main()
