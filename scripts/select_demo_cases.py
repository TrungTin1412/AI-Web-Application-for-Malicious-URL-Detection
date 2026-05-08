from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_url
from src.inference.predict_baseline import predict_url_baseline
from src.utils.url_normalization import normalize_url


def build_case(
    original_url: str,
    normalized_url: str,
    true_label: str,
    baseline: dict,
    hybrid: dict,
) -> dict:
    return {
        "original_url": original_url,
        "normalized_url": normalized_url,
        "true_label": true_label,
        "baseline_prediction": baseline["predicted_label"],
        "baseline_confidence": baseline["confidence"],
        "hybrid_prediction": hybrid["predicted_label"],
        "hybrid_confidence": hybrid["confidence"],
    }


def print_case(title: str, case: dict | None) -> None:
    print(f"\n{title}")
    if case is None:
        print("No matching case found.")
        return

    print(f"Original URL       : {case['original_url']}")
    print(f"Normalized URL     : {case['normalized_url']}")
    print(f"True label         : {case['true_label']}")
    print(
        f"Baseline prediction: {case['baseline_prediction']} "
        f"({case['baseline_confidence']:.1%})"
    )
    print(
        f"Hybrid prediction  : {case['hybrid_prediction']} "
        f"({case['hybrid_confidence']:.1%})"
    )


df = pd.read_csv(PROJECT_ROOT / "data" / "sample" / "sample_10k_val.csv")

case1 = None  # Hybrid wins
case2 = None  # Both correct
case3 = None  # Baseline wins

for _, row in df.iterrows():
    original_url = row["url"]
    normalized_url = normalize_url(original_url)
    true_label = row["label"]

    baseline = predict_url_baseline(normalized_url)
    hybrid = predict_url(normalized_url)

    baseline_label = baseline["predicted_label"]
    hybrid_label = hybrid["predicted_label"]

    if hybrid_label == true_label and baseline_label != true_label and case1 is None:
        case1 = build_case(original_url, normalized_url, true_label, baseline, hybrid)

    if hybrid_label == true_label and baseline_label == true_label and case2 is None:
        case2 = build_case(original_url, normalized_url, true_label, baseline, hybrid)

    if baseline_label == true_label and hybrid_label != true_label and case3 is None:
        if true_label == "malware":
            case3 = build_case(original_url, normalized_url, true_label, baseline, hybrid)

    if case1 and case2 and case3:
        break

print("\n=== DEMO CASES ===")
print_case("Case 1 (Hybrid wins)", case1)
print_case("Case 2 (Both correct)", case2)
print_case("Case 3 (Baseline wins)", case3)
