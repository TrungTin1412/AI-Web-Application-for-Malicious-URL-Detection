import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features.handcrafted_features import extract_features
from src.utils.url_normalization import normalize_url

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "saved" / "baseline_logreg_20k_plus_benign_300.pkl"
SCALER_PATH = BASE_DIR / "models" / "saved" / "baseline_scaler_20k_plus_benign_300.pkl"
LABEL_MAP_PATH = BASE_DIR / "models" / "saved" / "label_map.json"

_MODEL: Any | None = None
_SCALER: Any | None = None
_ID_TO_LABEL: dict[int, str] | None = None


def load_label_maps():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}
    return label_map, id_to_label


def initialize_baseline_inference():
    global _MODEL, _SCALER, _ID_TO_LABEL

    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)

    if _SCALER is None:
        _SCALER = joblib.load(SCALER_PATH)

    if _ID_TO_LABEL is None:
        _, _ID_TO_LABEL = load_label_maps()


def _get_status_level(predicted_label: str, probs: np.ndarray) -> str:
    sorted_probs = np.sort(probs)[::-1]
    top1_prob = float(sorted_probs[0])
    top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top1_prob - top2_prob

    benign_index = next(index for index, label in _ID_TO_LABEL.items() if label == "benign")  # type: ignore[union-attr]
    benign_prob = float(probs[benign_index])

    if predicted_label == "benign":
        if benign_prob >= 0.75 and margin >= 0.15:
            return "safe"
        if benign_prob >= 0.55:
            return "likely safe"
        return "uncertain"

    if top1_prob >= 0.80 and margin >= 0.20:
        return "high risk"
    if top1_prob >= 0.60 and margin >= 0.10:
        return "medium risk"
    return "uncertain"


def _get_response_label(predicted_label: str, risk_level: str) -> str:
    if predicted_label == "benign":
        return "safe"
    return risk_level


def predict_url_baseline(url: str):
    initialize_baseline_inference()
    normalized_url = normalize_url(url)

    features = extract_features(normalized_url).reshape(1, -1)
    features_scaled = _SCALER.transform(features)                 # type: ignore[union-attr]

    probs = _MODEL.predict_proba(features_scaled)[0]              # type: ignore[union-attr]
    pred_id = int(np.argmax(probs))
    confidence = float(np.max(probs))
    predicted_label = _ID_TO_LABEL[pred_id]                      # type: ignore[index]
    risk_level = _get_status_level(predicted_label, probs)
    sorted_probabilities = sorted(
        (
            {"label": _ID_TO_LABEL[i], "probability": float(probs[i])}  # type: ignore[index]
            for i in range(len(probs))
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "model": "baseline",
        "predicted_label": predicted_label,
        "confidence": confidence,
        "risk_level": risk_level,
        "probabilities": sorted_probabilities,
    }


if __name__ == "__main__":
    test_url = "http://secure-login-google.com.verify-update.ru/login.php"
    result = predict_url_baseline(test_url)
    print(json.dumps(result, indent=2))
