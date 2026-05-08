import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from src.features.handcrafted_features import SUSPICIOUS_KEYWORDS, extract_features
from src.training.hybrid_model import MLPClassifier
from src.utils.url_normalization import normalize_url
from src.urlbert.embedding_extractor import URLBERTEmbeddingExtractor

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "saved" / "mlp_hybrid_20k_plus_benign_300.pt"
SCALER_PATH = BASE_DIR / "models" / "saved" / "hybrid_scaler_20k_plus_benign_300.pkl"
LABEL_MAP_PATH = BASE_DIR / "models" / "saved" / "label_map.json"
VOCAB_PATH = BASE_DIR / "models" / "bert_tokenizer" / "vocab.txt"
ENCODER_STATE_PATH = BASE_DIR / "models" / "saved" / "urlbert_encoder_seed1337.pt"
VOCAB_SIZE = 5000
MAX_LENGTH = 128
NUM_CLASSES = 4

_EXTRACTOR: URLBERTEmbeddingExtractor | None = None
_SCALER: Any | None = None
_MODEL: MLPClassifier | None = None
_ID_TO_LABEL: dict[int, str] | None = None


def load_label_maps():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}
    return label_map, id_to_label


def load_model(input_dim: int):
    model = MLPClassifier(input_dim=input_dim, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def initialize_hybrid_inference():
    global _EXTRACTOR, _SCALER, _MODEL, _ID_TO_LABEL

    if _EXTRACTOR is None:
        _EXTRACTOR = URLBERTEmbeddingExtractor(
            vocab_path=str(VOCAB_PATH),
            vocab_size=VOCAB_SIZE,
            max_length=MAX_LENGTH,
            device="cpu",
            encoder_state_path=str(ENCODER_STATE_PATH),
            seed=1337,
        )

    if _SCALER is None:
        _SCALER = joblib.load(SCALER_PATH)

    if _ID_TO_LABEL is None:
        _, _ID_TO_LABEL = load_label_maps()

    if _MODEL is None:
        sample_embedding = _EXTRACTOR.encode_url("https://example.com").reshape(1, -1)
        sample_features = extract_features("https://example.com").reshape(1, -1)
        sample_features_scaled = _SCALER.transform(sample_features)
        input_dim = int(sample_embedding.shape[1] + sample_features_scaled.shape[1])
        _MODEL = load_model(input_dim=input_dim)


def _detect_triggered_keywords(url: str) -> list[str]:
    lowered = url.lower()
    return [keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in lowered]


def _build_risk_explanation(url: str, predicted_label: str, confidence: float) -> str:
    triggered_keywords = _detect_triggered_keywords(url)
    if predicted_label == "benign":
        return "The hybrid model found no strong malicious indicators in this URL."

    if triggered_keywords:
        return (
            f"The hybrid model flagged this URL as {predicted_label} with "
            f"{confidence:.1%} confidence. Suspicious keywords detected: "
            f"{', '.join(triggered_keywords)}."
        )

    return (
        f"The hybrid model flagged this URL as {predicted_label} with "
        f"{confidence:.1%} confidence based on its structural and contextual patterns."
    )


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


def predict_url(url: str):
    initialize_hybrid_inference()
    normalized_url = normalize_url(url)

    embedding = _EXTRACTOR.encode_url(normalized_url).reshape(1, -1)  # type: ignore[union-attr]
    features = extract_features(normalized_url).reshape(1, -1)
    features_scaled = _SCALER.transform(features)                 # type: ignore[union-attr]

    fused = np.concatenate([embedding, features_scaled], axis=1).astype(np.float32)
    x = torch.tensor(fused, dtype=torch.float32)

    with torch.no_grad():
        logits = _MODEL(x)                                        # type: ignore[operator]
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
        pred_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        predicted_label = _ID_TO_LABEL[pred_id]                   # type: ignore[index]

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
        "model": "hybrid",
        "predicted_label": predicted_label,
        "confidence": confidence,
        "risk_level": risk_level,
        "probabilities": sorted_probabilities,
        "explanation": _build_risk_explanation(normalized_url, predicted_label, confidence),
    }


if __name__ == "__main__":
    test_url = "http://secure-login-google.com.verify-update.ru/login.php"
    result = predict_url(test_url)
    print(json.dumps(result, indent=2))
