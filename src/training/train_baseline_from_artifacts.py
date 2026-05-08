import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


LABEL_MAP = {
    "benign": 0,
    "phishing": 1,
    "malware": 2,
    "defacement": 3,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--scaler_out", required=True)
    args = parser.parse_args()

    X = np.load(args.features)
    y = np.load(args.labels)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_val_scaled)

    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro")

    print("\nValidation Accuracy:", round(acc, 4))
    print("Validation Macro-F1:", round(macro_f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_val, preds))

    model_out = Path(args.model_out)
    scaler_out = Path(args.scaler_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    scaler_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)
    joblib.dump(scaler, scaler_out)

    with open(model_out.parent / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

    print("\nSaved", model_out)
    print("Saved", scaler_out)
    print("Saved label_map.json")


if __name__ == "__main__":
    main()
