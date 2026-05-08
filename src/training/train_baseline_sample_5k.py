import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load handcrafted features only
X = np.load("data/sample/sample_5k_features.npy")
y = np.load("data/sample/sample_5k_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = LogisticRegression(
    max_iter=1000,
    multi_class="multinomial",
    random_state=42
)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_val_scaled)

acc = accuracy_score(y_val, preds)
macro_f1 = f1_score(y_val, preds, average="macro")

print("\nValidation Accuracy:", round(acc, 4))
print("Validation Macro-F1:", round(macro_f1, 4))
print("\nClassification Report:")
print(classification_report(y_val, preds))

output_dir = Path("models/saved")
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, output_dir / "baseline_logreg_5k.pkl")
joblib.dump(scaler, output_dir / "baseline_scaler_5k.pkl")

label_map = {
    "benign": 0,
    "phishing": 1,
    "malware": 2,
    "defacement": 3
}
with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("Saved baseline_logreg_5k.pkl")
print("Saved baseline_scaler_5k.pkl")
print("Saved label_map.json")