import numpy as np
import torch
import torch.nn as nn
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pathlib import Path
from src.training.hybrid_model import MLPClassifier

# Load data
X = np.load("data/sample/sample_1k_fused.npy")
y = np.load("data/sample/sample_1k_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)



model = MLPClassifier(input_dim=X.shape[1], num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    logits = model(X_val)
    preds = torch.argmax(logits, dim=1)

acc = accuracy_score(y_val.numpy(), preds.numpy())
macro_f1 = f1_score(y_val.numpy(), preds.numpy(), average="macro")

output_dir = Path("models/saved")
output_dir.mkdir(parents=True, exist_ok=True)

# Save model
torch.save(model.state_dict(), output_dir / "mlp_sample_1k.pt")
print("Saved model to models/saved/mlp_sample_1k.pt")

# Save label map
label_map = {
    "benign": 0,
    "phishing": 1,
    "malware": 2,
    "defacement": 3
}

with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)
print("Saved label_map.json")

print("\nValidation Accuracy:", round(acc, 4))
print("Validation Macro-F1:", round(macro_f1, 4))
print("\nClassification Report:")
print(classification_report(y_val.numpy(), preds.numpy()))
