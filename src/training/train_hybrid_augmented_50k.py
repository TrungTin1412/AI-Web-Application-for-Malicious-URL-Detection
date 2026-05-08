import json
from pathlib import Path
import copy

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from src.training.hybrid_model import MLPClassifier


LABEL_MAP = {
    "benign": 0,
    "phishing": 1,
    "malware": 2,
    "defacement": 3,
}


def main():
    # Load train augmented artifacts
    train_embeddings = np.load("data/sample/sample_50k_train_augmented_embeddings.npy")
    train_features = np.load("data/sample/sample_50k_train_augmented_features.npy")
    train_labels = np.load("data/sample/sample_50k_train_augmented_labels.npy")

    # Load validation artifacts
    val_embeddings = np.load("data/sample/sample_50k_val_embeddings.npy")
    val_features = np.load("data/sample/sample_50k_val_features.npy")
    val_labels = np.load("data/sample/sample_50k_val_labels.npy")

    print("Train embeddings shape:", train_embeddings.shape)
    print("Train features shape:", train_features.shape)
    print("Train labels shape:", train_labels.shape)

    print("Val embeddings shape:", val_embeddings.shape)
    print("Val features shape:", val_features.shape)
    print("Val labels shape:", val_labels.shape)

    # Scale handcrafted features using TRAIN only
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)

    # Fuse after scaling
    X_train = np.concatenate([train_embeddings, train_features_scaled], axis=1).astype(np.float32)
    X_val = np.concatenate([val_embeddings, val_features_scaled], axis=1).astype(np.float32)

    print("Fused train shape:", X_train.shape)
    print("Fused val shape:", X_val.shape)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(val_labels, dtype=torch.long)

    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Model
    model = MLPClassifier(input_dim=X_train.shape[1], num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    best_macro_f1 = -1.0
    best_acc = -1.0
    best_state = None

    epochs = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.numpy().tolist())
                all_targets.extend(batch_y.numpy().tolist())

        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Acc: {acc:.4f} - "
            f"Val Macro-F1: {macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation on best model
    model.eval()
    with torch.no_grad():
        logits = model(X_val_tensor)
        preds = torch.argmax(logits, dim=1)

    print("\nBest Validation Accuracy:", round(best_acc, 4))
    print("Best Validation Macro-F1:", round(best_macro_f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_val_tensor.numpy(), preds.numpy()))

    # Save artifacts
    output_dir = Path("models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "mlp_hybrid_50k_augmented.pt")
    joblib.dump(scaler, output_dir / "hybrid_scaler_50k_augmented.pkl")

    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

    print("\nSaved model to models/saved/mlp_hybrid_50k_augmented.pt")
    print("Saved scaler to models/saved/hybrid_scaler_50k_augmented.pkl")
    print("Saved label_map.json")


if __name__ == "__main__":
    main()