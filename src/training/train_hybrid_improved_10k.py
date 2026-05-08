import json
from pathlib import Path
import copy

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from src.training.hybrid_model import MLPClassifier


LABEL_MAP = {
    "benign": 0,
    "phishing": 1,
    "malware": 2,
    "defacement": 3,
}

SEED = 1337


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    # Load raw components
    embeddings = np.load("data/sample/sample_10k_embeddings.npy")
    features = np.load("data/sample/sample_10k_features.npy")
    labels = np.load("data/sample/sample_10k_labels.npy")

    print("Embeddings shape:", embeddings.shape)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # Split BEFORE scaling
    emb_train, emb_val, feat_train, feat_val, y_train, y_val = train_test_split(
        embeddings,
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Scale handcrafted only
    scaler = StandardScaler()
    feat_train_scaled = scaler.fit_transform(feat_train)
    feat_val_scaled = scaler.transform(feat_val)

    # Fuse after scaling
    X_train = np.concatenate([emb_train, feat_train_scaled], axis=1).astype(np.float32)
    X_val = np.concatenate([emb_val, feat_val_scaled], axis=1).astype(np.float32)

    print("Fused train shape:", X_train.shape)
    print("Fused val shape:", X_val.shape)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # DataLoader (mini-batch)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
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

    torch.save(model.state_dict(), output_dir / "mlp_hybrid_10k_improved.pt")
    joblib.dump(scaler, output_dir / "hybrid_scaler_10k.pkl")

    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

    print("\nSaved model to models/saved/mlp_hybrid_10k_improved.pt")
    print("Saved scaler to models/saved/hybrid_scaler_10k.pkl")
    print("Saved label_map.json")


if __name__ == "__main__":
    main()
