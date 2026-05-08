import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.training.hybrid_model import MLPClassifier


SEED = 1337


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--scaler_out", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    set_seed(SEED)

    embeddings = np.load(args.embeddings)
    features = np.load(args.features)
    labels = np.load(args.labels)

    print("Embeddings shape:", embeddings.shape)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    emb_train, emb_val, feat_train, feat_val, y_train, y_val = train_test_split(
        embeddings,
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    scaler = StandardScaler()
    feat_train_scaled = scaler.fit_transform(feat_train)
    feat_val_scaled = scaler.transform(feat_val)

    X_train = np.concatenate([emb_train, feat_train_scaled], axis=1).astype(np.float32)
    X_val = np.concatenate([emb_val, feat_val_scaled], axis=1).astype(np.float32)

    print("Fused train shape:", X_train.shape)
    print("Fused val shape:", X_val.shape)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = MLPClassifier(input_dim=X_train.shape[1], num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_macro_f1 = -1.0
    best_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            preds = torch.argmax(logits, dim=1).numpy()

        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro")

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Acc: {acc:.4f} - "
            f"Val Macro-F1: {macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_logits = model(val_x)
        final_preds = torch.argmax(final_logits, dim=1).numpy()

    print("\nBest Validation Accuracy:", round(best_acc, 4))
    print("Best Validation Macro-F1:", round(best_macro_f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_val, final_preds))

    model_out = Path(args.model_out)
    scaler_out = Path(args.scaler_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    scaler_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_out)
    joblib.dump(scaler, scaler_out)

    print("\nSaved model to", model_out)
    print("Saved scaler to", scaler_out)


if __name__ == "__main__":
    main()
