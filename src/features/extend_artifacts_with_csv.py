import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.handcrafted_features import extract_features
from src.utils.url_normalization import normalize_url
from src.urlbert.embedding_extractor import URLBERTEmbeddingExtractor


LABEL_MAP = {
    "benign": 0,
    "phishing": 1,
    "malware": 2,
    "defacement": 3,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--augmented_csv", required=True)
    parser.add_argument("--base_embeddings", required=True)
    parser.add_argument("--base_features", required=True)
    parser.add_argument("--base_labels", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    base_df = pd.read_csv(args.base_csv)[["url", "label"]].dropna().copy()
    augmented_df = pd.read_csv(args.augmented_csv)[["url", "label"]].dropna().copy()

    base_df["normalized_url"] = base_df["url"].map(normalize_url)
    augmented_df["normalized_url"] = augmented_df["url"].map(normalize_url)

    added_df = augmented_df[~augmented_df["normalized_url"].isin(base_df["normalized_url"])].copy()
    added_df = added_df.drop_duplicates(subset=["normalized_url"])

    base_embeddings = np.load(args.base_embeddings)
    base_features = np.load(args.base_features)
    base_labels = np.load(args.base_labels)

    if added_df.empty:
        output_prefix = Path(args.output_prefix)
        np.save(f"{output_prefix}_embeddings.npy", base_embeddings)
        np.save(f"{output_prefix}_features.npy", base_features)
        np.save(f"{output_prefix}_labels.npy", base_labels)
        print("No new URLs to add. Copied base artifacts to new prefix.")
        return

    extractor = URLBERTEmbeddingExtractor(
        vocab_path="models/bert_tokenizer/vocab.txt",
        vocab_size=5000,
        max_length=128,
        device="cpu",
        encoder_state_path="models/saved/urlbert_encoder_seed1337.pt",
        seed=1337,
    )

    added_urls = added_df["normalized_url"].tolist()
    embeddings = []
    for i in tqdm(range(0, len(added_urls), args.batch_size), desc="Encoding added benign URLs"):
        batch_urls = added_urls[i:i + args.batch_size]
        embeddings.append(extractor.encode_batch(batch_urls))

    added_embeddings = np.vstack(embeddings).astype(np.float32)
    added_features = np.vstack(
        [extract_features(url) for url in tqdm(added_urls, desc="Building handcrafted features")]
    ).astype(np.float32)
    added_labels = added_df["label"].map(LABEL_MAP).to_numpy().astype(np.int64)

    out_embeddings = np.vstack([base_embeddings, added_embeddings]).astype(np.float32)
    out_features = np.vstack([base_features, added_features]).astype(np.float32)
    out_labels = np.concatenate([base_labels, added_labels]).astype(np.int64)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{output_prefix}_embeddings.npy", out_embeddings)
    np.save(f"{output_prefix}_features.npy", out_features)
    np.save(f"{output_prefix}_labels.npy", out_labels)

    print("Base rows:", len(base_df))
    print("Added rows:", len(added_df))
    print("Final embeddings shape:", out_embeddings.shape)
    print("Final features shape:", out_features.shape)
    print("Final labels shape:", out_labels.shape)
    print("Saved prefix:", output_prefix)


if __name__ == "__main__":
    main()
