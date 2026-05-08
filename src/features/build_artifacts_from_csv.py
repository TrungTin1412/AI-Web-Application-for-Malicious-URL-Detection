from pathlib import Path
import argparse
import pandas as pd
import numpy as np
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    urls = [normalize_url(url) for url in df["url"].tolist()]
    labels = df["label"].map(LABEL_MAP).to_numpy().astype(np.int64)

    extractor = URLBERTEmbeddingExtractor(
        vocab_path="models/bert_tokenizer/vocab.txt",
        vocab_size=5000,
        max_length=128,
        device="cpu",
        encoder_state_path="models/saved/urlbert_encoder_seed1337.pt",
        seed=1337,
    )

    embeddings = []
    handcrafted = []

    for url in tqdm(urls, desc=f"Building artifacts for {args.input_csv}"):
        embeddings.append(extractor.encode_url(url))
        handcrafted.append(extract_features(url))

    embeddings = np.vstack(embeddings).astype(np.float32)
    handcrafted = np.vstack(handcrafted).astype(np.float32)

    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.save(f"{out}_embeddings.npy", embeddings)
    np.save(f"{out}_features.npy", handcrafted)
    np.save(f"{out}_labels.npy", labels)

    print("Embeddings shape:", embeddings.shape)
    print("Features shape:", handcrafted.shape)
    print("Labels shape:", labels.shape)
    print("Saved prefix:", out)


if __name__ == "__main__":
    main()
