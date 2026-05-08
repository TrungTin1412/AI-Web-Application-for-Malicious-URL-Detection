from pathlib import Path
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
    input_path = Path("data/sample/sample_20k.csv")
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 256

    df = pd.read_csv(input_path)
    urls = [normalize_url(url) for url in df["url"].tolist()]
    labels = df["label"].map(LABEL_MAP).to_numpy()

    extractor = URLBERTEmbeddingExtractor(
        vocab_path="models/bert_tokenizer/vocab.txt",
        vocab_size=5000,
        max_length=128,
        device="cpu",
        encoder_state_path="models/saved/urlbert_encoder_seed1337.pt",
        seed=1337,
    )

    embeddings = []
    for i in tqdm(range(0, len(urls), batch_size), desc="Building sample_20k embeddings"):
        batch_urls = urls[i:i + batch_size]
        embeddings.append(extractor.encode_batch(batch_urls))

    handcrafted = [extract_features(url) for url in tqdm(urls, desc="Building handcrafted features")]

    embeddings = np.vstack(embeddings).astype(np.float32)
    handcrafted = np.vstack(handcrafted).astype(np.float32)
    labels = labels.astype(np.int64)

    np.save(output_dir / "sample_20k_embeddings.npy", embeddings)
    np.save(output_dir / "sample_20k_features.npy", handcrafted)
    np.save(output_dir / "sample_20k_labels.npy", labels)

    print("Embeddings shape:", embeddings.shape)
    print("Features shape:", handcrafted.shape)
    print("Labels shape:", labels.shape)
    print("NaN embeddings:", np.isnan(embeddings).sum())
    print("NaN features:", np.isnan(handcrafted).sum())
    print("Saved sample_20k artifacts")


if __name__ == "__main__":
    main()
