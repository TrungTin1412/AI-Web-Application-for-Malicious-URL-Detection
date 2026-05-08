from pathlib import Path
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.url_normalization import normalize_url
from src.urlbert.embedding_extractor import URLBERTEmbeddingExtractor


def extract_and_save_embeddings(input_csv: str, output_npy: str, batch_size: int = 256):
    input_csv_path = PROJECT_ROOT / input_csv
    output_npy_path = PROJECT_ROOT / output_npy

    df = pd.read_csv(input_csv_path)
    urls = [normalize_url(url) for url in df["url"].tolist()]

    extractor = URLBERTEmbeddingExtractor(
        vocab_path=str(PROJECT_ROOT / "models" / "bert_tokenizer" / "vocab.txt"),
        vocab_size=5000,
        max_length=128,
        device="cpu",
        encoder_state_path=str(PROJECT_ROOT / "models" / "saved" / "urlbert_encoder_seed1337.pt"),
        seed=1337,
    )

    all_embeddings = []

    for i in tqdm(range(0, len(urls), batch_size), desc=f"Extracting from {input_csv}"):
        batch_urls = urls[i:i + batch_size]
        batch_embeddings = extractor.encode_batch(batch_urls)
        all_embeddings.append(batch_embeddings)

    all_embeddings = np.vstack(all_embeddings)
    np.save(output_npy_path, all_embeddings)

    print(f"Saved embeddings to {output_npy_path}")
    print("Final shape:", all_embeddings.shape)


if __name__ == "__main__":
    (PROJECT_ROOT / "data" / "embeddings").mkdir(parents=True, exist_ok=True)

    extract_and_save_embeddings("data/processed/train.csv", "data/embeddings/train_embeddings.npy")
    extract_and_save_embeddings("data/processed/val.csv", "data/embeddings/val_embeddings.npy")
    extract_and_save_embeddings("data/processed/test.csv", "data/embeddings/test_embeddings.npy")
