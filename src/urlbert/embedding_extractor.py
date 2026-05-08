from pathlib import Path
import numpy as np
import torch

from src.urlbert.buildmodel import build_urlbert_encoder
from src.urlbert.url_tokenize import URLTokenizer


class URLBERTEmbeddingExtractor:
    def __init__(
        self,
        vocab_path: str,
        vocab_size: int = 5000,
        max_length: int = 128,
        device: str = "cpu",
        encoder_state_path: str | Path | None = None,
        seed: int = 1337,
    ):
        self.device = torch.device(device)
        self.max_length = max_length

        self.tokenizer = URLTokenizer(vocab_path)
        self.model = build_urlbert_encoder(
            vocab_size=vocab_size,
            encoder_state_path=encoder_state_path,
            seed=seed,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode_url(self, url: str) -> np.ndarray:
        encoded = self.tokenizer.encode(url, max_length=self.max_length)

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        token_type_ids = encoded["token_type_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )

        cls_embedding = outputs.hidden_states[-1][:, 0, :]   # shape: [1, hidden_size]
        return cls_embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_batch(self, urls: list[str]) -> np.ndarray:
        encoded = self.tokenizer.batch_encode(urls, max_length=self.max_length)

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        token_type_ids = encoded["token_type_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )

        cls_embeddings = outputs.hidden_states[-1][:, 0, :]
        return cls_embeddings.cpu().numpy().astype(np.float32)
