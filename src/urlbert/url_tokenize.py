from pathlib import Path

from transformers import BertTokenizer


_DEFAULT_VOCAB_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "bert_tokenizer" / "vocab.txt"
)


class URLTokenizer:
    def __init__(self, vocab_path=None):
        resolved_vocab_path = Path(vocab_path) if vocab_path else _DEFAULT_VOCAB_PATH
        self.vocab_path = resolved_vocab_path
        self.tokenizer = BertTokenizer(vocab_file=str(self.vocab_path))

    def encode(self, url, max_length=128):
        encoded = self.tokenizer(
            url,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded["token_type_ids"],
        }

    def batch_encode(self, urls, max_length=128):
        encoded = self.tokenizer(
            list(urls),
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded["token_type_ids"],
        }
