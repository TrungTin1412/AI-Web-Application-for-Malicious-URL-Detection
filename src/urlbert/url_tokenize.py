from pathlib import Path

from transformers import BertTokenizer
from transformers.models.bert.tokenization_bert import load_vocab


_DEFAULT_VOCAB_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "bert_tokenizer" / "vocab.txt"
)


class URLTokenizer:
    def __init__(self, vocab_path=None):
        resolved_vocab_path = (Path(vocab_path) if vocab_path else _DEFAULT_VOCAB_PATH).resolve()
        self.vocab_path = resolved_vocab_path

        try:
            # transformers 4.x expects the legacy `vocab_file` argument.
            self.tokenizer = BertTokenizer(vocab_file=str(self.vocab_path))
        except TypeError:
            # transformers 5.x expects a loaded vocab dictionary instead.
            self.tokenizer = BertTokenizer(vocab=load_vocab(str(self.vocab_path)))

        if self.tokenizer.vocab_size <= 5:
            try:
                self.tokenizer = BertTokenizer(vocab=load_vocab(str(self.vocab_path)))
            except TypeError:
                pass

        if self.tokenizer.vocab_size <= 5:
            raise ValueError(
                f"Failed to load URL tokenizer vocab from {self.vocab_path}. "
                "Only special tokens were loaded."
            )

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
