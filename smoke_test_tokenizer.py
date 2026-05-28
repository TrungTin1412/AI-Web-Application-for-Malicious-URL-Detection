from src.urlbert.url_tokenize import URLTokenizer


TEST_URL = "http://secure-login-google.com/login.php"


def main() -> None:
    tokenizer = URLTokenizer("models/bert_tokenizer/vocab.txt")

    if tokenizer.tokenizer.vocab_size <= 5:
        raise RuntimeError(
            "Tokenizer loaded only special tokens. Check transformers version and vocab loading."
        )

    tokens = tokenizer.tokenizer.tokenize(TEST_URL)
    if not tokens:
        raise RuntimeError("Tokenizer returned no tokens for the smoke-test URL.")

    if all(token == "[UNK]" for token in tokens):
        raise RuntimeError(
            "Tokenizer mapped the smoke-test URL entirely to [UNK]. "
            "This usually means the vocab was not loaded correctly."
        )

    encoded = tokenizer.encode(TEST_URL)
    input_ids = encoded["input_ids"][0].tolist()
    decoded_tokens = tokenizer.tokenizer.convert_ids_to_tokens(input_ids)

    if decoded_tokens[0] != "[CLS]":
        raise RuntimeError("Encoded sequence is missing [CLS] at the beginning.")

    sep_index = encoded["attention_mask"][0].sum().item() - 1
    if decoded_tokens[sep_index] != "[SEP]":
        raise RuntimeError("Encoded sequence is missing [SEP] before padding.")

    print("Tokenizer vocab size:", tokenizer.tokenizer.vocab_size)
    print("Smoke-test tokens:", tokens)
    print("Encoded keys:", encoded.keys())
    print("Tokenizer smoke test OK")


if __name__ == "__main__":
    main()
