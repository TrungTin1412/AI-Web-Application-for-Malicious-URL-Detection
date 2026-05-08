from src.urlbert.embedding_extractor import URLBERTEmbeddingExtractor

extractor = URLBERTEmbeddingExtractor(
    vocab_path="models/bert_tokenizer/vocab.txt",
    vocab_size=5000,
    max_length=128,
    device="cpu",
)

url = "http://secure-login-google.com.verify-update.ru/login.php?session=123"
embedding = extractor.encode_url(url)

print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])
print("embedding extractor OK")