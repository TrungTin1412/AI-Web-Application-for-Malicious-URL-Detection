from src.urlbert.url_tokenize import URLTokenizer

tokenizer = URLTokenizer("models/bert_tokenizer/vocab.txt")
encoded = tokenizer.encode("http://secure-login-google.com/login.php")

print(encoded.keys())
print(encoded["input_ids"][:10])
print("tokenizer OK")