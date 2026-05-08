import torch
from src.urlbert.buildmodel import build_urlbert_encoder
from src.urlbert.url_tokenize import URLTokenizer

device = torch.device("cpu")

tokenizer = URLTokenizer("models/bert_tokenizer/vocab.txt")
model = build_urlbert_encoder(vocab_size=5000)
model.to(device)
model.eval()

encoded = tokenizer.encode("http://secure-login-google.com/login.php", max_length=128)

input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)
token_type_ids = encoded["token_type_ids"].to(device)

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_hidden_states=True
    )

embedding = outputs.hidden_states[-1][:, 0, :]
print(embedding.shape)
print("embedding pipeline OK")
