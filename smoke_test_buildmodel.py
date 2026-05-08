from src.urlbert.buildmodel import build_urlbert_encoder

model = build_urlbert_encoder(vocab_size=5000)
print(type(model))
print("buildmodel OK")