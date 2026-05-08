from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM


_BERT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "models" / "bert_config"
_DEFAULT_ENCODER_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "saved" / "urlbert_encoder_seed1337.pt"
)
_DEFAULT_ENCODER_SEED = 1337


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_urlbert_config(vocab_size):
    config_kwargs = {
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
        "hidden_dropout_prob": 0.2,
        "vocab_size": vocab_size,
    }

    return AutoConfig.from_pretrained(str(_BERT_CONFIG_DIR), **config_kwargs)


def build_urlbert_mlm(vocab_size):
    config = _build_urlbert_config(vocab_size)
    model = AutoModelForMaskedLM.from_config(config=config)
    model.resize_token_embeddings(vocab_size)
    return model


def build_urlbert_encoder(
    vocab_size,
    *,
    encoder_state_path: str | Path | None = None,
    seed: int = _DEFAULT_ENCODER_SEED,
):
    config = _build_urlbert_config(vocab_size)
    _set_torch_seed(seed)
    model = AutoModel.from_config(config=config)
    model.resize_token_embeddings(vocab_size)

    state_path = Path(encoder_state_path) if encoder_state_path is not None else _DEFAULT_ENCODER_STATE_PATH
    if state_path.exists():
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), state_path)

    return model


def buildBERT(vocab_size):
    return build_urlbert_mlm(vocab_size)
