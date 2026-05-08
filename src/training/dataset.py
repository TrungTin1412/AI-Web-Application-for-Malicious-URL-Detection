import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def _to_tensor(data, dtype=None):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data if dtype is None else data.to(dtype=dtype)
    return torch.as_tensor(data, dtype=dtype)


def _get_size(name, value):
    if value is None:
        return None
    return len(value) if name == "raw_urls" else value.size(0)


def _validate_sample_count(**items):
    sample_count = None
    for name, value in items.items():
        size = _get_size(name, value)
        if size is None:
            continue
        if sample_count is None:
            sample_count = size
            continue
        if size != sample_count:
            raise ValueError(f"{name} contains {size} samples, expected {sample_count}.")
    if sample_count is None:
        raise ValueError("At least one dataset input must be provided.")
    return sample_count


def _validate_token_tensors(input_ids, attention_mask, token_type_ids):
    provided = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    if any(value is not None for value in provided.values()) and any(
        value is None for value in provided.values()
    ):
        raise ValueError(
            "input_ids, attention_mask, and token_type_ids must be provided together."
        )


class HybridURLDataset(Dataset):
    def __init__(
        self,
        raw_urls=None,
        url_embeddings=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        handcrafted_features=None,
        labels=None,
    ):
        self.raw_urls = list(raw_urls) if raw_urls is not None else None
        self.url_embeddings = _to_tensor(url_embeddings, dtype=torch.float32)
        self.input_ids = _to_tensor(input_ids, dtype=torch.long)
        self.attention_mask = _to_tensor(attention_mask, dtype=torch.long)
        self.token_type_ids = _to_tensor(token_type_ids, dtype=torch.long)
        self.handcrafted_features = _to_tensor(handcrafted_features, dtype=torch.float32)
        self.labels = _to_tensor(labels)

        _validate_token_tensors(
            self.input_ids,
            self.attention_mask,
            self.token_type_ids,
        )
        self.sample_count = _validate_sample_count(
            raw_urls=self.raw_urls,
            url_embeddings=self.url_embeddings,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            handcrafted_features=self.handcrafted_features,
            labels=self.labels,
        )

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        sample = {}

        if self.raw_urls is not None:
            sample["raw_url"] = self.raw_urls[index]
        if self.url_embeddings is not None:
            sample["url_embedding"] = self.url_embeddings[index]
        if self.input_ids is not None:
            sample["input_ids"] = self.input_ids[index]
            sample["attention_mask"] = self.attention_mask[index]
            sample["token_type_ids"] = self.token_type_ids[index]
        if self.handcrafted_features is not None:
            sample["handcrafted_features"] = self.handcrafted_features[index]
        if self.labels is not None:
            sample["label"] = self.labels[index]

        return sample


def build_hybrid_tensor_dataset(
    url_embeddings=None,
    handcrafted_features=None,
    labels=None,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
):
    url_embeddings = _to_tensor(url_embeddings, dtype=torch.float32)
    handcrafted_features = _to_tensor(handcrafted_features, dtype=torch.float32)
    labels = _to_tensor(labels)
    input_ids = _to_tensor(input_ids, dtype=torch.long)
    attention_mask = _to_tensor(attention_mask, dtype=torch.long)
    token_type_ids = _to_tensor(token_type_ids, dtype=torch.long)

    _validate_token_tensors(input_ids, attention_mask, token_type_ids)

    sample_count = _validate_sample_count(
        url_embeddings=url_embeddings,
        handcrafted_features=handcrafted_features,
        labels=labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    tensor_fields = [
        url_embeddings,
        input_ids,
        attention_mask,
        token_type_ids,
        handcrafted_features,
        labels,
    ]
    tensors = [tensor for tensor in tensor_fields if tensor is not None]

    if not tensors or sample_count == 0:
        raise ValueError("TensorDataset inputs cannot be empty.")

    return TensorDataset(*tensors)


def create_dataloader(dataset, batch_size=32, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
