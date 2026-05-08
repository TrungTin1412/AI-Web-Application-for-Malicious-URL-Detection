import re
import math
import numpy as np
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "secure",
    "update",
    "account",
    "signin",
    "bank",
    "confirm",
    "password",
    "paypal",
]


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0

    probs = [text.count(char) / len(text) for char in set(text)]
    return -sum(p * math.log2(p) for p in probs)


def has_ip_address(netloc: str) -> int:
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    return int(bool(re.match(ip_pattern, netloc)))


def count_special_chars(text: str) -> int:
    return len(re.findall(r"[^a-zA-Z0-9]", text))


def extract_features(url: str) -> np.ndarray:
    parsed = urlparse(url)

    full_url = url.strip()
    netloc = parsed.netloc
    path = parsed.path
    query = parsed.query

    url_length = len(full_url)
    domain_length = len(netloc)
    path_length = len(path)
    query_length = len(query)

    num_dots = full_url.count(".")
    num_hyphens = full_url.count("-")
    num_underscores = full_url.count("_")
    num_slashes = full_url.count("/")
    num_question_marks = full_url.count("?")
    num_equal_signs = full_url.count("=")
    num_ampersands = full_url.count("&")
    num_digits = sum(char.isdigit() for char in full_url)

    num_special_chars = count_special_chars(full_url)
    subdomain_depth = max(0, len(netloc.split(".")) - 2)

    digit_ratio = num_digits / url_length if url_length > 0 else 0.0
    special_char_ratio = num_special_chars / url_length if url_length > 0 else 0.0

    entropy_url = shannon_entropy(full_url)
    entropy_domain = shannon_entropy(netloc)

    ip_flag = has_ip_address(netloc)

    lowered = full_url.lower()
    keyword_flags = [int(keyword in lowered) for keyword in SUSPICIOUS_KEYWORDS]

    features = [
        url_length,
        domain_length,
        path_length,
        query_length,
        num_dots,
        num_hyphens,
        num_underscores,
        num_slashes,
        num_question_marks,
        num_equal_signs,
        num_ampersands,
        num_digits,
        digit_ratio,
        num_special_chars,
        special_char_ratio,
        subdomain_depth,
        ip_flag,
        entropy_url,
        entropy_domain,
        *keyword_flags,
    ]

    return np.array(features, dtype=np.float32)


def get_feature_names() -> list[str]:
    return [
        "url_length",
        "domain_length",
        "path_length",
        "query_length",
        "num_dots",
        "num_hyphens",
        "num_underscores",
        "num_slashes",
        "num_question_marks",
        "num_equal_signs",
        "num_ampersands",
        "num_digits",
        "digit_ratio",
        "num_special_chars",
        "special_char_ratio",
        "subdomain_depth",
        "has_ip_address",
        "entropy_url",
        "entropy_domain",
        *[f"contains_{kw}" for kw in SUSPICIOUS_KEYWORDS],
    ]