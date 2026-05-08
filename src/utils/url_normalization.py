from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    normalized = url.strip()
    if not normalized:
        raise ValueError("URL must not be empty.")

    parsed = urlparse(normalized)
    if not parsed.scheme:
        normalized = f"http://{normalized}"
        parsed = urlparse(normalized)
    elif not parsed.netloc and "://" not in normalized:
        normalized = f"http://{normalized}"
        parsed = urlparse(normalized)

    if not parsed.netloc:
        raise ValueError("A valid URL host is required.")

    return normalized
