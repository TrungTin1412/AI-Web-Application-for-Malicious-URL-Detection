import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "extra" / "benign_urls.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "extra" / "benign_real_life.csv"


CURATED_ROWS = [
    ("https://www.google.com", "benign", "homepage", "Popular search homepage"),
    ("https://www.youtube.com", "benign", "homepage", "Popular video platform homepage"),
    ("https://www.facebook.com", "benign", "homepage", "Popular social network homepage"),
    ("https://www.wikipedia.org", "benign", "homepage", "Popular encyclopedia homepage"),
    ("https://www.reddit.com", "benign", "homepage", "Popular community homepage"),
    ("https://www.amazon.com", "benign", "homepage", "Popular ecommerce homepage"),
    ("https://www.instagram.com", "benign", "homepage", "Popular social media homepage"),
    ("https://www.netflix.com", "benign", "homepage", "Popular streaming homepage"),
    ("https://www.linkedin.com", "benign", "homepage", "Popular professional network homepage"),
    ("https://www.microsoft.com", "benign", "homepage", "Popular software company homepage"),
    ("https://www.apple.com", "benign", "homepage", "Popular device company homepage"),
    ("https://www.github.com", "benign", "homepage", "Popular developer platform homepage"),
    ("https://www.openai.com", "benign", "homepage", "Popular AI company homepage"),
    ("https://www.cloudflare.com", "benign", "homepage", "Popular infrastructure company homepage"),
    ("https://www.paypal.com", "benign", "homepage", "Popular payment platform homepage"),
    ("https://www.dropbox.com", "benign", "homepage", "Popular cloud storage homepage"),
    ("https://support.google.com", "benign", "service", "Legitimate support subdomain"),
    ("https://docs.python.org/3/", "benign", "service", "Legitimate documentation page"),
    ("https://learn.microsoft.com", "benign", "service", "Legitimate documentation site"),
    ("https://support.apple.com", "benign", "service", "Legitimate support portal"),
    ("https://help.netflix.com", "benign", "service", "Legitimate help portal"),
    ("https://docs.github.com", "benign", "service", "Legitimate docs subdomain"),
    ("https://developer.mozilla.org", "benign", "service", "Legitimate developer docs"),
    ("https://www.bbc.com/news", "benign", "content", "Legitimate news section"),
    ("https://en.wikipedia.org/wiki/Phishing", "benign", "content", "Legitimate article page"),
    ("https://github.com/openai/openai-python", "benign", "content", "Legitimate repository page"),
    ("https://www.nytimes.com", "benign", "content", "Legitimate news homepage"),
    ("https://www.theguardian.com/international", "benign", "content", "Legitimate news section"),
    ("https://stackoverflow.com/questions", "benign", "content", "Legitimate developer Q and A page"),
    ("https://www.python.org/downloads/", "benign", "content", "Legitimate download page"),
    ("https://aws.amazon.com/free", "benign", "content", "Legitimate cloud product page"),
    ("https://azure.microsoft.com/en-us/products", "benign", "content", "Legitimate cloud product page"),
    ("https://www.canva.com", "benign", "homepage", "Popular design platform homepage"),
    ("https://www.spotify.com", "benign", "homepage", "Popular audio streaming homepage"),
    ("https://www.zoom.us", "benign", "homepage", "Popular communication platform homepage"),
    ("https://www.salesforce.com", "benign", "homepage", "Popular SaaS homepage"),
    ("https://www.adobe.com", "benign", "homepage", "Popular software homepage"),
    ("https://www.ibm.com", "benign", "homepage", "Popular enterprise homepage"),
    ("https://www.oracle.com", "benign", "homepage", "Popular enterprise homepage"),
    ("https://www.nasa.gov", "benign", "homepage", "Legitimate government science site"),
]


def to_https(url: str) -> str:
    cleaned = str(url).strip()
    if cleaned.startswith("https://"):
        return cleaned
    if cleaned.startswith("http://"):
        return "https://" + cleaned[len("http://") :]
    return "https://" + cleaned.lstrip("/")


def main() -> None:
    seen = set()
    rows = []

    for row in CURATED_ROWS:
        url = row[0]
        if url not in seen:
            seen.add(url)
            rows.append(row)

    source_df = pd.read_csv(INPUT_PATH)
    auto_urls = source_df["url"].astype(str).str.strip().tolist()
    for original in auto_urls:
        url = to_https(original)
        if url in seen:
            continue
        seen.add(url)
        rows.append((url, "benign", "homepage_auto", "Popular benign homepage from curated extra list"))
        if len(rows) >= 320:
            break

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label", "group", "notes"])
        writer.writerows(rows)

    print("Generated rows:", len(rows))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
