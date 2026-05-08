import json
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse, urlunparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency
    PlaywrightError = None
    sync_playwright = None


IGNORED_URL_TOKENS = {
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
    "html",
    "htm",
    "php",
    "asp",
    "aspx",
    "jsp",
    "index",
    "home",
    "page",
    "news",
    "www2",
    "www3",
}

LOGIN_HINTS = {
    "login",
    "signin",
    "sign in",
    "verify",
    "account",
    "password",
    "secure",
    "bank",
}


class SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignore_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.image_text_parts: list[str] = []
        self.background_image_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        attrs_dict = {key.lower(): value for key, value in attrs if key and value}
        if tag in {"script", "style", "noscript"}:
            self._ignore_depth += 1
        if tag == "title":
            self._in_title = True

        style = attrs_dict.get("style")
        if style:
            self.background_image_parts.extend(extract_background_image_tokens(style))

        if tag == "img":
            for attr_name in ("alt", "title"):
                attr_value = attrs_dict.get(attr_name)
                if attr_value:
                    normalized = normalize_html_text(attr_value)
                    if normalized:
                        self.image_text_parts.append(normalized)

            src = attrs_dict.get("src")
            if src:
                filename_text = extract_filename_text(src)
                if filename_text:
                    self.image_text_parts.append(filename_text)

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"} and self._ignore_depth > 0:
            self._ignore_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._ignore_depth > 0:
            return

        text = data.strip()
        if not text:
            return

        if self._in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)


def split_compound_token(token: str) -> list[str]:
    pieces = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", token)
    return [piece.lower() for piece in pieces if piece]


def extract_url_keywords(url: str) -> list[str]:
    tokens = re.split(r"[^a-zA-Z0-9]+", url.lower())
    keywords = []
    seen = set()

    for token in tokens:
        if len(token) < 4 or token in IGNORED_URL_TOKENS:
            continue
        if token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)

    return keywords


def extract_content_terms(text: str) -> set[str]:
    seen: set[str] = set()

    for raw_token in re.findall(r"[A-Za-z0-9]+", text):
        token = raw_token.lower()
        if len(token) >= 4:
            seen.add(token)

        for piece in split_compound_token(raw_token):
            if len(piece) >= 4:
                seen.add(piece)

    return seen


def collapse_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_html_text(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", unescape(value)).strip()
    return cleaned


def extract_filename_text(resource_url: str) -> str | None:
    parsed_resource = urlparse(resource_url)
    filename = parsed_resource.path.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    if not stem:
        return None

    normalized = normalize_html_text(stem.replace("-", " ").replace("_", " "))
    return normalized or None


def extract_background_image_tokens(style_value: str) -> list[str]:
    urls = re.findall(r"url\(([^)]+)\)", style_value, flags=re.IGNORECASE)
    extracted: list[str] = []

    for raw_url in urls:
        cleaned_url = raw_url.strip().strip("'\"")
        filename_text = extract_filename_text(cleaned_url)
        if filename_text:
            extracted.append(filename_text)

    return extracted


def extract_meta_content(html: str, key: str, attr: str = "name") -> str | None:
    pattern = (
        r"<meta[^>]*"
        + attr
        + r'\s*=\s*["\']'
        + re.escape(key)
        + r'["\'][^>]*content\s*=\s*["\']([^"\']+)["\']'
    )
    match = re.search(pattern, html, flags=re.IGNORECASE)
    if match:
        return normalize_html_text(match.group(1))

    reverse_pattern = (
        r"<meta[^>]*content\s*=\s*['\"]([^'\"]+)['\"][^>]*"
        + attr
        + r'\s*=\s*["\']'
        + re.escape(key)
        + r'["\']'
    )
    match = re.search(reverse_pattern, html, flags=re.IGNORECASE)
    if match:
        return normalize_html_text(match.group(1))

    return None


def extract_json_ld_text(html: str) -> str:
    blocks = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    texts: list[str] = []

    def walk(value) -> None:
        if isinstance(value, dict):
            for item_key, item_value in value.items():
                if item_key in {"name", "headline", "description", "alternateName"} and isinstance(item_value, str):
                    texts.append(normalize_html_text(item_value))
                else:
                    walk(item_value)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for block in blocks:
        raw = normalize_html_text(block)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        walk(parsed)

    return " ".join(texts).strip()


def fetch_html(url: str, timeout: int = 10) -> tuple[int, str]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        status_code = getattr(response, "status", 200)
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(charset, errors="replace")
        return status_code, html


def can_use_playwright() -> bool:
    return sync_playwright is not None


def fetch_rendered_html(url: str, timeout: int = 10) -> tuple[int, str]:
    if not can_use_playwright():
        raise RuntimeError("Playwright is not installed.")

    timeout_ms = timeout * 1000
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            )
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(1500)
            html = page.content()
            status_code = response.status if response is not None else 200
            return status_code, html
        finally:
            browser.close()


def swap_url_scheme(url: str, scheme: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(scheme=scheme))


def should_try_https_fallback(url: str, error_message: str) -> bool:
    lowered = error_message.lower()
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return False

    fallback_hints = (
        "10061",
        "actively refused",
        "connection refused",
        "forcibly rejected",
        "remote end closed",
        "wrong version number",
    )
    return any(hint in lowered for hint in fallback_hints)


def fetch_html_with_fallback(url: str, timeout: int = 10) -> tuple[int, str, str]:
    try:
        status_code, html = fetch_html(url, timeout=timeout)
        return status_code, html, url
    except HTTPError:
        raise
    except URLError as exc:
        error_message = str(exc.reason)
        if should_try_https_fallback(url, error_message):
            fallback_url = swap_url_scheme(url, "https")
            status_code, html = fetch_html(fallback_url, timeout=timeout)
            return status_code, html, fallback_url
        raise
    except Exception as exc:
        error_message = str(exc)
        if should_try_https_fallback(url, error_message):
            fallback_url = swap_url_scheme(url, "https")
            status_code, html = fetch_html(fallback_url, timeout=timeout)
            return status_code, html, fallback_url
        raise


def should_try_rendered_fetch(
    page_title: str,
    visible_text: str,
    image_text: str,
    background_image_text: str,
    match_score: float,
) -> bool:
    weak_content = not page_title and not visible_text and not image_text and not background_image_text
    return weak_content or match_score == 0.0


def analyze_url_content(url: str, prefer_rendered: bool = False) -> dict:
    try:
        status_code, html, fetched_url = fetch_html_with_fallback(url)
    except HTTPError as exc:
        return {
            "content_check_run": False,
            "fetch_status": "failed",
            "status_code": exc.code,
            "error": f"HTTP error while fetching page content: {exc.code}",
        }
    except URLError as exc:
        return {
            "content_check_run": False,
            "fetch_status": "failed",
            "status_code": None,
            "error": f"Network error while fetching page content: {exc.reason}",
        }
    except Exception as exc:
        return {
            "content_check_run": False,
            "fetch_status": "failed",
            "status_code": None,
            "error": f"Unexpected error while fetching page content: {exc}",
        }

    def build_content_view(source_html: str) -> dict:
        parser = SimpleHTMLTextExtractor()
        parser.feed(source_html)

        html_title = " ".join(parser.title_parts).strip()
        meta_title = (
            extract_meta_content(source_html, "title")
            or extract_meta_content(source_html, "og:title", attr="property")
            or extract_meta_content(source_html, "twitter:title")
        )
        json_ld_text = extract_json_ld_text(source_html)

        page_title_local = meta_title or html_title
        visible_text_local = " ".join(parser.text_parts).strip()
        image_text_local = " ".join(parser.image_text_parts).strip()
        background_image_text_local = " ".join(parser.background_image_parts).strip()
        content_sources_local = " ".join(
            part
            for part in [
                page_title_local,
                visible_text_local,
                json_ld_text,
                image_text_local,
                background_image_text_local,
            ]
            if part
        ).strip()
        content_blob_local = content_sources_local.lower()
        content_terms_local = extract_content_terms(content_sources_local)
        collapsed_content_blob_local = collapse_token(content_sources_local)

        url_keywords_local = extract_url_keywords(fetched_url)
        matched_keywords_local = []
        seen_matches_local = set()

        for keyword in url_keywords_local:
            keyword_parts = [keyword, *split_compound_token(keyword)]
            normalized_parts = [part.lower() for part in keyword_parts if len(part) >= 4]
            collapsed_keyword = collapse_token(keyword)

            direct_match = any(part in content_terms_local or part in content_blob_local for part in normalized_parts)
            collapsed_match = len(collapsed_keyword) >= 4 and collapsed_keyword in collapsed_content_blob_local

            if direct_match or collapsed_match:
                if keyword not in seen_matches_local:
                    seen_matches_local.add(keyword)
                    matched_keywords_local.append(keyword)

        match_score_local = (
            len(matched_keywords_local) / len(url_keywords_local)
            if url_keywords_local
            else 0.0
        )

        login_form_detected_local = "<form" in source_html.lower() and any(
            hint in content_blob_local for hint in LOGIN_HINTS
        )

        return {
            "page_title": page_title_local,
            "visible_text": visible_text_local,
            "image_text": image_text_local,
            "background_image_text": background_image_text_local,
            "content_blob": content_blob_local,
            "url_keywords": url_keywords_local,
            "matched_keywords": matched_keywords_local,
            "match_score": match_score_local,
            "login_form_detected": login_form_detected_local,
        }

    content_view = build_content_view(html)

    rendering_used = "static"
    should_render = prefer_rendered or should_try_rendered_fetch(
        content_view["page_title"],
        content_view["visible_text"],
        content_view["image_text"],
        content_view["background_image_text"],
        content_view["match_score"],
    )

    if can_use_playwright() and should_render:
        try:
            rendered_status, rendered_html = fetch_rendered_html(fetched_url)
            rendered_view = build_content_view(rendered_html)
            static_score = content_view["match_score"]
            rendered_score = rendered_view["match_score"]
            if prefer_rendered or rendered_score > static_score or (
                not content_view["page_title"] and rendered_view["page_title"]
            ):
                html = rendered_html
                status_code = rendered_status
                content_view = rendered_view
                rendering_used = "playwright"
        except (RuntimeError, PlaywrightError):
            pass

    page_title = content_view["page_title"]
    url_keywords = content_view["url_keywords"]
    matched_keywords = content_view["matched_keywords"]
    match_score = content_view["match_score"]
    login_form_detected = content_view["login_form_detected"]

    if match_score >= 0.6:
        consistency = "high"
    elif match_score >= 0.3 or (matched_keywords and login_form_detected):
        consistency = "medium"
    else:
        consistency = "low"

    if not url_keywords:
        final_assessment = "not enough brand or topic keywords in the URL"
        explanation = "The URL does not contain enough distinctive keywords for a meaningful content correlation check."
    elif consistency == "high":
        final_assessment = "content matches url context"
        explanation = "The page content contains multiple keywords that align with the URL structure and topic."
    elif consistency == "medium":
        final_assessment = "partial content match"
        explanation = "The page content partially matches the URL keywords, but the correlation is not strong."
    else:
        final_assessment = "content mismatch"
        explanation = "The page content shows weak overlap with the URL keywords, which may indicate misleading or unrelated content."

    return {
        "content_check_run": True,
        "fetch_status": "success",
        "render_source": "javascript" if rendering_used == "playwright" else "static",
        "status_code": status_code,
        "page_title": page_title,
        "content_match_score": round(match_score, 4),
        "brand_keywords_in_url": url_keywords,
        "brand_keywords_in_content": matched_keywords,
        "login_form_detected": login_form_detected,
        "content_consistency": consistency,
        "final_assessment": final_assessment,
        "explanation": explanation if rendering_used == "static" else f"{explanation} JavaScript-rendered content was used for this check.",
    }
