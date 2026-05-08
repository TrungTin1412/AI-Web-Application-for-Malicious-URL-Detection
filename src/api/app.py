from collections import OrderedDict
from contextlib import asynccontextmanager
from copy import deepcopy
from time import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.content.check_content import analyze_url_content
from src.inference.predict import initialize_hybrid_inference, predict_url
from src.inference.predict_baseline import initialize_baseline_inference, predict_url_baseline
from src.utils.url_normalization import normalize_url


class PredictionCache:
    def __init__(self, max_size: int = 256, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def get(self, key: str) -> dict | None:
        now = time()
        entry = self._store.get(key)
        if entry is None:
            return None

        expires_at, value = entry
        if expires_at <= now:
            self._store.pop(key, None)
            return None

        self._store.move_to_end(key)
        return deepcopy(value)

    def set(self, key: str, value: dict) -> None:
        expires_at = time() + self.ttl_seconds
        self._store[key] = (expires_at, deepcopy(value))
        self._store.move_to_end(key)

        while len(self._store) > self.max_size:
            self._store.popitem(last=False)


prediction_cache = PredictionCache()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_baseline_inference()
    initialize_hybrid_inference()
    yield


app = FastAPI(
    title="Malicious URL Detection API",
    description="Baseline 10K vs Hybrid Improved 10K malicious URL detection",
    version="1.1",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class URLRequest(BaseModel):
    url: str = Field(..., min_length=3, description="The URL to classify")


class ModelPrediction(BaseModel):
    model: str
    predicted_label: str
    confidence: float
    risk_level: str
    probabilities: list[dict[str, float | str]]
    explanation: str | None = None


class PredictionResponse(BaseModel):
    url: str
    normalized_url: str
    summary: ModelPrediction
    models: dict[str, ModelPrediction]
    meta: dict[str, object]


class ContentCheckRequest(BaseModel):
    url: str = Field(..., min_length=3, description="The URL whose content should be inspected")
    predicted_label: str | None = None
    risk_level: str | None = None
    force: bool = False


class ContentCheckResponse(BaseModel):
    url: str
    normalized_url: str
    predicted_label: str | None = None
    risk_level: str | None = None
    content_check_run: bool
    fetch_status: str
    render_source: str | None = None
    status_code: int | None = None
    page_title: str | None = None
    content_match_score: float | None = None
    brand_keywords_in_url: list[str] = []
    brand_keywords_in_content: list[str] = []
    login_form_detected: bool | None = None
    content_consistency: str | None = None
    final_assessment: str | None = None
    final_decision: str | None = None
    explanation: str | None = None
    error: str | None = None

@app.get("/")
def root():
    return {
        "message": "API is running",
        "docs": "/docs",
        "health": "ok",
    }


def should_auto_run_content_check(risk_level: str | None) -> bool:
    return risk_level in {"uncertain", "medium risk"}


def build_final_content_decision(
    risk_level: str | None,
    content_result: dict,
) -> tuple[str | None, str | None]:
    fetch_status = content_result.get("fetch_status")
    consistency = content_result.get("content_consistency")

    if fetch_status != "success":
        if risk_level == "uncertain":
            return (
                "needs review",
                "The URL-based prediction is uncertain, and the page content could not be fetched for further verification.",
            )
        if risk_level == "medium risk":
            return (
                "suspicious",
                "The URL appears moderately risky, and the page content could not be fetched for additional verification.",
            )
        if risk_level == "likely safe":
            return (
                "verification unavailable",
                "The URL appears likely safe, but the page content could not be fetched to confirm that assessment.",
            )
        return None, content_result.get("explanation")

    if risk_level == "uncertain":
        if consistency == "high":
            return (
                "likely safe",
                "The URL-based prediction was uncertain, but the fetched page content aligns strongly with the URL context.",
            )
        if consistency == "medium":
            return (
                "needs review",
                "The URL-based prediction was uncertain, and the page content only partially matches the URL context.",
            )
        return (
            "suspicious",
            "The URL-based prediction was uncertain, and the page content shows weak alignment with the URL context.",
        )

    if risk_level == "medium risk":
        if consistency == "high":
            return (
                "needs review",
                "The URL appears moderately risky, but the page content aligns well enough that manual review is recommended before blocking it.",
            )
        if consistency == "medium":
            return (
                "suspicious",
                "The URL appears moderately risky and the page content only partially supports the URL context.",
            )
        return (
            "high risk",
            "The URL appears moderately risky and the fetched page content shows weak alignment with the URL context.",
        )

    return content_result.get("final_assessment"), content_result.get("explanation")


@app.post("/predict-url", response_model=PredictionResponse)
def predict(request: URLRequest):
    try:
        normalized_url = normalize_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cached_response = prediction_cache.get(normalized_url)
    if cached_response is not None:
        cached_response["meta"]["cached"] = True
        return cached_response

    baseline_result = predict_url_baseline(normalized_url)
    hybrid_result = predict_url(normalized_url)
    response = {
        "url": request.url,
        "normalized_url": normalized_url,
        "summary": hybrid_result,
        "models": {
            "baseline": baseline_result,
            "hybrid": hybrid_result,
        },
        "meta": {
            "cached": False,
            "primary_model": "hybrid",
            "compared_models": ["baseline", "hybrid"],
        },
    }
    prediction_cache.set(normalized_url, response)

    return response


@app.post("/check-content", response_model=ContentCheckResponse)
def check_content(request: ContentCheckRequest):
    try:
        normalized_url = normalize_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not request.force and not should_auto_run_content_check(request.risk_level):
        return {
            "url": request.url,
            "normalized_url": normalized_url,
            "predicted_label": request.predicted_label,
            "risk_level": request.risk_level,
            "content_check_run": False,
            "fetch_status": "skipped",
            "render_source": None,
            "final_decision": None,
            "explanation": "Content check is optional for this risk level. Set force=true to run it manually.",
        }

    try:
        content_result = analyze_url_content(normalized_url, prefer_rendered=request.force)
        final_decision, final_explanation = build_final_content_decision(
            request.risk_level,
            content_result,
        )
        return {
            "url": request.url,
            "normalized_url": normalized_url,
            "predicted_label": request.predicted_label,
            "risk_level": request.risk_level,
            **content_result,
            "final_decision": final_decision,
            "explanation": final_explanation or content_result.get("explanation"),
        }
    except Exception as exc:
        return {
            "url": request.url,
            "normalized_url": normalized_url,
            "predicted_label": request.predicted_label,
            "risk_level": request.risk_level,
            "content_check_run": False,
            "fetch_status": "failed",
            "render_source": None,
            "final_decision": None,
            "explanation": "The content check could not be completed.",
            "error": str(exc),
        }
