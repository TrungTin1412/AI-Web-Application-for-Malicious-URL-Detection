import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const RISK_TONES = {
  safe: "risk-safe",
  "likely safe": "risk-safe",
  uncertain: "risk-medium",
  "high risk": "risk-high",
  "medium risk": "risk-medium",
  suspicious: "risk-high",
  "needs review": "risk-medium",
  "verification unavailable": "risk-medium",
  inconclusive: "risk-medium",
  benign: "risk-safe",
  phishing: "risk-high",
  malware: "risk-high",
  defacement: "risk-high",
};

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function labelTitle(label) {
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function statusText(level) {
  if (!level) {
    return "Unknown";
  }

  return level
    .split(" ")
    .map((word) => labelTitle(word))
    .join(" ");
}

function renderSourceText(source) {
  if (source === "javascript") {
    return "JavaScript-rendered content";
  }
  if (source === "static") {
    return "Static HTML";
  }
  return "Unavailable";
}

function buildFinalConclusion(result, contentCheck) {
  if (!result) {
    return null;
  }

  const effectiveRiskLevel = contentCheck?.risk_level || result.summary.risk_level;
  const effectivePredictedLabel = contentCheck?.predicted_label || result.summary.predicted_label;

  if (!contentCheck || contentCheck.fetch_status === "skipped") {
    return {
      level: result.summary.predicted_label,
      title: labelTitle(result.summary.predicted_label),
      explanation:
        result.summary.explanation ||
        "The final conclusion currently follows the hybrid model because no content check decision is available.",
      source: "Hybrid model decision",
      confidence: result.summary.confidence,
      riskLevel: result.summary.risk_level,
      verificationStatus: "Model only",
    };
  }

  if (contentCheck.fetch_status !== "success") {
    if (effectiveRiskLevel === "uncertain") {
      return {
        level: "inconclusive",
        title: "Inconclusive",
        explanation:
          contentCheck.explanation ||
          "The URL-based prediction is uncertain, and page content could not be fetched for verification.",
        source: "Verification unavailable",
        confidence: result.summary.confidence,
        riskLevel: "verification unavailable",
        verificationStatus: "Verification unavailable",
      };
    }

    return {
      level: effectivePredictedLabel,
      title: labelTitle(effectivePredictedLabel),
      explanation:
        contentCheck.explanation ||
        "Page content could not be verified, so the final conclusion falls back to the hybrid model output.",
      source: "Hybrid model decision",
      confidence: result.summary.confidence,
      riskLevel: "verification unavailable",
      verificationStatus: "Verification unavailable",
    };
  }

  if (contentCheck.fetch_status === "success") {
    const consistency = contentCheck.content_consistency;
    const finalRiskLevel = contentCheck.final_decision || effectiveRiskLevel;

    if (effectiveRiskLevel === "uncertain" && consistency === "low") {
      return {
        level: "inconclusive",
        title: "Inconclusive",
        explanation:
          contentCheck.explanation ||
          "The URL-based prediction is uncertain and the fetched page content does not support a confident conclusion.",
        source: "Content-verified decision",
        confidence: null,
        riskLevel: finalRiskLevel,
        verificationStatus: "Not supported",
      };
    }

    let verificationStatus = "Model only";
    if (consistency === "high") {
      verificationStatus = "Confirmed";
    } else if (consistency === "medium") {
      verificationStatus = "Weakly supported";
    } else if (consistency === "low") {
      verificationStatus = "Not supported";
    }

    return {
      level: effectivePredictedLabel,
      title: labelTitle(effectivePredictedLabel),
      explanation:
        contentCheck.explanation ||
        "The final conclusion combines the URL prediction with content verification.",
      source: "Content-verified decision",
      confidence: null,
      riskLevel: finalRiskLevel,
      verificationStatus,
    };
  }

  return {
    level: result.summary.predicted_label,
    title: labelTitle(result.summary.predicted_label),
    explanation:
      result.summary.explanation ||
      "The final conclusion currently follows the hybrid model because no content check decision is available.",
    source: "Hybrid model decision",
    confidence: result.summary.confidence,
    riskLevel: result.summary.risk_level,
    verificationStatus: "Model only",
  };
}

async function parseJsonResponse(response) {
  const raw = await response.text();

  if (!raw) {
    return {};
  }

  try {
    return JSON.parse(raw);
  } catch {
    throw new Error("The server returned an empty or invalid response.");
  }
}

function ProbabilityList({ probabilities }) {
  return (
    <div className="probability-list">
      {probabilities.map((item) => (
        <div className="probability-row" key={item.label}>
          <div className="probability-header">
            <span>{labelTitle(item.label)}</span>
            <span>{formatPercent(item.probability)}</span>
          </div>
          <div className="probability-track">
            <div
              className="probability-fill"
              style={{ width: `${Math.max(item.probability * 100, 2)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function ContentCheckPanel({ contentCheck, onCheckContent, isCheckingContent, canAutoCheck }) {
  const detailMessage = contentCheck?.error || contentCheck?.explanation;

  return (
    <article className="panel content-card">
      <div className="panel-topline">Content Check</div>
      {contentCheck ? (
        <>
          <div className="content-head">
            <div>
              <h3>{labelTitle(contentCheck.fetch_status || "unknown")}</h3>
              <p className="muted">{detailMessage}</p>
            </div>
            {contentCheck.content_consistency ? (
              <span
                className={`risk-pill ${
                  {
                    high: "risk-safe",
                    medium: "risk-medium",
                    low: "risk-high",
                  }[contentCheck.content_consistency] || "risk-medium"
                }`}
              >
                {labelTitle(contentCheck.content_consistency)} Consistency
              </span>
            ) : null}
          </div>
          <div className="content-meta">
            <div>
              <span className="metric-label">Page title</span>
              <strong>{contentCheck.page_title || "Unavailable"}</strong>
            </div>
            <div>
              <span className="metric-label">Match score</span>
              <strong>
                {typeof contentCheck.content_match_score === "number"
                  ? formatPercent(contentCheck.content_match_score)
                  : "N/A"}
              </strong>
            </div>
            <div>
              <span className="metric-label">Login form</span>
              <strong>
                {typeof contentCheck.login_form_detected === "boolean"
                  ? contentCheck.login_form_detected
                    ? "Detected"
                    : "Not found"
                  : "N/A"}
              </strong>
            </div>
            <div>
              <span className="metric-label">Fetch error</span>
              <strong>{contentCheck.error || "None"}</strong>
            </div>
            <div>
              <span className="metric-label">Render source</span>
              <strong>{renderSourceText(contentCheck.render_source)}</strong>
            </div>
          </div>
          <div className="keyword-grid">
            <div>
              <span className="metric-label">URL keywords</span>
              <p className="keyword-list">
                {contentCheck.brand_keywords_in_url?.length
                  ? contentCheck.brand_keywords_in_url.join(", ")
                  : "None"}
              </p>
            </div>
            <div>
              <span className="metric-label">Content keywords found</span>
              <p className="keyword-list">
                {contentCheck.brand_keywords_in_content?.length
                  ? contentCheck.brand_keywords_in_content.join(", ")
                  : "None"}
              </p>
            </div>
          </div>
        </>
      ) : (
        <p className="muted">
          {canAutoCheck
            ? "This result is a good candidate for a follow-up content consistency check."
            : "Run an optional content check to compare URL keywords with the fetched page content."}
        </p>
      )}
      <button
        className="secondary-button"
        type="button"
        onClick={onCheckContent}
        disabled={isCheckingContent}
      >
        {isCheckingContent ? "Checking Content..." : contentCheck ? "Re-check Content" : "Check Content"}
      </button>
    </article>
  );
}

function ModelCard({ title, model }) {
  return (
    <article className="panel model-card">
      <div className="panel-topline">{title}</div>
      <div className="result-head">
        <div>
          <h3>{labelTitle(model.predicted_label)}</h3>
          <p className="muted">Confidence {formatPercent(model.confidence)}</p>
        </div>
        <span className={`risk-pill ${RISK_TONES[model.risk_level] || "risk-low"}`}>
          {statusText(model.risk_level)}
        </span>
      </div>
      <ProbabilityList probabilities={model.probabilities} />
    </article>
  );
}

export default function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [contentCheck, setContentCheck] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isCheckingContent, setIsCheckingContent] = useState(false);
  const finalConclusion = buildFinalConclusion(result, contentCheck);

  async function runContentCheck(predictionResult, force = true) {
    if (!predictionResult) {
      return;
    }

    setIsCheckingContent(true);
    try {
      const response = await fetch(`${API_BASE_URL}/check-content`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: predictionResult.normalized_url,
          predicted_label: predictionResult.summary.predicted_label,
          risk_level: predictionResult.summary.risk_level,
          force,
        }),
      });

      const payload = await parseJsonResponse(response);
      if (!response.ok) {
        throw new Error(payload.error || payload.detail || payload.explanation || "Content check failed.");
      }

      setContentCheck(payload);
    } catch (checkError) {
      setContentCheck({
        fetch_status: "failed",
        error: checkError.message || "Content check failed.",
        explanation: "The content fetch did not complete successfully.",
      });
    } finally {
      setIsCheckingContent(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setIsLoading(true);
    setError("");
    setContentCheck(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict-url`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      const payload = await parseJsonResponse(response);

      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed.");
      }

      setResult(payload);
      if (["uncertain", "medium risk"].includes(payload.summary.risk_level)) {
        runContentCheck(payload, false);
      }
    } catch (submitError) {
      setResult(null);
      setError(submitError.message || "Prediction failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Hybrid malicious URL detection</p>
          <h1>Score suspicious links with contextual and structural signals.</h1>
          <p className="hero-text">
            Submit a URL to compare the baseline classifier with the hybrid model and
            inspect confidence, label probabilities, and the primary risk explanation.
          </p>
        </div>
        <div className="hero-metrics panel">
          <div className="metric">
            <span className="metric-label">Primary model</span>
            <strong>Hybrid MLP</strong>
          </div>
          <div className="metric">
            <span className="metric-label">Output</span>
            <strong>Real-time risk summary</strong>
          </div>
          <div className="metric">
            <span className="metric-label">Comparison</span>
            <strong>Baseline vs hybrid</strong>
          </div>
        </div>
      </section>

      <section className="workspace">
        <form className="panel submission-form" onSubmit={handleSubmit}>
          <label className="field-label" htmlFor="url-input">
            URL to analyze
          </label>
          <div className="input-row">
            <input
              id="url-input"
              className="url-input"
              type="text"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="Enter a URL such as example.com/login"
            />
            <button className="submit-button" type="submit" disabled={isLoading}>
              {isLoading ? "Analyzing..." : "Analyze URL"}
            </button>
          </div>
          {error ? <p className="error-text">{error}</p> : null}
        </form>

        {result ? (
          <section className="results-grid">
            <article className="panel summary-card">
              <div className="panel-topline">Final conclusion</div>
              <div className="summary-main">
                <div>
                  <h2>{finalConclusion?.title || "Unknown"}</h2>
                  <p className="muted">Normalized URL: {result.normalized_url}</p>
                </div>
                <span
                  className={`risk-pill ${RISK_TONES[finalConclusion?.level] || "risk-low"}`}
                >
                  {finalConclusion?.title || "Unknown"}
                </span>
              </div>
              <p className="summary-explanation">{finalConclusion?.explanation}</p>
              <div className="summary-meta">
                <div>
                  <span className="metric-label">Decision source</span>
                  <strong>{finalConclusion?.source || "Unknown"}</strong>
                </div>
                <div>
                  <span className="metric-label">Cache</span>
                  <strong>{result.meta.cached ? "Hit" : "Miss"}</strong>
                </div>
                <div>
                  <span className="metric-label">Confidence</span>
                  <strong>
                    {typeof finalConclusion?.confidence === "number"
                      ? formatPercent(finalConclusion.confidence)
                      : "Verified by content check"}
                  </strong>
                </div>
                <div>
                  <span className="metric-label">Risk level</span>
                  <strong>{finalConclusion?.riskLevel ? statusText(finalConclusion.riskLevel) : "Unknown"}</strong>
                </div>
                <div>
                  <span className="metric-label">Verification status</span>
                  <strong>{finalConclusion?.verificationStatus || "Unknown"}</strong>
                </div>
              </div>
            </article>

            <div className="model-grid">
              <ModelCard title="Hybrid model" model={result.models.hybrid} />
              <ModelCard title="Baseline model" model={result.models.baseline} />
            </div>
            <ContentCheckPanel
              contentCheck={contentCheck}
              onCheckContent={() => runContentCheck(result, true)}
              isCheckingContent={isCheckingContent}
              canAutoCheck={["uncertain", "medium risk"].includes(result.summary.risk_level)}
            />
          </section>
        ) : (
          <section className="empty-state panel">
            <div className="panel-topline">Waiting for input</div>
            <h2>Run a prediction to inspect the model comparison.</h2>
            <p className="muted">
              The frontend is wired to the FastAPI response shape and will render the
              primary summary, per-model probabilities, and cache metadata.
            </p>
          </section>
        )}
      </section>
    </main>
  );
}
