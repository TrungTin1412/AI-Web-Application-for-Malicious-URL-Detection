# AI Web Application for Malicious URL Detection

This project detects malicious URLs by combining:

- a `Baseline model` based on handcrafted URL features
- a `Hybrid model` that combines URL embeddings with handcrafted features
- a `Content check` step for additional verification when the prediction is uncertain

The project has two main parts:

- `src/api`: `FastAPI` backend
- `frontend`: `React + Vite` frontend

## 1. Environment Requirements

- Python `3.10+`
- Node.js `18+`
- npm `9+`

Using a Python virtual environment is recommended before running the project.

## 2. Project Structure

```text
AI-Web-Application-for-Malicious-URL-Detection/
|- src/
|  |- api/                 # FastAPI backend
|  |- inference/           # Inference logic
|  |- content/             # Web page content checking
|  |- training/            # Training scripts
|  |- preprocessing/       # Data preprocessing
|  `- urlbert/             # URL embedding / tokenizer
|- frontend/               # React + Vite frontend
|- models/saved/           # Pretrained models and scalers
|- data/                   # Sample, raw, and processed datasets
`- requirements.txt
```

## 3. Backend Setup

Move into the project folder:

```powershell
cd AI-Web-Application-for-Malicious-URL-Detection
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the basic dependencies:

```powershell
pip install -r requirements.txt
pip install fastapi uvicorn joblib numpy scikit-learn pydantic
```

Notes:

- `requirements.txt` currently does not include all runtime dependencies used by the backend, so the second install command is still needed.
- After installing dependencies, run `python smoke_test_tokenizer.py` once to verify the URL tokenizer and vocab load correctly.
- If you want to use `content check` on JavaScript-rendered pages, install Playwright as well:

```powershell
pip install playwright
playwright install chromium
```

## 4. Run the Backend

From the project root, run:

```powershell
python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

After startup:

- API root: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

When the backend starts, it automatically loads:

- the baseline model
- the hybrid model
- the tokenizer / encoder
- the scaler and label map

The first startup may take a few extra seconds depending on your machine.

## 5. Frontend Setup

Open a new terminal and move into the frontend folder:

```powershell
cd frontend
npm install
```

Create a `.env` file inside `frontend` if you want to explicitly configure the API base URL:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

If you do not create `.env`, the frontend will use relative API paths. For local development, setting `VITE_API_BASE_URL` is the safest option.

## 6. Run the Frontend

Inside the `frontend` folder, run:

```powershell
npm run dev
```

Vite will usually start at:

```text
http://127.0.0.1:5173
```

The backend already allows CORS for:

- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:5173`
- `http://127.0.0.1:5173`

## 7. How to Use the System

1. Start the backend.
2. Start the frontend.
3. Open the web interface in your browser.
4. Enter a URL to analyze, for example:

```text
example.com/login
http://secure-login-google.com.verify-update.ru/login.php
```

5. Click `Analyze URL`.
6. The system will return:

- the `Hybrid model` prediction
- the `Baseline model` prediction
- the confidence score
- the risk level
- the probability distribution across labels

If the result falls into `uncertain` or `medium risk`, the system may automatically run a `content check` to compare the actual page content with the URL context.

## 8. Main API Endpoints

### `GET /`

Checks whether the backend is running.

### `POST /predict-url`

Request:

```json
{
  "url": "example.com/login"
}
```

Purpose:

- normalizes the URL
- runs both the baseline and hybrid models
- returns a combined prediction response

### `POST /check-content`

Request:

```json
{
  "url": "https://example.com/login",
  "predicted_label": "phishing",
  "risk_level": "medium risk",
  "force": true
}
```

Purpose:

- fetches page content
- compares URL keywords with page content
- detects possible login-form behavior
- provides extra verification for the URL classification result

## 9. Data and Models

The project already includes the artifacts needed for inference in:

- `models/saved/`
- `models/bert_tokenizer/`
- `models/bert_config/`

That means you can run the system demo without retraining the models.

### Dataset Setup for Local Use

Some raw and processed datasets are intentionally kept out of GitHub because:

- they can be large
- they may trigger secret-scanning or push-protection rules
- they are better managed as local-only research data

Dataset download link:

- [Google Drive dataset folder](https://drive.google.com/drive/folders/1aBTJV9DDhPpityTQIM4GLhhz3HCe8_cf?usp=sharing)

If you need to run preprocessing, evaluation, or retraining, prepare the dataset locally using this structure:

```text
data/
|- raw/
|  `- malicious_phish.csv
|- processed/
|  |- train.csv
|  |- val.csv
|  `- test.csv
|- sample/
|  |- sample_20k_plus_benign_300.csv
|  |- sample_20k_plus_benign_300_embeddings.npy
|  |- sample_20k_plus_benign_300_features.npy
|  `- sample_20k_plus_benign_300_labels.npy
`- extra/
   |- benign_real_life.csv
   |- benign_urls.csv
   `- real_life_test_urls.csv
```

Recommended workflow:

1. Keep large datasets in local storage, Google Drive, OneDrive, or another external storage location.
2. Download the dataset from the Google Drive link above.
3. Copy the required files into the `data/` folders shown above.
4. Do not commit `data/raw/` or `data/processed/` back to GitHub unless you have reviewed them carefully for size and secret-scanning issues.

If you are sharing this project with others, provide the dataset separately and ask them to place the files into the same folder structure before running training-related scripts.

## 10. Related Scripts

Some useful scripts included in the project:

- `smoke_test_buildmodel.py`
- `smoke_test_embedding.py`
- `smoke_test_embedding_extractor.py`
- `smoke_test_tokenizer.py`
- `scripts/evaluate_real_life_urls.py`

These scripts are helpful for quickly checking the model, tokenizer, or sample data pipeline.

## 11. Notes Before Pushing to Git

Before pushing, check that you are not committing:

- `.venv/`
- local-only generated files
- `node_modules/` unless the repository intentionally keeps dependencies checked in
- `frontend/.env` if it contains machine-specific local settings
- raw or processed datasets that may contain sensitive patterns or very large files

Recommended quick check:

```powershell
git status
```

Make sure only the files you actually want to commit are listed.

## 12. Possible Next Improvements

- add all missing runtime dependencies to `requirements.txt`
- add a proper `.gitignore` for Python and Node.js
- add a script to run frontend and backend together
- add Docker support for demo or production deployment
