# ToxicGuard

Vietnamese toxic comment detection API built with FastAPI, Scikit-learn, and LightGBM.

The project uses:

* TF-IDF vectorization
* Ensemble machine learning models
* Vietnamese text preprocessing
* REST API for inference

---

# Features

* Vietnamese toxic comment classification
* TF-IDF + Ensemble VotingClassifier
* FastAPI REST API
* Swagger documentation
* Config management with `.env`
* Clean project structure
* Probability prediction support

---

# Tech Stack

* Python 3.12
* FastAPI
* Scikit-learn
* LightGBM
* Pandas
* Joblib
* Pydantic

---

# Project Structure

```text
ToxicGuard/
├── pyproject.toml
├── README.md
├── models/
│   ├── toxic_ensemble_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── AIE_data.ipynb
│   └── AIO_BANANA_WARMUP-1.ipynb
│
├── src/
│   └── toxic_guard/
│       ├── __init__.py
│       ├── main.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── base.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   └── routers.py
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── prediction.py
│       │
│       └── services/
│           ├── __init__.py
│           ├── predictor.py
│           └── preprocess.py
│       
└── .env_example
```

---

# Installation

Clone repository:

```bash
git clone <your-repository-url>
cd ToxicGuard
```

Install dependencies:

```bash
uv sync --all-packages
```

---

# Environment Variables

Create `.env` file:

```env
MODEL_PATH=models/toxic_ensemble_model.pkl
TFIDF_PATH=models/tfidf_vectorizer.pkl
```

---

# Run API

```bash
PYTHONPATH=src uvicorn toxic_guard.main:app --reload
```

Server will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoint

## POST `/predict`

Classify Vietnamese text as toxic or non-toxic.

### Request

Form Data:

| Field | Type   | Description           |
| ----- | ------ | --------------------- |
| text  | string | Vietnamese text input |

Example:

```text
Đồ ngốc quá.
```

---

### Response

```json
{
  "prediction": "toxic",
  "clean_text": "đồ ngốc quá",
  "probability": {
    "non_toxic": 0.04350329286771378,
    "toxic": 0.9564967071322863
  }
}
```


# Training

Training notebook:

```text
notebooks/training.ipynb
```

Saved artifacts:

```text
models/toxic_ensemble_model.pkl
models/tfidf_vectorizer.pkl
```

---

# Example Curl Request

```bash
curl -X POST \
  "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=Đồ ngốc quá."
```

