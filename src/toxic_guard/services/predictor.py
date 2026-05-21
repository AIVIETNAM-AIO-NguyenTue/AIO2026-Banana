import warnings
import joblib

from toxic_guard.config import settings
from toxic_guard.services.preprocess import preprocess_text


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)

# load once
model = joblib.load(settings.MODEL_PATH)
tfidf = joblib.load(settings.TFIDF_PATH)


LABEL_MAP = {
    0: "non_toxic",
    1: "toxic"
}


def predict(text: str) -> dict:

    # preprocess
    clean_text = preprocess_text(text)

    # tfidf transform
    x = tfidf.transform([clean_text])

    # predict
    prediction = int(model.predict(x)[0])

    result = {
        "prediction": LABEL_MAP[prediction],
        "clean_text": clean_text
    }

    # probability
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(x)[0]

        result["probability"] = {
            "non_toxic": float(probability[0]),
            "toxic": float(probability[1])
        }

    return result