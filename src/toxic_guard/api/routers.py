from typing import Annotated

from fastapi import APIRouter, Form

from toxic_guard.schemas.prediction import PredictionResponse
from toxic_guard.services.predictor import predict


router = APIRouter()


@router.post(
    "/predict",
    tags=["Prediction"],
    response_model=PredictionResponse,
    summary="Predict toxic comment",
    description="Detect whether a Vietnamese comment is toxic or non-toxic"
)
def predict_api(
    text: Annotated[
        str,
        Form(
            ...,
            description="Enter text to classify",
            examples=["Xin chào"]
        )
    ]
):

    result = predict(text)

    return result