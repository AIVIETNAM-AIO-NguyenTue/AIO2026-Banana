from pydantic import BaseModel


class ProbabilityResponse(BaseModel):
    non_toxic: float
    toxic: float


class PredictionResponse(BaseModel):
    prediction: str
    clean_text: str
    probability: ProbabilityResponse | None = None