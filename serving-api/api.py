import datetime
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Pedictor API", description="Serving version of the Predictor API.", version="1.2.1")
class PredictionResult(BaseModel):
    date: datetime.date = None
    prediction_available: bool

class PredictionRequest(BaseModel):
    api_key: str
    channel_id: int
    prediction_date: datetime.date

class APIStatus(BaseModel):
    request_received: datetime.datetime = datetime.datetime.now()
    internal_api_status: str = "Down"
    messsage: str = "You've hit the API! Autch..."

@app.get("/", response_model=APIStatus)
async def root():
    return APIStatus()

@app.post("/predict", response_model=PredictionResult)
async def predict(prediction_request : PredictionRequest):
    if True:
        return PredictionResult(prediction_available=False)