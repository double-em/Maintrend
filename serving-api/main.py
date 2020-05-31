import datetime
import logging
import requests
import json
import os
import numpy as np
from util.data_puller import apicallv3 as apicall
from json import JSONEncoder
from fastapi import FastAPI
from pydantic import BaseModel

serving_logger = logging.getLogger('serving-api')
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(name)s: %(message)s")
serving_logger.setLevel(logging.DEBUG)

app = FastAPI(title="Serving API", description="Serving version of the Predictor API.", version="1.2.1")

history_size = 60

class PredictionResult(BaseModel):
    date: datetime.date = None
    prediction_available: bool

class PredictionRequest(BaseModel):
    api_key: str
    channel_id: int
    prediction_date: datetime.date

class APIStatus(BaseModel):
    request_received: datetime.datetime = datetime.datetime.now()
    messsage: str = "You've hit the API! Autch..."
    model_version: int = None
    model_state: str
    model_error_code: str
    model_error_message: str



@app.get("/", response_model=APIStatus)
async def root():
    try:
        json_response = requests.get("http://predictor-service:8501/v1/models/predictor")
        mvs = json.loads(json_response.text)['model_version_status'][0]
        apistatus = APIStatus(
            model_version=mvs['version'],
            model_state=mvs['state'],
            model_error_code=mvs['status']['error_code'],
            model_error_message=mvs['status']['error_message']
        )
    except Exception as ex:
        serving_logger.error(ex)
        apistatus = APIStatus(
            model_state="UNAVAILABLE",
            model_error_code="404",
            model_error_message="Not found. Check log for details."
        )

    return apistatus

@app.post("/predict", response_model=PredictionResult)
async def predict(prediction_request : PredictionRequest):

    serving_logger.debug(f"Got API call for Predictor!")

    apikey = prediction_request.api_key
    channel_id = prediction_request.channel_id
    datetime_to = datetime.datetime.strptime(f"{prediction_request.prediction_date} 23:59:59", "%Y-%m-%d %H:%M:%S")

    # NOTE: Take the day before to ensure only completed days
    datetime_to = datetime_to - datetime.timedelta(1)

    # NOTE: Add 1 second to get the start of the next day so it matches history_size.
    datetime_from = (datetime_to - datetime.timedelta(history_size)) + datetime.timedelta(seconds=1)

    req_url = os.environ['API_BASE_URL'] + '/' + str(channel_id) + '/' + os.environ['API_F']

    try:
        train = apicall(req_url, apikey, datetime_to.strftime("%Y-%m-%d %H:%M:%S"), datetime_from.strftime("%Y-%m-%d %H:%M:%S"), predictor_call=True)
    except Exception as ex:
        serving_logger.error(ex)
        return PredictionResult(prediction_available=False)

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    tcx = [[]]
    tcx[0] = train.values

    data = json.dumps({"signature_name":"serving_default", "instances":tcx}, cls=NumpyArrayEncoder)
    headers = {"content-type":"application/json"}

    try:
        json_response = requests.post("http://predictor-service:8501/v1/models/predictor:predict", data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']
    except Exception as ex:
        serving_logger.error(ex)
        return PredictionResult(prediction_available=False)


    single_prediction = round(float(predictions[0][0]))

    prediction_date = datetime_to + datetime.timedelta(single_prediction)
    prediction_date = prediction_date.date()

    return PredictionResult(date=prediction_date, prediction_available=True)