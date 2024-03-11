from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Path
from models import ModelType, get_cache_assignments, get_cache_assignments_lstm, get_plays, get_plays_lstm, run_experiment

app = FastAPI()


@app.get("/api/status")
async def get_status():
    """
    Simple function that allows one to check if the server is running

    :return: ok status if server is running
    """
    return {"status": "OK"}


@app.get("/api/predict/assign/{model_type}")
async def predict(model_type: Annotated[ModelType, Path(title="Model type, available - xgboost, lstm")], lookback_days: Annotated[int, Query(title="amount of days to look back for LSTM", gt=0, lt=365)] = 7):
    """
    Function that generates suggested cache levels
    for the next 24 hours for all tracks

    :param model_type: enum - type of model - xgboost or lstm
    :param lookback_days: int - only used when model_type == lstm - amount of days
    to look back when predicting amount of plays using LSTM
    :return: suggested cache levels for all tracks
    """
    if model_type == ModelType.xgboost:
        predictions = get_cache_assignments(model_type=model_type)
    elif model_type == ModelType.lstm:
        predictions = get_cache_assignments_lstm(lookback_days=lookback_days)

    return predictions


@app.get("/api/predict/plays/{model_type}")
async def predict_plays(model_type: Annotated[ModelType, Path(title="Model type, available - xgboost, lstm")], lookback_days: Annotated[int, Query(title="amount of days to look back for LSTM", gt=0, lt=365)] = 7):
    """
    Function that generates predictions of play amount
    for the next 24 hours for all tracks

    :param model_type: enum - type of model - xgboost or lstm
    :param lookback_days: int - only used when model_type == lstm - amount of days
    to look back when predicting amount of plays using LSTM
    :return: predicted plays for all tracks
    """
    if model_type == ModelType.xgboost:
        predictions = get_plays(model_type=model_type)
    elif model_type == ModelType.lstm:
        predictions = get_plays_lstm(lookback_days=lookback_days)

    return predictions


@app.post("/api/predict/ab")
async def ab_experiment(lookback_days: Annotated[int, Query(title="amount of days to look back for LSTM", gt=0, lt=365)] = 7):
    """
    Function that generates predictions of play amount
    for the next 24 hours for all tracks with both models
    Saves result to file for later calculations

    :return: None
    """
    return run_experiment(lookback_days=lookback_days)
