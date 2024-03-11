from enum import Enum
from abc import abstractmethod, ABC
from typing import Optional
import pickle
import os
import json
from xgboost import XGBRegressor
from ml_models.predict_lstm import TrackModel, get_next_day_cache_assignments as lstm_get_assignments
from ml_models.predict_lstm import get_play_predictions as lstm_get_play_predictions
from ml_models.model_xgb import get_play_predictions, get_next_day_cache_assignments as xgb_get_assignments

PATH_XGB = "pretrained/xgb/models_xgb.pkl"
PATH_LSTM = "pretrained/pytorch"
PATH_PREDICTIONS = "../results/results_ab.json"


class ModelType(Enum):
    xgboost = "xgboost"
    lstm = "lstm"


class StorageService(ABC):
    @abstractmethod
    def load_models():
        ...

    @abstractmethod
    def get_single_model():
        ...

    @abstractmethod
    def save_predictions():
        ...


class FileSystemStorage(StorageService):
    def __init__(self):
        self.models_dict = {}

    def load_models(self, model_type: ModelType) -> dict[str, XGBRegressor | TrackModel]:
        if model_type == ModelType.xgboost:
            with open(PATH_XGB, 'rb') as fh:
                self.models_dict = pickle.load(fh)
        elif model_type == ModelType.lstm:
            for fname in os.listdir(PATH_LSTM):
                with open(os.path.join(PATH_LSTM, fname), 'rb') as fh:
                    self.models_dict[fname] = pickle.load(fh)
        return self.models_dict

    def get_single_model(self, track_id: str) -> XGBRegressor | TrackModel:
        return self.models_dict[track_id]

    def save_predictions(self, path, data):
        with open(path, 'w') as fh:
            json.dump(data, fh, indent=4)


def load_models(storage: StorageService, model_type: ModelType) -> dict[str, XGBRegressor | TrackModel]:
    return storage.load_models(model_type)


def get_model_by_track_id(track_id: str, models: dict[str, Optional[TrackModel | XGBRegressor]]) -> Optional[TrackModel | XGBRegressor]: # add pytorch model instead of None in argument and add to return types
    return models[track_id]


def get_cache_assignments(model_type: ModelType) -> dict[str, str]:
    storage = FileSystemStorage()
    models = storage.load_models(model_type)
    cache_assignments, _ = xgb_get_assignments(models)
    return cache_assignments


def get_plays(model_type: ModelType) -> dict[str, str]:
    storage = FileSystemStorage()
    models = storage.load_models(model_type)
    predicted_days = get_play_predictions(models)
    return predicted_days


def get_cache_assignments_lstm(lookback_days: int) -> dict[str, str]:
    cache_assignments, _ = lstm_get_assignments(lookback_days=lookback_days)
    return cache_assignments


def get_plays_lstm(lookback_days: int) -> dict[str, str]:
    days_predictions = lstm_get_play_predictions(lookback_days)
    return days_predictions


def run_experiment(lookback_days: int) -> dict[str, str]:
    storage = FileSystemStorage()
    models = storage.load_models(model_type=ModelType.xgboost)
    predictions_xgb, costs_xgb = xgb_get_assignments(models)
    predictions_lstm, costs_lstm = lstm_get_assignments(lookback_days=lookback_days)
    results_xgb = {"XGB results - track_id: pred plays, cost": {track_id: (predictions_xgb[track_id], costs_xgb[track_id]) for track_id in predictions_xgb}}
    results_lstm = {"LSTM results - track_id: pred plays, cost": {track_id: (predictions_lstm[track_id], costs_lstm[track_id]) for track_id in predictions_lstm}}
    storage.save_predictions(PATH_PREDICTIONS, results_xgb)
    storage.save_predictions(PATH_PREDICTIONS, results_lstm)
    return {"status" : "success"}