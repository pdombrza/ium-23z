import pickle
import json
import numpy as np
import datetime
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
PRETRAINED_DIRECTORY = "pretrained/xgb/"
MODEL_SAVE_PATH = PRETRAINED_DIRECTORY + "models_xgb.pkl"
DATA_DIRECTORY = "data_v2/"
TEST_DATA_DIRECTORY = "../test_data/"

PERCENT_SLOW = 84.972 / 100
PERCENT_MEDIUM = 14.93 / 100
PERCENT_FAST =  0.09816 / 100


def load_data():
    session_data = pd.read_json(TEST_DATA_DIRECTORY + "sessions_reduced.jsonl", lines=True)
    tracks_data = pd.read_json(DATA_DIRECTORY + "tracks.jsonl", lines=True)
    return session_data, tracks_data


def prepare_data(session_data, split_date):
    # split_date = '8-Nov-2023'
    train_data = session_data.loc[session_data.timestamp <= split_date].copy()
    test_data = session_data.loc[session_data.timestamp > split_date].copy()

    return train_data, test_data


def prepare_data_prod(session_data, split_date_prod):
    prod_data = session_data.loc[session_data.timestamp > split_date_prod].copy()
    return prod_data


def create_features(df, tracks_data):

    play_events = df[df['event_type'] == 'play']

    beg = df['timestamp'].min().date()
    end = df['timestamp'].max().date()

    date_range = pd.date_range(start=beg, end=end, freq='D')
    track_ids = tracks_data["id"]
    track_count_df = pd.DataFrame(index=date_range, columns=track_ids).fillna(0)


    for track_id in track_ids:
        track_events = play_events[play_events['track_id'] == track_id]
        daily_counts = track_events.groupby(track_events['timestamp'].dt.date).size()
        track_count_df[track_id] = track_count_df.index.map(daily_counts).fillna(0)


    dfs_dict = {col: group_df.reset_index() for col, group_df in track_count_df.items()}

    X = {}
    y = {}

    for track_id, df in dfs_dict.items():
        df['date'] = pd.to_datetime(df['index'])
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofmonth'] = df['date'].dt.day
        X[track_id] = df[['dayofweek', 'dayofmonth']]
        y[track_id] = df[track_id]

    return X, y

# Train models
def create_models(tracks_data):
  return {track_id:xgb.XGBRegressor(n_estimators=100) for track_id in tracks_data['id']}


def train_models(models, train_X, train_y, test_X, test_y):
  for track_id in train_X:
      models[track_id].fit(train_X[track_id], train_y[track_id],
              eval_set=[(train_X[track_id], train_y[track_id]), (test_X[track_id], test_y[track_id])],
              early_stopping_rounds=50,
          verbose=False)


def train_prod_models(models, dataset_X, dataset_y):
  for track_id in dataset_X:
      models[track_id].fit(dataset_X[track_id], dataset_y[track_id], verbose=False)


def save_trained_models(models, path):
  with open(path, 'wb') as fh:
    pickle.dump(models, fh)


def load_saved_models(path):
  with open(path, 'rb') as fh:
    models = pickle.load(fh)
  return models


# Calculate errors
def calc_errors(y_pred, y_true):
    mean_squared_errors = {}
    mean_absolute_errors = {}

    for track_id in y_pred:
        mean_squared_errors[track_id] = mean_squared_error(y_true=y_true[track_id], y_pred=y_pred[track_id])
        mean_absolute_errors[track_id] = mean_absolute_error(y_true=y_true[track_id], y_pred=y_pred[track_id])

    return mean_squared_errors, mean_absolute_errors


# Predict
def predict(models, X_test):
    predictions = {track_id: models[track_id].predict(X_test[track_id]) for track_id in X_test}
    return predictions


def get_cache_assignments(predictions, tracks_data):
    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1])


    all_tracks = len(sorted_predictions)

    slow = sorted_predictions[:int(PERCENT_SLOW*all_tracks)]
    medium = sorted_predictions[int(PERCENT_SLOW*all_tracks):int((PERCENT_MEDIUM + PERCENT_SLOW)*all_tracks)]
    fast = sorted_predictions[int((PERCENT_SLOW + PERCENT_MEDIUM)*all_tracks):]

    caches = {"slow": 0.005 * 1e-5, "medium": 0.01*1e-5, "fast": 0.05*1e-5}
    tracks_caches = {}

    for index, (track_id, value) in enumerate(slow):
        if value < 0.0:
            slow[index] = 0
        tracks_caches[track_id] = "slow"
    for track_id, _ in medium:
        tracks_caches[track_id] = "medium"
    for track_id, _ in fast:
        tracks_caches[track_id] = "fast"

    costs = {}

    for track in tracks_caches:
        costs[track] = tracks_data.loc[tracks_data['id'] == track]['duration_ms'] * caches[tracks_caches[track]]

    return tracks_caches, costs


def predict_single(model):
   predictions = model.predict()
   return predictions


def prep_X_dict(tracks_data):
    X_dict = {}
    track_ids = tracks_data["id"]
    for track_id in track_ids:
        tomorrow_date = datetime.date.today() + datetime.timedelta(days=1)
        df = pd.DataFrame({'date': [pd.to_datetime(tomorrow_date)]})
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofmonth'] = df['date'].dt.day
        X_dict[track_id] = df[['dayofweek', 'dayofmonth']]

    return X_dict


def get_next_day_cache_assignments(models):
    tracks_data = pd.read_json(DATA_DIRECTORY + "tracks.jsonl", lines=True)
    X_dict = prep_X_dict(tracks_data)
    predictions = predict(models, X_dict)
    cache_assignments, costs = get_cache_assignments(predictions, tracks_data)
    return cache_assignments, costs


def get_play_predictions(models):
    tracks_data = pd.read_json(DATA_DIRECTORY + "tracks.jsonl", lines=True)
    X_dict = prep_X_dict(tracks_data)
    predictions = predict(models, X_dict)
    predictions_converted = {pred: str(max(predictions[pred][0], 0)) for pred in predictions}
    return predictions_converted



def main():
    split_date_prod = '1-Feb-2023'
    split_date_test = '8-Dec-2022'
    session_data, tracks_data = load_data()
    print("Finished loading datasets")

    prod_data = prepare_data_prod(session_data, split_date_test)
    models = load_saved_models(MODEL_SAVE_PATH)
    cache_assignments = get_next_day_cache_assignments(models)
    print(cache_assignments)
    X, y = create_features(prod_data, tracks_data)
    # X_train, y_train = create_features(train_data)
    # X_test, y_test = create_features(test_data)
    print("Finished preparing data")

    models = create_models(tracks_data)
    print("Finished creating models")

    train_prod_models(models, X, y)
    print("Finished fitting models")

    save_trained_models(models, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()