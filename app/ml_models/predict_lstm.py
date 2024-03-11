import pickle
import json
import datetime
import pandas as pd
import torch
import torch.nn as nn
import concurrent.futures
import torch.optim as optim
import torch.utils.data as data

PERCENT_SLOW = 84.972 / 100
PERCENT_MEDIUM = 14.93 / 100
PERCENT_FAST =  0.09816 / 100

DIRECTORY = "data_v2/"
TEST_DIRECTORY = "test_data/"


class TrackModel(nn.Module):
    def __init__(self, track_id):
        super().__init__()
        self.track_id = track_id
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        self.optimizer = optim.Adam(self.parameters())
        self.loss_fn = nn.MSELoss()

    def set_loader(self, X_data, y_data):
        self.loader = data.DataLoader(data.TensorDataset(X_data[self.track_id], y_data[self.track_id]), shuffle=True, batch_size=10)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'TrackModel':
            return TrackModel
        return super().find_class(module, name)


def load_data():
    session_data = pd.read_json(TEST_DIRECTORY + "sessions_reduced.jsonl", lines=True)
    tracks_data = pd.read_json(DIRECTORY + "tracks.jsonl", lines=True)
    return session_data, tracks_data


def create_dataset(dataset, lookback, tracks_data):
    play_events = dataset[dataset['event_type'] == 'play']

    beg = dataset['timestamp'].min().date()
    end = dataset['timestamp'].max().date()

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
        timeseries = df[[track_id]].values.astype('float32')
        for i in range(len(timeseries) - lookback):
            X[track_id] = torch.tensor(timeseries[i:i+lookback])
            y[track_id] = torch.tensor(timeseries[i+1:i+lookback+1])
    return X, y


def prepare_data_parallel(session_data, lookback, tracks_data):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_full = executor.submit(create_dataset, session_data, lookback, tracks_data)

    X_full, y_full = future_full.result()
    return X_full, y_full


def generate_predictions(tracks_data, X_full):
    predictions = {}

    for track_id in tracks_data['id']:
        # Load the model
        with open(f"pretrained/pytorch/{track_id}", "rb") as file:
            model = CustomUnpickler(file).load()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_full[track_id])
            pred = max(float(y_pred[0][0]), 0)
            predictions[track_id] = pred

    return predictions


def load_to_pred(days: int):
    tracks_data = pd.read_json(DIRECTORY + "tracks.jsonl", lines=True)
    if days > 364:
        raise ValueError("days must be less than 365")
    session_data = pd.read_json(DIRECTORY + "sessions.jsonl", lines=True)
    split_date = session_data['timestamp'].max().date() - datetime.timedelta(days=days)
    X_test = session_data.loc[session_data.timestamp >= str(split_date)].copy()
    return X_test, tracks_data



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


def get_play_predictions(lookback_days):
    X_test, tracks_data = load_to_pred(lookback_days)
    X_test, _ = prepare_data_parallel(X_test, lookback_days, tracks_data)
    predictions = generate_predictions(tracks_data, X_test)
    predictions_str = {pred: str(predictions[pred]) for pred in predictions}
    return predictions_str


def get_next_day_cache_assignments(lookback_days):
    X_test, tracks_data = load_to_pred(lookback_days)
    X_test, _ = prepare_data_parallel(X_test, lookback_days, tracks_data)
    predictions = generate_predictions(tracks_data, X_test)
    assignments, costs = get_cache_assignments(predictions, tracks_data)
    return assignments, costs



def main():
    # with open("predictions_pytorch.json", "w") as file:
    #     json.dump(predictions, file)
    predictions = get_play_predictions(7)

    print(predictions)


if __name__ == "__main__":
    main()