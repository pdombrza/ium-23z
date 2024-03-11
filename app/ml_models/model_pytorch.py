import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import concurrent.futures
import torch.optim as optim
import torch.utils.data as data

DIRECTORY = "../data_v2/"

MODELS_PATH = "../pretrained/pytorch/"


class TrackModelLSTM(nn.Module):
    def __init__(self, track_id):
        super().__init__()
        self.track_id = track_id
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        self.optimizer = optim.Adam(self.parameters())
        self.loss_fn = nn.MSELoss()

    def set_loader(self, X_train, y_train):
        self.loader = data.DataLoader(data.TensorDataset(X_train[self.track_id], y_train[self.track_id]), shuffle=True, batch_size=8)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


def load_data():
    session_data = pd.read_json("test_data/" + "sessions_reduced.jsonl", lines=True)
    tracks_data = pd.read_json(DIRECTORY + "tracks.jsonl", lines=True)
    return session_data, tracks_data


def prepare_data(session_data):
    train_size = int(len(session_data)*0.67)
    test_size = len(session_data) - train_size

    train, test = session_data[:train_size], session_data[train_size:]
    return train, test


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


def prepare_data_paralell(train, test):
    lookback = 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_train = executor.submit(create_dataset, train, lookback)
        future_test = executor.submit(create_dataset, test, lookback)

    X_train, y_train = future_train.result()
    X_test, y_test = future_test.result()

    return X_train, y_train, X_test, y_test


def create_models(tracks_data):
    models = {track_id: TrackModelLSTM(track_id) for track_id in tracks_data['id']}
    return models


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    n_epochs = 200

    predictions = {}

    for track_id, model in models.items():
        for epoch in range(n_epochs):
            model.train()
            model.set_loader(X_train, y_train)
            for X_batch, y_batch in model.loader:
                y_pred = model(X_batch)
                predictions[track_id] = y_pred
                loss = model.loss_fn(y_pred, y_batch)
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
            # Validation
            if epoch % 100 != 0:
                continue
            model.eval()
            with torch.no_grad():
                y_pred = model(X_train[track_id])
                train_rmse = np.sqrt(model.loss_fn(y_pred, y_train[track_id]))
                y_pred = model(X_test[track_id])
                test_rmse = np.sqrt(model.loss_fn(y_pred, y_test[track_id]))
            print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    print(predictions)


def main():
    pass

if __name__ == "__main__":
    main()