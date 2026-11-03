import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
pd.set_option("display.width", 250)
torch.manual_seed(42)
np.random.seed(42)

MODEL = "XGBoost"
SEQ = 24           # number of hours the LSTM looks back

df = pd.read_csv("bangladesh_power_generation_2019_2026_processed.csv")      # Load, modify the data & preprocessing.
hour = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour
df["timestamp"] = pd.to_datetime(df["date"], format="%m/%d/%Y") + pd.to_timedelta(hour, unit="h")
targets = ["generation_mw", "gas_mw", "liquid_fuel_mw", "coal_mw", "hydro_mw", "solar_mw",
           "wind_mw", "india_bheramara_hvdc_mw", "india_tripura_mw", "india_adani_mw", "nepal_import_mw"]
names = ["Generation (MW)", "Gas", "Liquid fuel", "Coal", "Hydro", "Solar", "Wind",
         "Bheramara", "Tripura", "Adani", "Nepal"]
data = df.groupby("timestamp")[targets].mean()
for col in targets:
    data.loc[data[col] > 2 * data[col].quantile(0.999), col] = np.nan
data.loc[data["generation_mw"] < 2000, "generation_mw"] = np.nan
data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq="h"))   # Fill missing values with Median Method
data = data.fillna(data.groupby([data.index.year, data.index.month, data.index.hour]).transform("median"))
data = data.fillna(data.median())

first_time, last_time = data.index.min(), data.index.max()
fixed_holidays = [(2, 21), (3, 17), (3, 26), (4, 14), (5, 1), (8, 5), (8, 15), (12, 16), (12, 25)]
def calendar(times):
    holiday = times.dayofweek.isin([4, 5]) | np.array([(t.month, t.day) in fixed_holidays for t in times])
    return pd.DataFrame({
        "hour": times.hour,
        "weekday": times.dayofweek,
        "month": times.month,
        "day_of_year": times.dayofyear,
        "day_no": (times - first_time).days,
        "holiday": holiday.astype(int),
    }, index=times)

def fit_baseline(train_end):                # Models Training and Prediction. Baseline Model.
    train = data.loc[:train_end]
    cal = calendar(train.index)
    means = train.groupby([cal["month"].values, cal["holiday"].values, cal["hour"].values]).mean()

    def predict(times):
        c = calendar(times)
        keys = pd.MultiIndex.from_arrays([c["month"].values, c["holiday"].values, c["hour"].values])
        return means.reindex(keys).fillna(train.mean()).values
    return predict

def fit_xgboost(train_end):              #XGBoost Model.
    train = data.loc[:train_end]
    mean, std = train.mean(), train.std().replace(0, 1)
    model = MultiOutputRegressor(XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                              subsample=0.8, colsample_bytree=0.8, random_state=42))
    model.fit(calendar(train.index), (train - mean) / std)

    def predict(times):
        return model.predict(calendar(times)) * std.values + mean.values
    return predict

def seq_features(times):                # LSTM Model.
    cal = calendar(times)
    hour_angle = 2 * np.pi * cal["hour"].values / 24
    year_angle = 2 * np.pi * cal["day_of_year"].values / 365.25
    week_angle = 2 * np.pi * cal["weekday"].values / 7
    return np.column_stack([
        np.sin(hour_angle), np.cos(hour_angle),
        np.sin(year_angle), np.cos(year_angle),
        np.sin(week_angle), np.cos(week_angle),
        cal["holiday"].values, cal["day_no"].values / 2500,
    ]).astype("float32")
def windows(times):
    rng = pd.date_range(times.min() - pd.Timedelta(hours=SEQ - 1), times.max(), freq="h")
    all_windows = sliding_window_view(seq_features(rng), SEQ, axis=0)
    pos = np.asarray((times - rng[0]) / pd.Timedelta(hours=1)).astype(int)
    return np.ascontiguousarray(all_windows[pos - (SEQ - 1)].transpose(0, 2, 1))
class Net(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.lstm = nn.LSTM(n_in, 48, batch_first=True)
        self.out = nn.Linear(48, n_out)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.out(o[:, -1])
def fit_lstm(train_end):
    train = data.loc[:train_end]
    mean, std = train.mean(), train.std().replace(0, 1)
    x = torch.tensor(windows(train.index))
    y = torch.tensor(((train - mean) / std).values, dtype=torch.float32)
    val = torch.tensor(np.asarray(train.index > train_end - pd.Timedelta(days=90)))
    x_train, y_train, x_val, y_val = x[~val], y[~val], x[val], y[val]

    net = Net(x.shape[2], len(targets))
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    best, best_state = float("inf"), None
    for epoch in range(20):  # the last 90 days of training data pick the best epoch
        net.train()
        order = torch.randperm(len(x_train))
        for b in range(0, len(order), 256):
            idx = order[b:b + 256]
            opt.zero_grad()
            nn.functional.mse_loss(net(x_train[idx]), y_train[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            loss = nn.functional.mse_loss(net(x_val), y_val).item()
        if loss < best:
            best, best_state = loss, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    def predict(times):
        with torch.no_grad():
            out = net(torch.tensor(windows(times))).numpy()
        return out * std.values + mean.values
    return predict

models = {"Baseline": fit_baseline, "XGBoost": fit_xgboost, "LSTM": fit_lstm}      #Model can be picked by its name

#================================ Will Be Update Soon ================================