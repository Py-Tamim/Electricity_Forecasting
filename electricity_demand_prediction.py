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
def compare():                                # Compare different modles on the test period.
    split = pd.Timestamp("2025-12-31 23:00")  # train up to this hour, test on the hours after it
    test = data.loc[split + pd.Timedelta(hours=1):]
    cal = calendar(test.index)
    groups = {
        "All hours": np.ones(len(test), bool),
        "Working days": cal["holiday"].values == 0,
        "Weekends/holidays": cal["holiday"].values == 1,
        "Night (12 am-5 am)": test.index.hour <= 5,
        "Evening peak (6 pm-10 pm)": (test.index.hour >= 18) & (test.index.hour <= 22),
    }
    actual = test["generation_mw"].values

    summary, by_group, by_column = {}, {}, {}
    for name, fit in models.items():
        start = time.time()
        predict = fit(split)
        seconds = time.time() - start
        err = test.values - np.clip(predict(test.index), 0, None)
        gen_err = err[:, 0]
        summary[name] = {"MAE (MW)": np.abs(gen_err).mean(), "RMSE (MW)": np.sqrt((gen_err ** 2).mean()),
                         "MAPE (%)": (np.abs(gen_err) / actual).mean() * 100, "Train time (s)": seconds}
        by_group[name] = {g: np.abs(gen_err[m]).mean() for g, m in groups.items()}
        by_column[name] = dict(zip(names, np.abs(err).mean(axis=0)))

    summary = pd.DataFrame(summary).T
    summary["MAE gain vs baseline (%)"] = (1 - summary["MAE (MW)"] / summary.loc["Baseline", "MAE (MW)"]) * 100
    print("\nTest period:", test.index[0], "to", test.index[-1])
    print("\nTotal generation")
    print(summary.round(1))
    print("\nTotal generation, MAE (MW) by type of hour")
    print(pd.DataFrame(by_group).round(1))
    print("\nMAE (MW) of every column")
    print(pd.DataFrame(by_column).round(1))
    gain = (1 - summary.loc["LSTM", "MAE (MW)"] / summary.loc["XGBoost", "MAE (MW)"]) * 100
    print(f"\nLSTM vs XGBoost: {gain:+.1f}% MAE, training time {summary.loc['LSTM', 'Train time (s)']:.0f}s "
          f"vs {summary.loc['XGBoost', 'Train time (s)']:.0f}s")

def read_dates(text):                                       # User Inpute Method
    pattern = r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"
    found = re.findall(pattern, text.lower())
    left = re.sub(r"from|to|[\s,-]", "", re.sub(pattern, "", text.lower()))
    if len(found) not in (1, 2) or left:
        raise ValueError
    dates = [pd.Timestamp(year=int(y), month=int(m), day=int(d)) for d, m, y in found]
    if len(dates) == 1:
        dates = dates * 2
    if dates[0] < first_time.normalize() or dates[1] < dates[0] or (dates[1] - dates[0]).days > 366:
        raise ValueError
    return dates

def read_hours(text):
    text = text.lower().strip()
    if not text:
        return list(range(24))
    pattern = r"\b(\d{1,2})(?::00)?\s*(am|pm)?\b"
    found = re.findall(pattern, text)
    left = re.sub(r"from|to|[\s-]", "", re.sub(pattern, "", text))
    if len(found) not in (1, 2) or left:
        raise ValueError
    hours = []
    for h, am_pm in found:
        h = int(h)
        if am_pm:
            if not 1 <= h <= 12:
                raise ValueError
            h = h % 12 + (12 if am_pm == "pm" else 0)
        elif h > 23:
            raise ValueError
        hours.append(h)
    if len(hours) == 2 and hours[1] < hours[0]:
        raise ValueError
    return list(range(hours[0], hours[-1] + 1))

def search(start, end, hours):
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in pd.date_range(start, end) for h in hours])
    known = times.isin(data.index)
    values = np.zeros((len(times), len(targets)))
    values[known] = data.loc[times[known]].values
    if (~known).any():
        print("\nPredicting the hours after the data ends, please wait...")
        predict = models[MODEL](last_time)
        values[~known] = np.clip(predict(times[~known]), 0, None)

    result = pd.DataFrame(values.round().astype(int), columns=names)
    result.insert(0, "Time", times.strftime("%I %p").str.lstrip("0"))
    result.insert(0, "Day", times.day_name())
    result.insert(0, "Date", times.strftime("%d-%m-%Y"))
    return result

def plot_results(result):                       # Plot graphs using Matplotlib library.
    """Show the final result and, for more than 3 rows, compare with prior-year actual data."""
    result = result.copy()
    result["DateTime"] = pd.to_datetime(
        result["Date"] + " " + result["Time"],
        format="%d-%m-%Y %I %p"
    )
    result = result.sort_values("DateTime")
    page_blue = "#061A40"
    axes_blue = "#082654"
    text_color = "#FFFFFF"
    grid_color = "#E0DD08"
    result_line = "#FFFFFF"
    previous_line = "#FFC857"

    def style_axis(ax, fig, title):
        fig.patch.set_facecolor(page_blue)
        ax.set_facecolor(axes_blue)
        ax.set_title(title, color=text_color, fontsize=14, weight="bold", pad=14)
        ax.set_xlabel("Date and time", color=text_color)
        ax.set_ylabel("Generation (MW)", color=text_color)
        ax.tick_params(axis="x", colors=text_color, rotation=45)
        ax.tick_params(axis="y", colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(True, color=grid_color, alpha=0.35, linestyle="--")
        legend = ax.legend(facecolor=axes_blue, edgecolor=grid_color)
        for label in legend.get_texts():
            label.set_color(text_color)
    
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        result["DateTime"], result["Generation (MW)"],
        color=result_line, marker="o", markersize=4, linewidth=2,
        label="Final result — generation (MW)"
    )
    style_axis(ax, fig, "Bangladesh Electricity Generation")
    fig.tight_layout()
    plt.show()

    if len(result) > 3:                                   # compare with actual data
        previous_times = result["DateTime"] - pd.DateOffset(years=1)
        previous_values = pd.Series(np.nan, index=result.index, dtype="float64")

        matched = previous_times.isin(data.index)
        if matched.any():
            previous_values.loc[matched] = data.loc[
                pd.DatetimeIndex(previous_times[matched]), "generation_mw"
            ].to_numpy()

            fig, ax = plt.subplots(figsize=(13, 6))
            ax.plot(
                result["DateTime"], result["Generation (MW)"],
                color=result_line, marker="o", markersize=4, linewidth=2,
                label="Current / predicted result"
            )
            ax.plot(
                result["DateTime"], previous_values,
                color=previous_line, marker="s", markersize=4, linewidth=1.8,
                linestyle="--", label="Previous year's actual result"
            )
            style_axis(ax, fig, "Current Result vs Previous Year's Actual Generation")
            fig.tight_layout()
            plt.show()
        else:
            print("Previous-year comparison skipped: no matching historical timestamps were found.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    else:
        try:
            start, end = read_dates(input("Enter a date or a timeline (e.g. 18/12/2026 or 18/11/2026 to 17/01/2027): "))
            hours = read_hours(input("Enter the time (e.g. 10 am or 10 am to 1 pm, press Enter for all 24 hours): "))
        except ValueError:
            print("Invalide input, please try again after some time.")
        else:
            print()
            result = search(start, end, hours)
            print(result.to_string(index=False))
            plot_results(result)
#================================ Will Be Update Soon ================================