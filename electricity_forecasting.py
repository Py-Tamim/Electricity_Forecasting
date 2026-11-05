import re
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBRegressor
pd.set_option("display.width", 250)
torch.manual_seed(42)
np.random.seed(42)
MODEL = "XGBoost"
SEQ = 14           # number of days the LSTM looks back
 
df = pd.read_csv("Bangladesh_Electricity_Dataset_2019_2026.csv")      # Load, modify the data & preprocessing.
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
is_weekend_day = df["Day of the week"].isin(["Friday", "Saturday"])
is_no_holiday = df["Holiday name/Weekend"] == "No Holiday"
df.loc[is_weekend_day & is_no_holiday, "Holiday name/Weekend"] = "Weekend"
is_holiday_or_weekend = df["Holiday name/Weekend"] != "No Holiday"
df.loc[is_holiday_or_weekend, "Holiday_cat/weekend"] = 1
df.loc[~is_holiday_or_weekend, "Holiday_cat/weekend"] = 0

temp_col = "Maximum Temperature in Dhaka was"
df.loc[df[temp_col] == 0, temp_col] = np.nan  # 0 degree is not a real reading for Dhaka
num_cols = df.select_dtypes("number").columns
df[num_cols] = df[num_cols].fillna(df.groupby(["Year", "Month"])[num_cols].transform("median"))
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in ["Day of the week", "Holiday name/Weekend"]:
    df[col] = df[col].fillna(df[col].mode()[0])
areas = [c.replace("_demand", "") for c in df.columns if c.endswith("_demand")]
records = df.set_index("Date")
first_day, last_day = df["Date"].min(), df["Date"].max()
all_days = pd.date_range(first_day, last_day)
demand = records[[a + "_demand" for a in areas]].set_axis(areas, axis=1).reindex(all_days)
supply = records[[a + "_supply" for a in areas]].set_axis(areas, axis=1).reindex(all_days)
temp_by_month = df.groupby("Month")[temp_col].median()
temp_mean, temp_std = df[temp_col].mean(), df[temp_col].std()
fixed_holidays = [(2, 21), (3, 17), (3, 26), (4, 14), (5, 1), (8, 5), (8, 15), (12, 16), (12, 25)]
def day_kind(days):
    kind = pd.Series("No Holiday", index=days)
    kind.loc[np.array([(d.month, d.day) in fixed_holidays for d in days])] = "Public holiday"
    kind.loc[days.dayofweek.isin([4, 5])] = "Weekend"
    return kind

def covariates(days, known_until):
    kind = day_kind(days)
    cov = pd.DataFrame({"holiday": (kind != "No Holiday").astype(int).values,
                        "temp": days.month.map(temp_by_month).values}, index=days)
    known = days.isin(records.index) & (days <= known_until)
    cov.loc[known, "holiday"] = records.loc[days[known], "Holiday_cat/weekend"].astype(int).values
    cov.loc[known, "temp"] = records.loc[days[known], temp_col].values
    return cov
 
def fit_baseline(train_end, data):                      #Baseline Model
    days = all_days[all_days <= train_end]
    train = data.loc[days].copy()
    train["month"] = days.month
    train["holiday"] = covariates(days, train_end)["holiday"].values
    means = train.groupby(["month", "holiday"]).mean()
 
    def predict(area_no, days, known_until):
        cov = covariates(days, known_until)
        return means.loc[list(zip(days.month, cov["holiday"])), areas[area_no]].values
    return predict

def table_features(days, area_no, cov):                   #XGBoost Model
    return pd.DataFrame({
        "area": pd.Categorical([area_no] * len(days), categories=range(len(areas))),
        "day_no": (days - first_day).days,
        "month": days.month,
        "day_of_year": days.dayofyear,
        "weekday": days.dayofweek,
        "holiday": cov["holiday"].values,
        "temp": cov["temp"].values,
    }, index=days)

def fit_xgboost(train_end, data):
    days = all_days[all_days <= train_end]
    cov = covariates(days, train_end)
    scale = data.loc[days].mean()
    xs, ys = [], []
    for i, a in enumerate(areas):
        y = data.loc[days, a] / scale[a]
        ok = y.notna().values
        xs.append(table_features(days, i, cov)[ok])
        ys.append(y[ok])
    model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, enable_categorical=True, tree_method="hist",
                         random_state=42)
    model.fit(pd.concat(xs), pd.concat(ys))
 
    def predict(area_no, days, known_until):
        cov = covariates(days, known_until)
        return model.predict(table_features(days, area_no, cov)) * scale[areas[area_no]]
    return predict

def seq_features(days, cov):                               # LSTM Model 
    year_angle = 2 * np.pi * days.dayofyear.values / 365.25
    week_angle = 2 * np.pi * days.dayofweek.values / 7
    return np.column_stack([
        (days - first_day).days.values / 2500,
        np.sin(year_angle), np.cos(year_angle),
        np.sin(week_angle), np.cos(week_angle),
        cov["holiday"].values,
        (cov["temp"].values - temp_mean) / temp_std,
    ]).astype("float32") 
def windows(area_no, targets, known_until):
    # for every target day: the features of that day and the SEQ-1 days before it
    days = pd.date_range(targets.min() - pd.Timedelta(days=SEQ - 1), targets.max())
    m = seq_features(days, covariates(days, known_until))
    pos = (targets - days[0]).days.values
    x = np.stack([m[p - SEQ + 1:p + 1] for p in pos])
    area_flag = np.zeros((len(targets), SEQ, len(areas)), dtype="float32")
    area_flag[:, :, area_no] = 1
    return np.concatenate([x, area_flag], axis=2)
class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm = nn.LSTM(n_in, 48, batch_first=True)
        self.out = nn.Linear(48, 1)
 
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.out(o[:, -1]).squeeze(1)

def fit_lstm(train_end, data):
    days = all_days[all_days <= train_end][SEQ - 1:]
    scale = data.loc[all_days <= train_end].mean()
    xs, ys, is_val = [], [], []
    for i, a in enumerate(areas):
        y = data.loc[days, a] / scale[a]
        ok = y.notna().values
        xs.append(windows(i, days[ok], train_end))
        ys.append(y.values[ok])
        is_val.append(np.asarray(days[ok] > train_end - pd.Timedelta(days=90)))
    x = torch.tensor(np.concatenate(xs))
    y = torch.tensor(np.concatenate(ys).astype("float32"))
    val = torch.tensor(np.concatenate(is_val))
    x_train, y_train, x_val, y_val = x[~val], y[~val], x[val], y[val]
 
    net = Net(x.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    best, best_state = float("inf"), None
    for epoch in range(30):  # the last 90 days of training data pick the best epoch
        net.train()
        order = torch.randperm(len(x_train))
        for b in range(0, len(order), 128):
            idx = order[b:b + 128]
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
    def predict(area_no, days, known_until):
        with torch.no_grad():
            out = net(torch.tensor(windows(area_no, days, known_until))).numpy()
        return out * scale[areas[area_no]]
    return predict
 
models = {"Baseline": fit_baseline, "XGBoost": fit_xgboost, "LSTM": fit_lstm}
def compare():
    split = pd.Timestamp("2025-12-31")                    # Data training and models comparison
    test_days = all_days[all_days > split]
    actual = demand.loc[test_days].stack()
    when = actual.index.get_level_values(0)
    holiday = records["Holiday_cat/weekend"].reindex(when).values == 1
    hot = when.month.isin([4, 5, 6, 7, 8])
    groups = {"All days": np.ones(len(actual), bool), "Working days": ~holiday,
              "Weekends/holidays": holiday, "Hot months (Apr-Aug)": hot,
              "Cool months (Jan-Mar)": ~hot}
 
    summary, by_group = {}, {}
    for name, fit in models.items():
        start = time.time()
        predict = fit(split, demand)
        seconds = time.time() - start
        pred = pd.DataFrame({a: predict(i, test_days, split) for i, a in enumerate(areas)},
                            index=test_days).stack().loc[actual.index]
        err = actual - pred
        summary[name] = {"MAE (MW)": err.abs().mean(), "RMSE (MW)": np.sqrt((err ** 2).mean()),
                         "MAPE (%)": (err.abs() / actual).mean() * 100, "Train time (s)": seconds}
        by_group[name] = {g: err[m].abs().mean() for g, m in groups.items()}
 
    summary = pd.DataFrame(summary).T
    summary["MAE gain vs baseline (%)"] = (1 - summary["MAE (MW)"] / summary.loc["Baseline", "MAE (MW)"]) * 100
    print("\nTest period:", test_days[0].date(), "to", test_days[-1].date())
    print(summary.round(1))
    print("\nMAE (MW) by type of day")
    print(pd.DataFrame(by_group).round(1))
    gain = (1 - summary.loc["LSTM", "MAE (MW)"] / summary.loc["XGBoost", "MAE (MW)"]) * 100
    print(f"\nLSTM vs XGBoost: {gain:+.1f}% MAE, training time {summary.loc['LSTM', 'Train time (s)']:.0f}s "
          f"vs {summary.loc['XGBoost', 'Train time (s)']:.0f}s")

def read_dates(text):                                # For Inpute 
    pattern = r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"
    found = re.findall(pattern, text.lower())
    left = re.sub(r"from|to|[\s,-]", "", re.sub(pattern, "", text.lower()))
    if len(found) not in (1, 2) or left:
        raise ValueError
    dates = [pd.Timestamp(year=int(y), month=int(m), day=int(d)) for d, m, y in found]
    if len(dates) == 1:
        dates = dates * 2
    if dates[0] < first_day or dates[1] < dates[0] or (dates[1] - dates[0]).days > 366:
        raise ValueError
    return dates
 
def search(predict_demand, predict_supply, start, end, area_list):
    days = pd.date_range(start, end)
    known = days.isin(records.index)
    day_type = day_kind(days).values
 
    result = []
    for area in area_list:
        out = pd.DataFrame({
            "Date": days, "Day": days.day_name(), "Area": area, "Holiday": day_type,
            "Demand (MW)": np.nan, "Supply (MW)": np.nan, "Load shedding (MW)": np.nan,
        })
        if (~known).any():
            i, new_days = areas.index(area), days[~known]
            new_demand = predict_demand(i, new_days, last_day)
            new_supply = np.minimum(predict_supply(i, new_days, last_day), new_demand)
            out.loc[~known, "Demand (MW)"] = new_demand.round()
            out.loc[~known, "Supply (MW)"] = new_supply.round()
            out.loc[~known, "Load shedding (MW)"] = (new_demand - new_supply).round()
        rec = records.loc[days[known]]
        out.loc[known, "Holiday"] = rec["Holiday name/Weekend"].values
        out.loc[known, "Demand (MW)"] = rec[area + "_demand"].values
        out.loc[known, "Supply (MW)"] = rec[area + "_supply"].values
        out.loc[known, "Load shedding (MW)"] = rec[area + "_load"].values
        result.append(out)
 
    result = pd.concat(result).sort_values(["Date", "Area"])
    result["Date"] = result["Date"].dt.strftime("%d-%m-%Y")
    return result
 
def plot(result, start, end, area_list):                            # For Graphical Representation (You can modify this part as per your requirements)
    columns = ["Demand (MW)", "Supply (MW)", "Load shedding (MW)"]
    where = area_list[0] if len(area_list) == 1 else "All areas"
 
    if start == end:  # one day: bars for every area
        result.set_index("Area")[columns].plot(kind="bar", figsize=(10, 5), rot=45)
        plt.title(f"{where} - {start:%d-%m-%Y}" + (" (forecast)" if start > last_day else ""))
        plt.ylabel("MW")
    else:  # many days: one line per area in three panels
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for ax, col in zip(axes, columns):
            for area, part in result.groupby("Area"):
                ax.plot(pd.to_datetime(part["Date"], format="%d-%m-%Y"), part[col], label=area)
            ax.set_ylabel(col)
            ax.grid(alpha=0.3)
            if end > last_day:  # grey area = forecast days
                ax.axvspan(max(start, last_day), end, color="grey", alpha=0.15)
        axes[0].set_title(f"{where} - {start:%d-%m-%Y} to {end:%d-%m-%Y} (grey = forecast)")
        axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.autofmt_xdate()
 
    plt.tight_layout()
    plt.savefig("result.png", dpi=120)
    plt.show()
 
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    else:
        try:
            start, end = read_dates(input("Enter a date or a timeline (e.g. 18/12/2026 or 18/11/2026 to 17/01/2027): "))
            print("\nAreas:", ", ".join(areas))
            area_text = input("Area name (press Enter for all areas): ").strip().title()
            if area_text and area_text not in areas:
                raise ValueError
        except ValueError:
            print("Invalide input, please try again after some time.")
        else:
            predict_demand = models[MODEL](last_day, demand)
            predict_supply = models[MODEL](last_day, supply)
            area_list = [area_text] if area_text else areas
            result = search(predict_demand, predict_supply, start, end, area_list)
            print()
            print(result.to_string(index=False))
            plot(result, start, end, area_list)