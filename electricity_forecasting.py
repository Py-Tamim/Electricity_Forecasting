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
 
MODEL = "XGBoost"  # model used for the date search: "XGBoost" or "LSTM"
SEQ = 14           # number of days the LSTM looks back
 

df = pd.read_csv("Bangladesh_Electricity_Dataset_2019_2026.csv")      # Load, modify the data & preprocessing.
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
 
# Step 1: Friday/Saturday with 'No Holiday' become 'Weekend'
is_weekend_day = df["Day of the week"].isin(["Friday", "Saturday"])
is_no_holiday = df["Holiday name/Weekend"] == "No Holiday"
df.loc[is_weekend_day & is_no_holiday, "Holiday name/Weekend"] = "Weekend"
 
# Step 2: every holiday and weekend = 1, 'No Holiday' = 0
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
fixed_holidays = [(2, 21), (3, 17), (3, 26), (4, 14), (5, 1), (12, 16), (12, 25)]

def day_kind(days):
    kind = pd.Series("No Holiday", index=days)
    kind.loc[np.array([(d.month, d.day) in fixed_holidays for d in days])] = "Public holiday"
    kind.loc[days.dayofweek.isin([4, 5])] = "Weekend"
    return kind

def covariates(days, known_until):
    # holiday flag and temperature: real values up to known_until, estimates after that
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
