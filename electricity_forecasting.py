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
 
# ---------------- Load and modify the data ----------------
df = pd.read_csv("Bangladesh_Electricity_Dataset_2019_2026.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
 
# Step 1: Friday/Saturday with 'No Holiday' become 'Weekend'
is_weekend_day = df["Day of the week"].isin(["Friday", "Saturday"])
is_no_holiday = df["Holiday name/Weekend"] == "No Holiday"
df.loc[is_weekend_day & is_no_holiday, "Holiday name/Weekend"] = "Weekend"
 
# Step 2: every holiday and weekend = 1, 'No Holiday' = 0
is_holiday_or_weekend = df["Holiday name/Weekend"] != "No Holiday"
df.loc[is_holiday_or_weekend, "Holiday_cat/weekend"] = 1
df.loc[~is_holiday_or_weekend, "Holiday_cat/weekend"] = 0
 
# Fill empty cells: median for numbers (same year and month first), mode for text
temp_col = "Maximum Temperature in Dhaka was"
df.loc[df[temp_col] == 0, temp_col] = np.nan  # 0 degree is not a real reading for Dhaka
num_cols = df.select_dtypes("number").columns
df[num_cols] = df[num_cols].fillna(df.groupby(["Year", "Month"])[num_cols].transform("median"))
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in ["Day of the week", "Holiday name/Weekend"]:
    df[col] = df[col].fillna(df[col].mode()[0])
