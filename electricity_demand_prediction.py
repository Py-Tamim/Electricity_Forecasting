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