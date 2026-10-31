import re
from datetime import datetime
from tabnanny import check
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("dhaka_traffic_dataset.csv") #Load The CSV File.

for column in df.select_dtypes(include="number").columns:
    df[column] = df[column].fillna(df[column].median())

for column in df.select_dtypes(exclude="number").columns:
    most_common = df[column].mode()
    df[column] = df[column].fillna(
        most_common.iloc[0] if not most_common.empty else "Unknown"
    )

traffic_labels = {
    "A": "No traffic",
    "B": "Light traffic",
    "C": "Moderate traffic",
    "D": "Heavy traffic",
    "E": "Very heavy traffic",
    "F": "Extreme traffic",
}
df["los_category"] = df["los_category"].astype(str).str.strip().str.upper().str[0]
df["traffic_level"] = df["los_category"].map(traffic_labels)
df = df.dropna(subset=["corridor_name", "traffic_level"]).copy()

severity_order = {
    "No traffic": 0,
    "Light traffic": 1,
    "Moderate traffic": 2,
    "Heavy traffic": 3,
    "Very heavy traffic": 4,
    "Extreme traffic": 5,
}

numeric_features = ["hour_of_day", "day_of_week", "is_weekend", "sin_time", "cos_time"] #[Starting Model Training]
categorical_features = ["corridor_name"]
features = categorical_features + numeric_features
time_columns = ["sin_time", "cos_time"]
target_columns = [
    "flow_q_vph",
    "density_k_vpk",
    "speed_v_kmh",
    "vc_ratio",
    "non_motorized_pct",
    "cng_three_wheeler_pct",
    "bus_minibus_pct",
    "private_car_suv_pct",
    "motorcycle_pct",
    "cav_v2x_penetration_rate",
]

column_titles = {
    "sin_time": "sin_time",
    "cos_time": "cos_time",
    "flow_q_vph": "Flow (vehicles/hour)",
    "density_k_vpk": "Density (vehicles/km)",
    "speed_v_kmh": "Speed (km/h)",
    "vc_ratio": "Volume/Capacity ratio",
    "non_motorized_pct": "Non-motorized (%)",
    "cng_three_wheeler_pct": "CNG three-wheeler (%)",
    "bus_minibus_pct": "Bus / minibus (%)",
    "private_car_suv_pct": "Private car / SUV (%)",
    "motorcycle_pct": "Motorcycle (%)",
    "cav_v2x_penetration_rate": "CAV V2X penetration rate",
    "severity": "Traffic severity (0-5)",
}

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numbers", SimpleImputer(strategy="median"), numeric_features),
            (
                "categories",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

