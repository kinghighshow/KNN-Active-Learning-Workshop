"""Step 2, cleaning and preprocessing: labels, leakage removal, split, imputer + scaler fit on train only. Owner: Eche.

Part of the KNN Workshop pipeline: DataLoader -> Preprocessor -> KNNModel -> Evaluator,
chained by KNNPipeline. Used by KNN_Workshop_Solution.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


class Preprocessor:
    LEAKAGE_COLUMNS = ["weather_code", "rain", "snowfall", "precipitation"]
    FEATURES = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
        "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
        "wind_gusts_10m", "hour", "month",
    ]
    CLASSES = ["Clear", "Cloudy", "Rain", "Snow"]

    def __init__(self, features=None):
        self.features = features or self.FEATURES
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    @staticmethod
    def to_condition(code):
        if pd.isna(code):
            return np.nan
        code = int(code)
        if code <= 1:
            return "Clear"
        if code <= 3:
            return "Cloudy"
        if 51 <= code <= 67 or 80 <= code <= 82:
            return "Rain"
        if 71 <= code <= 77 or 85 <= code <= 86:
            return "Snow"
        return np.nan

    def clean(self, df):
        data = df.drop_duplicates(subset="time").copy()
        data["time"] = pd.to_datetime(data["time"])
        data["condition"] = data["weather_code"].map(self.to_condition)
        data["hour"] = data["time"].dt.hour
        data["month"] = data["time"].dt.month
        data = data.dropna(subset=["condition"]).reset_index(drop=True)
        self.rows_dropped = len(df) - len(data)
        self.time = data["time"]
        return data[self.features], data["condition"]

    def split(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    def fit_transform(self, X_train):
        return self.scaler.fit_transform(self.imputer.fit_transform(X_train))

    def transform(self, X):
        return self.scaler.transform(self.imputer.transform(X))
