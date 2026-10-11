"""Step 1, data acquisition: Open-Meteo API -> Neon table knn_weather (CSV fallback). Owner: John.

Part of the KNN Workshop pipeline: DataLoader -> Preprocessor -> KNNModel -> Evaluator,
chained by KNNPipeline. Used by KNN_Workshop_Solution.ipynb.
"""

import os
from pathlib import Path

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

from src.config import DATA_PATH


class DataLoader:
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    TABLE = "knn_weather"
    COLUMNS = [
        "time", "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "apparent_temperature", "precipitation", "rain", "snowfall", "weather_code",
        "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
        "wind_gusts_10m",
    ]

    def __init__(self, csv_path=DATA_PATH, latitude=43.4643, longitude=-80.5204, year=2025):
        self.csv_path = Path(csv_path)
        self.latitude = latitude
        self.longitude = longitude
        self.year = year
        self.database_url = os.getenv("DATABASE_URL")
        self.source = None

    def fetch_from_api(self):
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": f"{self.year}-01-01",
            "end_date": f"{self.year}-12-31",
            "hourly": self.COLUMNS[1:],
            "timezone": "America/Toronto",
        }
        response = requests.get(self.API_URL, params=params, timeout=60)
        response.raise_for_status()
        df = pd.DataFrame(response.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, index=False)
        return df[self.COLUMNS]

    def read_csv(self):
        return pd.read_csv(self.csv_path, parse_dates=["time"])[self.COLUMNS]

    def save_to_database(self, df):
        column_types = ", ".join(
            f"{c} INTEGER" if c == "weather_code" else f"{c} DOUBLE PRECISION"
            for c in self.COLUMNS[1:]
        )
        rows = list(zip(df["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        *(df[c].tolist() for c in self.COLUMNS[1:])))
        conn = psycopg2.connect(self.database_url, connect_timeout=10)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(f"CREATE TABLE IF NOT EXISTS {self.TABLE} (time TIMESTAMP PRIMARY KEY, {column_types})")
                cur.execute(f"SELECT COUNT(*) FROM {self.TABLE}")
                if cur.fetchone()[0] == 0:
                    execute_values(cur, f"INSERT INTO {self.TABLE} ({', '.join(self.COLUMNS)}) VALUES %s", rows)
        finally:
            conn.close()

    def load_from_database(self):
        conn = psycopg2.connect(self.database_url, connect_timeout=10)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {', '.join(self.COLUMNS)} FROM {self.TABLE} ORDER BY time")
                df = pd.DataFrame(cur.fetchall(), columns=self.COLUMNS)
        finally:
            conn.close()
        df["time"] = pd.to_datetime(df["time"])
        return df

    def load(self, refresh=False):
        raw = self.fetch_from_api() if refresh or not self.csv_path.exists() else self.read_csv()
        if self.database_url:
            try:
                self.save_to_database(raw)
                self.source = f"Neon table {self.TABLE}"
                return self.load_from_database()
            except psycopg2.Error as exc:
                print("Could not reach Neon, using the local copy instead:", exc)
        self.source = str(self.csv_path)
        return raw
