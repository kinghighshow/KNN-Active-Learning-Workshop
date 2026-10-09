import argparse
import csv
import math
import os
import time
from datetime import datetime

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


class WeatherDatabase:
    TABLE = "weather_stream"

    COLUMNS = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "surface_pressure",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "latitude",
        "longitude",
        "elevation",
        "timezone",
    ]

    def __init__(self, database_url=None):
        load_dotenv()
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is missing. Add it to your .env file.")
        self.conn = None

    def connect(self):
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(self.database_url)
        return self.conn

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()

    def create_table(self):
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.TABLE} (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP UNIQUE NOT NULL,
            temperature_2m REAL,
            relative_humidity_2m REAL,
            dew_point_2m REAL,
            apparent_temperature REAL,
            precipitation REAL,
            rain REAL,
            snowfall REAL,
            weather_code INTEGER,
            surface_pressure REAL,
            cloud_cover REAL,
            wind_speed_10m REAL,
            wind_direction_10m REAL,
            wind_gusts_10m REAL,
            latitude REAL,
            longitude REAL,
            elevation REAL,
            timezone TEXT,
            streamed_at TIMESTAMP DEFAULT NOW()
        );
        """
        with self.connect().cursor() as cur:
            cur.execute(sql)
        self.conn.commit()

    def insert_reading(self, record):
        values = [self._clean(record.get(col)) for col in self.COLUMNS]
        cols = ", ".join(self.COLUMNS)
        placeholders = ", ".join(["%s"] * len(self.COLUMNS))
        sql = f"""
        INSERT INTO {self.TABLE} ({cols})
        VALUES ({placeholders})
        ON CONFLICT (time) DO NOTHING;
        """
        with self.connect().cursor() as cur:
            cur.execute(sql, values)
        self.conn.commit()

    def fetch_latest(self, limit=100):
        sql = f"""
        SELECT * FROM (
            SELECT * FROM {self.TABLE} ORDER BY time DESC LIMIT %s
        ) recent
        ORDER BY time ASC;
        """
        with self.connect().cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()

    def last_streamed_time(self):
        with self.connect().cursor() as cur:
            cur.execute(f"SELECT MAX(time) FROM {self.TABLE};")
            return cur.fetchone()[0]

    def count_rows(self):
        with self.connect().cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.TABLE};")
            return cur.fetchone()[0]

    def reset_table(self):
        with self.connect().cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.TABLE} RESTART IDENTITY;")
        self.conn.commit()

    @staticmethod
    def _clean(value):
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value


class WeatherStreamer:
    def __init__(self, csv_path, interval=2):
        self.csv_path = csv_path
        self.interval = interval
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def read_records(self, start_after=None):
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                record = self._parse(row)
                if start_after and record["time"] <= start_after:
                    continue
                yield record

    def run(self, start_after=None, limit=None):
        sent = 0
        for record in self.read_records(start_after):
            for callback in self.subscribers:
                callback(record)
            sent += 1
            if limit and sent >= limit:
                break
            time.sleep(self.interval)
        return sent

    @staticmethod
    def _parse(row):
        record = {}
        for key, value in row.items():
            if key == "time":
                record[key] = datetime.fromisoformat(value)
            elif key == "timezone":
                record[key] = value
            elif value in ("", None):
                record[key] = None
            elif key == "weather_code":
                record[key] = int(float(value))
            else:
                record[key] = float(value)
        return record




def print_record(record):
    print(
        f"{record['time']}  |  temp {record['temperature_2m']}°C  |  "
        f"humidity {record['relative_humidity_2m']}%  |  wind {record['wind_speed_10m']} km/h"
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stream weather CSV rows into Neon every 2 seconds.")
    parser.add_argument("--csv", default=os.getenv("CSV_PATH", "../data/waterloo_weather_2025.csv"))
    parser.add_argument("--interval", type=float, default=float(os.getenv("STREAM_INTERVAL", 2)))
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many rows")
    parser.add_argument("--reset", action="store_true", help="Clear the table and start from the first row")
    args = parser.parse_args()

    db = WeatherDatabase()
    db.create_table()

    if args.reset:
        db.reset_table()
        print("Table cleared, starting from the beginning.")

    resume_from = db.last_streamed_time()
    if resume_from:
        print(f"Resuming after {resume_from} ({db.count_rows()} rows already saved).")

    streamer = WeatherStreamer(args.csv, interval=args.interval)
    streamer.subscribe(db.insert_reading)
    streamer.subscribe(print_record)

    print(f"Streaming {args.csv} every {args.interval}s. Press Ctrl+C to stop.\n")
    try:
        sent = streamer.run(start_after=resume_from, limit=args.limit)
        print(f"\nDone. Sent {sent} rows.")
    except KeyboardInterrupt:
        print("\nStopped. Run it again and it will pick up where it left off.")
    finally:
        print(f"Rows in {db.TABLE}: {db.count_rows()}")
        db.close()


if __name__ == "__main__":
    main()