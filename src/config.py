"""Shared settings for the KNN Workshop pipeline.

Part of the KNN Workshop pipeline: DataLoader -> Preprocessor -> KNNModel -> Evaluator,
chained by KNNPipeline. Used by KNN_Workshop_Solution.ipynb.
"""

from pathlib import Path

RANDOM_STATE = 42
DATA_PATH = Path("data/waterloo_weather_2025.csv")
