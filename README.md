# K-Nearest Neighbors Workshop

KNN classifier that predicts the hourly weather condition in Waterloo, ON (Clear, Cloudy, Rain or Snow) using 2025 data from the Open-Meteo API stored in Neon PostgreSQL. The solution is in `KNN_Workshop_Solution.ipynb`, and more details are in `doc.txt`.

**Team:** Sultan Atanda (9114837) · John (9115726) · Antonio Sainz (9072844) · Eche (9078881)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and add your Neon `DATABASE_URL`. Without it, the notebook uses `data/waterloo_weather_2025.csv`.
2. Open `KNN_Workshop_Solution.ipynb` and click **Run All**.\
3. Other details included in the doc.txt file
