
import os

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Waterloo Weather Dashboard",
    page_icon="🌤️",
    layout="wide",
)

st.title("🌤️ Waterloo Weather — Live Dashboard")
st.caption("Hourly weather data streamed into Neon PostgreSQL every 2 seconds.")


def get_weather_data(limit=100):
    """Read the latest weather records from Neon PostgreSQL."""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing from .env")

    conn = psycopg2.connect(database_url, connect_timeout=10)

    try:
        query = """
            SELECT
                time,
                temperature_2m,
                relative_humidity_2m,
                precipitation,
                wind_speed_10m
            FROM public.weather_stream
            ORDER BY time DESC
            LIMIT %s
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(limit,),
        )
    finally:
        conn.close()

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

    return df


@st.fragment(run_every="2s")
def live_dashboard():
    """Refresh weather charts from the database every two seconds."""
    try:
        df = get_weather_data()
    except Exception as exc:
        st.error(f"Unable to load weather data: {exc}")
        return

    if df.empty:
        st.info("Waiting for weather records to arrive...")
        return

    latest = df.iloc[-1]

    st.subheader("Latest Weather Reading")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Temperature", f"{latest['temperature_2m']:.1f} °C")
    col2.metric("Humidity", f"{latest['relative_humidity_2m']:.0f}%")
    col3.metric("Wind Speed", f"{latest['wind_speed_10m']:.1f} km/h")
    col4.metric("Precipitation", f"{latest['precipitation']:.1f} mm")

    st.caption(
        f"Latest weather timestamp: {latest['time']} "
        f"| Displaying {len(df)} recent records"
    )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            df,
            x="time",
            y="temperature_2m",
            title="Temperature Over Time",
            labels={
                "time": "Date and Time",
                "temperature_2m": "Temperature (°C)",
            },
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            df,
            x="time",
            y="relative_humidity_2m",
            title="Relative Humidity Over Time",
            labels={
                "time": "Date and Time",
                "relative_humidity_2m": "Humidity (%)",
            },
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            df,
            x="time",
            y="precipitation",
            title="Precipitation Over Time",
            labels={
                "time": "Date and Time",
                "precipitation": "Precipitation (mm)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            df,
            x="time",
            y="wind_speed_10m",
            title="Wind Speed Over Time",
            labels={
                "time": "Date and Time",
                "wind_speed_10m": "Wind Speed (km/h)",
            },
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Database Records")
    st.dataframe(df.sort_values("time", ascending=False))


live_dashboard()
