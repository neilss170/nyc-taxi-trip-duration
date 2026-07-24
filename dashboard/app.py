# ============================================================
# NYC Taxi Trip Duration Dashboard - Clean SaaS Style
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NYC Taxi Analytics Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COLORS
# ============================================================

COLOR_BG = "#F6F8FB"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#E5E7EB"

COLOR_BLUE = "#2563EB"
COLOR_TEAL = "#14B8A6"
COLOR_GREEN = "#10B981"
COLOR_ORANGE = "#D97706"
COLOR_ORANGE_LIGHT = "#FFFBEB"
COLOR_RED = "#EF4444"
COLOR_PURPLE = "#8B5CF6"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
}

.stApp {
    background-color: #F6F8FB !important;
    color: #0F172A !important;
}

.block-container {
    padding-top: 1.2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background: #F6F8FB !important;
    height: 2.75rem !important;
}

div[data-testid="stDecoration"] {
    display: none !important;
}

#MainMenu {
    visibility: hidden !important;
}

footer {
    visibility: hidden !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #D1D5DB !important;
}

section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

section[data-testid="stSidebar"] label {
    color: #111827 !important;
    font-weight: 850 !important;
}

section[data-testid="stSidebar"] details {
    background-color: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 8px !important;
    margin-bottom: 12px !important;
}

section[data-testid="stSidebar"] details summary {
    color: #111827 !important;
    font-weight: 900 !important;
}

section[data-testid="stSidebar"] details[open] summary {
    color: #2563EB !important;
}

/* Selectbox / dropdown fix */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #111827 !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #111827 !important;
    background-color: #FFFFFF !important;
}

div[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
}

div[data-baseweb="popover"] div {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

div[data-baseweb="popover"] li {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #E0F2FE !important;
    color: #075985 !important;
}

div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #DBEAFE !important;
    color: #1D4ED8 !important;
    font-weight: 850 !important;
}

/* Sidebar toggle */
button[kind="header"],
[data-testid="collapsedControl"],
button[title="Open sidebar"],
button[title="Close sidebar"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.22) !important;
    opacity: 1 !important;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    padding: 28px 30px;
    border-radius: 24px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
    margin-bottom: 22px;
}

.title {
    font-size: 42px;
    font-weight: 950 !important;
    color: #0F172A !important;
    margin: 0;
    line-height: 1.1;
}

.subtitle {
    font-size: 15px;
    font-weight: 600;
    color: #64748B !important;
    margin-top: 10px;
    max-width: 900px;
    line-height: 1.45;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background-color: #DBEAFE;
    color: #1D4ED8;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 14px;
}

/* Cards */
.metric-card {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
    min-height: 128px;
}

.metric-label {
    font-size: 13px;
    color: #64748B;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.metric-value {
    font-size: 30px;
    color: #0F172A;
    font-weight: 950;
    margin-top: 8px;
    line-height: 1.1;
}

.metric-note {
    font-size: 13px;
    color: #2563EB;
    font-weight: 800;
    margin-top: 10px;
}

.section-title {
    font-size: 24px;
    font-weight: 950 !important;
    color: #0F172A !important;
    margin-bottom: 4px;
}

.section-subtitle {
    font-size: 14px;
    font-weight: 650;
    color: #64748B !important;
    margin-bottom: 16px;
}

.summary-card {
    background: #FFFBEB;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #FDE68A;
    box-shadow: 0 8px 22px rgba(217, 119, 6, 0.06);
    margin-bottom: 16px;
}

.summary-title {
    color: #92400E !important;
    font-size: 19px;
    font-weight: 950 !important;
    margin-bottom: 6px;
}

.summary-text {
    color: #78350F !important;
    font-size: 13.5px;
    font-weight: 650;
    line-height: 1.45;
}

.insight-card {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
    margin-bottom: 14px;
    border-left: 6px solid #2563EB;
}

.insight-title {
    font-size: 18px;
    font-weight: 950 !important;
    color: #0F172A !important;
    margin-bottom: 6px;
}

.insight-text {
    font-size: 13.5px;
    font-weight: 650;
    color: #475569 !important;
    line-height: 1.45;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 10px 18px;
    border: 1px solid #E5E7EB;
    color: #111827;
    font-weight: 900;
}

.stTabs [aria-selected="true"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
}

div[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border-radius: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def load_data():
    paths = [
        Path("../data/nyc_taxi_sample.csv"),
        Path("../data/nyc_taxi_trip_duration.csv"),
        Path("data/nyc_taxi_sample.csv"),
        Path("data/nyc_taxi_trip_duration.csv"),
        Path("./nyc_taxi_sample.csv"),
        Path("./nyc_taxi_trip_duration.csv"),
    ]

    for path in paths:
        if path.exists():
            return pd.read_csv(path)

    st.error("NYC taxi data not found. Add the full dataset or the included sample.")
    st.stop()


df = load_data()

# ============================================================
# FEATURE PREPARATION
# ============================================================

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_day_name"] = df["pickup_datetime"].dt.day_name()
df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
df["pickup_month"] = df["pickup_datetime"].dt.month
df["trip_duration_minutes"] = df["trip_duration"] / 60

df["is_weekend"] = df["pickup_dayofweek"].apply(lambda x: 1 if x in [5, 6] else 0)

df["is_rush_hour"] = df["pickup_hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 20) else 0
)

df["is_office_hour"] = df["pickup_hour"].apply(lambda x: 1 if 9 <= x <= 18 else 0)

# Additional time-based features added from the updated feature engineering notebook
df["is_morning_peak"] = df["pickup_hour"].between(7, 10).astype(int)
df["is_evening_peak"] = df["pickup_hour"].between(16, 20).astype(int)

df["is_late_night"] = df["pickup_hour"].apply(
    lambda x: 1 if (x >= 22) or (x <= 5) else 0
)

df["is_business_day"] = (df["pickup_dayofweek"] < 5).astype(int)
df["is_month_start"] = df["pickup_datetime"].dt.is_month_start.astype(int)
df["is_month_end"] = df["pickup_datetime"].dt.is_month_end.astype(int)


def create_time_slot(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


df["time_slot"] = df["pickup_hour"].apply(create_time_slot)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


df["trip_distance_km"] = haversine_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)

# Advanced distance features
df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]

df["manhattan_distance_km"] = (np.abs(df["lat_diff"]) + np.abs(df["lon_diff"])) * 111

df["direction_angle"] = np.degrees(np.arctan2(df["lat_diff"], df["lon_diff"]))

# Airport-related features
airport_locations = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
}

pickup_airport_distances = []
dropoff_airport_distances = []

for airport_name, (airport_lat, airport_lon) in airport_locations.items():
    pickup_airport_distances.append(
        haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            airport_lat,
            airport_lon,
        )
    )

    dropoff_airport_distances.append(
        haversine_distance(
            df["dropoff_latitude"],
            df["dropoff_longitude"],
            airport_lat,
            airport_lon,
        )
    )

df["pickup_airport_distance_km"] = np.min(pickup_airport_distances, axis=0)
df["dropoff_airport_distance_km"] = np.min(dropoff_airport_distances, axis=0)

airport_threshold_km = 3

df["is_pickup_near_airport"] = (
    df["pickup_airport_distance_km"] <= airport_threshold_km
).astype(int)

df["is_dropoff_near_airport"] = (
    df["dropoff_airport_distance_km"] <= airport_threshold_km
).astype(int)

conditions = [
    (df["is_pickup_near_airport"] == 1) & (df["is_dropoff_near_airport"] == 1),
    (df["is_pickup_near_airport"] == 1),
    (df["is_dropoff_near_airport"] == 1),
]

choices = ["airport_to_airport", "airport_pickup", "airport_dropoff"]

df["airport_trip_type"] = np.select(conditions, choices, default="non_airport_trip")

# Interaction features
df["distance_rush_interaction"] = df["trip_distance_km"] * df["is_rush_hour"]
df["distance_late_night_interaction"] = df["trip_distance_km"] * df["is_late_night"]
df["distance_weekend_interaction"] = df["trip_distance_km"] * df["is_weekend"]
df["passenger_distance_interaction"] = df["passenger_count"] * df["trip_distance_km"]

# Visual-clean dataset
raw_total_rows = len(df)

df = df[
    (df["passenger_count"] > 0)
    & (df["trip_duration"] > 0)
    & (df["trip_duration_minutes"] <= 180)
    & (df["trip_distance_km"] <= 100)
].copy()

cleaned_total_rows = len(df)

# ============================================================
# HELPERS
# ============================================================


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title, text):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_card(title, text):
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-title">{title}</div>
            <div class="summary-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_header(title, subtitle):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def clean_plot(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=25, r=25, t=60, b=35),
        font=dict(family="Inter, Segoe UI, Arial", size=13, color=COLOR_TEXT),
        title=dict(font=dict(size=18, color=COLOR_TEXT), x=0.02, xanchor="left"),
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=COLOR_BORDER,
            font_size=13,
            font_family="Inter, Segoe UI, Arial",
            font_color=COLOR_TEXT,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
        ),
        colorway=[
            COLOR_BLUE,
            COLOR_TEAL,
            COLOR_PURPLE,
            COLOR_ORANGE,
            COLOR_GREEN,
            COLOR_RED,
        ],
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_MUTED),
        title_font=dict(color=COLOR_TEXT),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EEF2F7",
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_MUTED),
        title_font=dict(color=COLOR_TEXT),
    )

    return fig


def show_chart(fig, height=400):
    fig = clean_plot(fig, height=height)
    st.plotly_chart(fig, use_container_width=True)


def styled_table(dataframe):
    return dataframe.style.set_table_styles(
        [
            {
                "selector": "thead th",
                "props": [
                    ("background-color", COLOR_ORANGE),
                    ("color", "white"),
                    ("font-weight", "bold"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", COLOR_ORANGE_LIGHT)],
            },
        ]
    )


# ============================================================
# SIDEBAR FILTERS - CHECKBOX STYLE
# ============================================================

st.sidebar.markdown("## 🚕 NYC Taxi Analytics")
st.sidebar.caption("Trip Duration SaaS Dashboard")
st.sidebar.markdown("---")


def checkbox_filter(title, options):
    selected = []

    all_key = f"{title}_select_all"
    clear_key = f"{title}_clear_all"

    with st.sidebar.expander(title, expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            select_all_clicked = st.button("Select All", key=all_key)

        with col2:
            clear_all_clicked = st.button("Clear All", key=clear_key)

        for option in options:
            option_key = f"{title}_{option}"

            if option_key not in st.session_state:
                st.session_state[option_key] = True

            if select_all_clicked:
                st.session_state[option_key] = True

            if clear_all_clicked:
                st.session_state[option_key] = False

            checked = st.checkbox(str(option), key=option_key)

            if checked:
                selected.append(option)

    return selected


vendor_filter = checkbox_filter("Vendor ID", sorted(df["vendor_id"].unique()))

month_filter = checkbox_filter("Pickup Month", sorted(df["pickup_month"].unique()))

weekday_filter = checkbox_filter(
    "Weekday",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)

time_slot_filter = checkbox_filter(
    "Time Slot", ["Morning", "Afternoon", "Evening", "Night"]
)

passenger_filter = checkbox_filter(
    "Passenger Count", sorted(df["passenger_count"].unique())
)

with st.sidebar.expander("Range Filters", expanded=False):

    hour_range = st.slider(
        "Pickup Hour",
        0,
        23,
        (0, 23),
    )

    duration_range = st.slider(
        "Trip Duration (Minutes)",
        float(df["trip_duration_minutes"].min()),
        float(df["trip_duration_minutes"].quantile(0.99)),
        (
            float(df["trip_duration_minutes"].min()),
            float(df["trip_duration_minutes"].quantile(0.99)),
        ),
    )

    distance_range = st.slider(
        "Trip Distance (KM)",
        float(df["trip_distance_km"].min()),
        float(df["trip_distance_km"].quantile(0.99)),
        (
            float(df["trip_distance_km"].min()),
            float(df["trip_distance_km"].quantile(0.99)),
        ),
    )
# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df[
    (df["vendor_id"].isin(vendor_filter))
    & (df["pickup_month"].isin(month_filter))
    & (df["pickup_day_name"].isin(weekday_filter))
    & (df["time_slot"].isin(time_slot_filter))
    & (df["passenger_count"].isin(passenger_filter))
    & (df["pickup_hour"].between(hour_range[0], hour_range[1]))
    & (df["trip_duration_minutes"].between(duration_range[0], duration_range[1]))
    & (df["trip_distance_km"].between(distance_range[0], distance_range[1]))
].copy()

if filtered_df.empty:
    st.warning("No records match selected filters. Adjust sidebar filters.")
    st.stop()

filtered_map_df = filtered_df.sample(min(25000, len(filtered_df)), random_state=42)

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Trips", f"{filtered_df.shape[0]:,}")
st.sidebar.metric("Cleaned Trips", f"{cleaned_total_rows:,}")
st.sidebar.metric("Raw Rows", f"{raw_total_rows:,}")
st.sidebar.metric(
    "Avg Duration", f"{filtered_df['trip_duration_minutes'].mean():.1f} min"
)
st.sidebar.metric("Avg Distance", f"{filtered_df['trip_distance_km'].mean():.2f} km")

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div class="badge">NYC Taxi Trip Duration</div>
        <div class="title">Taxi Trip Analytics Dashboard</div>
        <div class="subtitle">
            Interactive dashboard for trip duration, time patterns, distance, location, feature creation, and business inference.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# KPI CARDS
# ============================================================

total_trips = filtered_df.shape[0]
avg_duration = filtered_df["trip_duration_minutes"].mean()
median_duration = filtered_df["trip_duration_minutes"].median()
avg_distance = filtered_df["trip_distance_km"].mean()
total_passengers = int(filtered_df["passenger_count"].sum())

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    metric_card("Trips", f"{total_trips:,}", "Filtered view")

with k2:
    metric_card("Avg Duration", f"{avg_duration:.1f} min", "Mean trip time")

with k3:
    metric_card("Median Duration", f"{median_duration:.1f} min", "Robust center")

with k4:
    metric_card("Avg Distance", f"{avg_distance:.2f} km", "Haversine distance")

with k5:
    metric_card("Passengers", f"{total_passengers:,}", "Total filtered passengers")

st.markdown("")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Summary",
        "Time Heatmap",
        "Trip & Distance",
        "Location Map",
        "Feature Creation",
        "Feature Impact",
        "Data Quality",
        "Insights",
    ]
)

# ============================================================
# TAB 1 - SUMMARY
# ============================================================

with tab1:
    chart_header(
        "Summary View",
        "Compact overview of trips, vendors, passengers, duration, and time slots.",
    )

    col1, col2 = st.columns([1.4, 1])

    with col1:
        hist_data = np.histogram(filtered_df["trip_duration_minutes"], bins=35)
        duration_df = pd.DataFrame(
            {"duration_minutes": hist_data[1][:-1], "trips": hist_data[0]}
        )

        fig = px.area(
            duration_df,
            x="duration_minutes",
            y="trips",
            title="Trip Duration Distribution",
            markers=True,
        )

        fig.update_traces(
            line=dict(color=COLOR_TEAL, width=3),
            fillcolor="rgba(20,184,166,0.20)",
            marker=dict(size=5, color=COLOR_TEAL),
        )

        show_chart(fig, height=390)

    with col2:
        fig = px.pie(
            filtered_df,
            names="vendor_id",
            title="Vendor Share",
            hole=0.65,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="white", width=3)),
        )

        show_chart(fig, height=390)

    col3, col4 = st.columns(2)

    with col3:
        passenger_counts = filtered_df["passenger_count"].value_counts().reset_index()
        passenger_counts.columns = ["passenger_count", "trips"]

        fig = px.bar(
            passenger_counts,
            x="passenger_count",
            y="trips",
            title="Passenger Count Distribution",
            text_auto=True,
        )

        fig.update_traces(marker_color=COLOR_BLUE, opacity=0.9)
        show_chart(fig, height=370)

    with col4:
        slot_counts = filtered_df["time_slot"].value_counts().reset_index()
        slot_counts.columns = ["time_slot", "trips"]

        fig = px.bar(
            slot_counts,
            x="time_slot",
            y="trips",
            title="Trips by Time Slot",
            text_auto=True,
        )

        fig.update_traces(marker_color=COLOR_PURPLE, opacity=0.9)
        show_chart(fig, height=370)

# ============================================================
# TAB 2 - TIME HEATMAP
# ============================================================

with tab2:
    chart_header(
        "Time Heatmap",
        "Trips and duration patterns by pickup hour and weekday.",
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    heatmap_data = filtered_df.pivot_table(
        values="trip_duration_minutes",
        index="pickup_day_name",
        columns="pickup_hour",
        aggfunc="mean",
    ).reindex(day_order)

    fig = px.imshow(
        heatmap_data,
        text_auto=".1f",
        color_continuous_scale="RdBu_r",
        title="Average Trip Duration by Day and Hour",
        aspect="auto",
    )

    show_chart(fig, height=520)

    col1, col2 = st.columns(2)

    with col1:
        hourly_trips = filtered_df.groupby("pickup_hour", as_index=False).agg(
            trips=("trip_duration", "count")
        )

        fig = px.line(
            hourly_trips,
            x="pickup_hour",
            y="trips",
            markers=True,
            title="Trips by Pickup Hour",
        )

        fig.update_traces(line=dict(color=COLOR_BLUE, width=3))
        show_chart(fig, height=360)

    with col2:
        hourly_duration = filtered_df.groupby("pickup_hour", as_index=False).agg(
            avg_duration=("trip_duration_minutes", "mean")
        )

        fig = px.line(
            hourly_duration,
            x="pickup_hour",
            y="avg_duration",
            markers=True,
            title="Average Duration by Pickup Hour",
        )

        fig.update_traces(line=dict(color=COLOR_TEAL, width=3))
        show_chart(fig, height=360)

# ============================================================
# TAB 3 - TRIP AND DISTANCE
# ============================================================

with tab3:
    chart_header(
        "Trip & Distance",
        "Relationship between distance, duration, passenger count, and vendor.",
    )

    sample_df = filtered_df.sample(min(35000, len(filtered_df)), random_state=42)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            sample_df,
            x="trip_distance_km",
            y="trip_duration_minutes",
            color="vendor_id",
            size="passenger_count",
            title="Distance vs Duration",
            hover_data=["pickup_hour", "time_slot", "is_rush_hour"],
        )

        fig.update_traces(marker=dict(opacity=0.65, line=dict(width=1, color="white")))
        show_chart(fig, height=430)

    with col2:
        fig = px.box(
            filtered_df,
            x="vendor_id",
            y="trip_duration_minutes",
            color="vendor_id",
            title="Trip Duration by Vendor",
        )

        show_chart(fig, height=430)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.box(
            filtered_df,
            x="passenger_count",
            y="trip_duration_minutes",
            title="Trip Duration by Passenger Count",
        )

        show_chart(fig, height=390)

    with col4:
        rush_summary = filtered_df.groupby("is_rush_hour", as_index=False).agg(
            avg_duration=("trip_duration_minutes", "mean")
        )

        rush_summary["Rush Hour"] = rush_summary["is_rush_hour"].map(
            {0: "No", 1: "Yes"}
        )

        fig = px.bar(
            rush_summary,
            x="Rush Hour",
            y="avg_duration",
            title="Rush Hour vs Average Duration",
            text_auto=".2f",
        )

        fig.update_traces(marker_color=COLOR_RED, opacity=0.85)
        show_chart(fig, height=390)

# ============================================================
# TAB 4 - LOCATION MAP
# ============================================================

with tab4:
    chart_header(
        "Location Map",
        "Pickup and dropoff concentration using sampled filtered trips.",
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter_mapbox(
            filtered_map_df,
            lat="pickup_latitude",
            lon="pickup_longitude",
            color="trip_duration_minutes",
            zoom=10,
            title="Pickup Locations",
            hover_data=["pickup_hour", "trip_duration_minutes", "passenger_count"],
            color_continuous_scale="Viridis",
        )

        fig.update_layout(mapbox_style="open-street-map")
        show_chart(fig, height=520)

    with col2:
        fig = px.scatter_mapbox(
            filtered_map_df,
            lat="dropoff_latitude",
            lon="dropoff_longitude",
            color="trip_duration_minutes",
            zoom=10,
            title="Dropoff Locations",
            hover_data=["pickup_hour", "trip_duration_minutes", "passenger_count"],
            color_continuous_scale="Plasma",
        )

        fig.update_layout(mapbox_style="open-street-map")
        show_chart(fig, height=520)

# ============================================================
# TAB 5 - FEATURE CREATION
# ============================================================

with tab5:
    chart_header(
        "Feature Creation",
        "Raw columns converted into useful ML features.",
    )

    feature_table = pd.DataFrame(
        {
            "Feature Group": [
                "Datetime Features",
                "Datetime Features",
                "Datetime Features",
                "Datetime Features",
                "Datetime Features",
                "Distance Features",
                "Distance Features",
                "Distance Features",
                "Airport Features",
                "Airport Features",
                "Location Features",
                "Interaction Features",
                "Interaction Features",
                "Encoding",
                "Scaling",
                "Leakage Safety",
            ],
            "Created Feature": [
                "pickup_hour, pickup_dayofweek, pickup_month",
                "is_weekend",
                "is_morning_peak / is_evening_peak",
                "is_late_night",
                "time_slot",
                "trip_distance_km",
                "manhattan_distance_km",
                "direction_angle",
                "is_pickup_near_airport / is_dropoff_near_airport",
                "airport_trip_type",
                "pickup_cluster / dropoff_cluster",
                "distance_rush_interaction",
                "distance_weekend_interaction / distance_late_night_interaction",
                "encoded categorical columns",
                "StandardScaler, MinMaxScaler, RobustScaler",
                "leakage-safe final feature list",
            ],
            "Purpose": [
                "Extract useful time information from pickup datetime",
                "Check weekday vs weekend effect",
                "Capture office commute behavior",
                "Capture night travel behavior",
                "Group trips into readable time periods",
                "Measure straight-line trip distance",
                "Approximate grid-style NYC movement",
                "Capture travel direction",
                "Identify airport-related pickup/dropoff trips",
                "Separate airport and non-airport trips",
                "Convert raw GPS into location zones in notebook",
                "Check distance effect during rush hour",
                "Check distance behavior during weekend and late night",
                "Convert categories into model-readable format",
                "Prepare numerical features for ML algorithms",
                "Avoid target leakage before model building",
            ],
        }
    )

    st.dataframe(styled_table(feature_table), use_container_width=True, height=300)

    feature_preview = filtered_df.copy()

    feature_preview["distance_category"] = pd.cut(
        feature_preview["trip_distance_km"],
        bins=[0, 2, 8, 100],
        labels=["Short", "Medium", "Long"],
    )

    feature_preview["duration_category"] = pd.cut(
        feature_preview["trip_duration_minutes"],
        bins=[0, 10, 30, 180],
        labels=["Quick", "Normal", "Long"],
    )

    feature_preview["is_airport_distance"] = feature_preview["trip_distance_km"].apply(
        lambda x: 1 if x >= 15 else 0
    )

    feature_preview["airport_rush_hour_interaction"] = (
        feature_preview["is_pickup_near_airport"] * feature_preview["is_rush_hour"]
    )

    feature_preview["pickup_day_type"] = feature_preview["is_weekend"].map(
        {0: "Weekday", 1: "Weekend"}
    )

    # These interaction columns are already created in the main feature preparation section.

    preview_cols = [
        "pickup_hour",
        "time_slot",
        "is_weekend",
        "is_rush_hour",
        "is_office_hour",
        "is_late_night",
        "is_morning_peak",
        "is_evening_peak",
        "is_business_day",
        "pickup_day_type",
        "trip_distance_km",
        "manhattan_distance_km",
        "direction_angle",
        "distance_category",
        "duration_category",
        "is_airport_distance",
        "is_pickup_near_airport",
        "is_dropoff_near_airport",
        "airport_trip_type",
        "distance_rush_interaction",
        "distance_late_night_interaction",
        "distance_weekend_interaction",
        "passenger_distance_interaction",
        "airport_rush_hour_interaction",
        "trip_duration_minutes",
    ]

    st.dataframe(
        styled_table(feature_preview[preview_cols].head(50)),
        use_container_width=True,
        height=360,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        rush_summary = feature_preview.groupby("is_rush_hour", as_index=False).agg(
            avg_duration=("trip_duration_minutes", "mean")
        )

        rush_summary["Rush Hour"] = rush_summary["is_rush_hour"].map(
            {0: "No", 1: "Yes"}
        )

        fig = px.bar(
            rush_summary,
            x="Rush Hour",
            y="avg_duration",
            title="Rush Hour vs Avg Duration",
            text_auto=".2f",
        )

        fig.update_traces(marker_color=COLOR_BLUE, opacity=0.9)
        show_chart(fig, height=350)

    with col2:
        office_summary = feature_preview.groupby("is_office_hour", as_index=False).agg(
            avg_duration=("trip_duration_minutes", "mean")
        )

        office_summary["Office Hour"] = office_summary["is_office_hour"].map(
            {0: "No", 1: "Yes"}
        )

        fig = px.bar(
            office_summary,
            x="Office Hour",
            y="avg_duration",
            title="Office Hour vs Avg Duration",
            text_auto=".2f",
        )

        fig.update_traces(marker_color=COLOR_GREEN, opacity=0.9)
        show_chart(fig, height=350)

    with col3:
        late_night_summary = feature_preview.groupby(
            "is_late_night", as_index=False
        ).agg(avg_duration=("trip_duration_minutes", "mean"))

        late_night_summary["Late Night"] = late_night_summary["is_late_night"].map(
            {0: "No", 1: "Yes"}
        )

        fig = px.bar(
            late_night_summary,
            x="Late Night",
            y="avg_duration",
            title="Late Night vs Avg Duration",
            text_auto=".2f",
        )

        fig.update_traces(marker_color=COLOR_ORANGE, opacity=0.9)
        show_chart(fig, height=350)

    col4, col5 = st.columns(2)

    with col4:
        distance_summary = feature_preview.groupby(
            "distance_category", as_index=False
        ).agg(
            avg_duration=("trip_duration_minutes", "mean"),
            trips=("trip_duration_minutes", "count"),
        )

        fig = px.bar(
            distance_summary,
            x="distance_category",
            y="avg_duration",
            title="Distance Category vs Avg Duration",
            text_auto=".2f",
        )

        fig.update_traces(marker_color=COLOR_TEAL, opacity=0.9)
        show_chart(fig, height=350)

    with col5:
        duration_summary = feature_preview.groupby(
            "duration_category", as_index=False
        ).agg(trips=("trip_duration_minutes", "count"))

        fig = px.bar(
            duration_summary,
            x="duration_category",
            y="trips",
            title="Duration Category Distribution",
            text_auto=True,
        )

        fig.update_traces(marker_color=COLOR_PURPLE, opacity=0.9)
        show_chart(fig, height=350)

    insight_card(
        "Inference",
        "Feature creation makes raw datetime and location columns meaningful. Office hour, rush hour, late night, distance category, and duration category help explain trip duration from different business angles.",
    )
# ============================================================
# TAB 6 - FEATURE IMPACT
# ============================================================

with tab6:
    chart_header(
        "Feature Impact",
        "Feature relationships with trip duration before model building.",
    )

    impact_df = filtered_df.copy()

    # Create readable distance category for impact analysis
    impact_df["distance_category"] = pd.cut(
        impact_df["trip_distance_km"],
        bins=[0, 2, 8, 100],
        labels=["Short", "Medium", "Long"],
        include_lowest=True,
    )

    # -------------------------------
    # Correlation with target
    # -------------------------------
    target_col = "trip_duration_minutes"

    impact_corr_cols = [
        "trip_distance_km",
        "manhattan_distance_km",
        "pickup_hour",
        "pickup_month",
        "passenger_count",
        "is_weekend",
        "is_rush_hour",
        "is_office_hour",
        "is_late_night",
        "is_morning_peak",
        "is_evening_peak",
        "is_business_day",
        "pickup_airport_distance_km",
        "dropoff_airport_distance_km",
        "distance_rush_interaction",
        "distance_late_night_interaction",
        "distance_weekend_interaction",
        "passenger_distance_interaction",
    ]

    available_impact_cols = [
        col
        for col in impact_corr_cols
        if col in impact_df.columns and col != target_col
    ]

    correlation_data = []

    for col in available_impact_cols:
        corr_value = impact_df[[col, target_col]].corr().iloc[0, 1]

        if not pd.isna(corr_value):
            correlation_data.append(
                {
                    "Feature": col,
                    "Correlation": corr_value,
                    "Absolute Correlation": abs(corr_value),
                }
            )

    correlation_df = pd.DataFrame(correlation_data).sort_values(
        "Absolute Correlation", ascending=False
    )

    col1, col2 = st.columns([1.35, 1])

    with col1:
        fig = px.bar(
            correlation_df.head(12),
            x="Correlation",
            y="Feature",
            orientation="h",
            title="Feature Relationship with Trip Duration",
            text="Correlation",
        )

        fig.update_traces(
            marker_color=COLOR_BLUE,
            opacity=0.9,
            texttemplate="%{text:.2f}",
            textposition="outside",
        )

        fig.update_layout(yaxis=dict(autorange="reversed"))
        show_chart(fig, height=460)

    with col2:
        st.dataframe(
            styled_table(
                correlation_df[["Feature", "Correlation"]]
                .head(12)
                .round({"Correlation": 3})
            ),
            use_container_width=True,
            height=460,
        )

    insight_card(
        "Correlation Inference",
        "Correlation gives a basic statistical relationship with trip duration. It is useful before model building, but final feature importance should be checked after training models.",
    )

    # -------------------------------
    # Time-based impact cards
    # -------------------------------
    chart_header(
        "Time Feature Impact",
        "Average duration comparison across rush hour, late night, and weekend features.",
    )

    rush_avg = impact_df.groupby("is_rush_hour")["trip_duration_minutes"].mean()
    late_avg = impact_df.groupby("is_late_night")["trip_duration_minutes"].mean()
    weekend_avg = impact_df.groupby("is_weekend")["trip_duration_minutes"].mean()

    rush_diff = rush_avg.get(1, 0) - rush_avg.get(0, 0)
    late_diff = late_avg.get(1, 0) - late_avg.get(0, 0)
    weekend_diff = weekend_avg.get(1, 0) - weekend_avg.get(0, 0)

    t1, t2, t3 = st.columns(3)

    with t1:
        metric_card(
            "Rush Hour Impact",
            f"{rush_diff:.2f} min",
            "Avg duration difference",
        )

    with t2:
        metric_card(
            "Late Night Impact",
            f"{late_diff:.2f} min",
            "Avg duration difference",
        )

    with t3:
        metric_card(
            "Weekend Impact",
            f"{weekend_diff:.2f} min",
            "Avg duration difference",
        )

    time_impact = pd.DataFrame(
        {
            "Feature": ["Rush Hour", "Late Night", "Weekend"],
            "Average Difference (Minutes)": [rush_diff, late_diff, weekend_diff],
        }
    )

    fig = px.bar(
        time_impact,
        x="Feature",
        y="Average Difference (Minutes)",
        title="Time-Based Feature Impact on Average Duration",
        text_auto=".2f",
    )

    fig.update_traces(marker_color=COLOR_TEAL, opacity=0.9)
    show_chart(fig, height=380)

    # -------------------------------
    # Distance and airport impact
    # -------------------------------
    chart_header(
        "Distance and Airport Impact",
        "Average duration and trip count by distance category and airport trip type.",
    )

    col3, col4 = st.columns(2)

    with col3:
        distance_impact = impact_df.groupby(
            "distance_category", as_index=False, observed=False
        ).agg(
            avg_duration=("trip_duration_minutes", "mean"),
            trips=("trip_duration_minutes", "count"),
        )

        fig = px.bar(
            distance_impact,
            x="distance_category",
            y="avg_duration",
            title="Distance Category vs Avg Duration",
            text_auto=".2f",
        )

        fig.update_traces(marker_color=COLOR_PURPLE, opacity=0.9)
        show_chart(fig, height=400)

    with col4:
        airport_impact = impact_df.groupby("airport_trip_type", as_index=False).agg(
            avg_duration=("trip_duration_minutes", "mean"),
            avg_distance=("trip_distance_km", "mean"),
            trips=("trip_duration_minutes", "count"),
        )

        fig = px.bar(
            airport_impact,
            x="airport_trip_type",
            y="avg_duration",
            title="Airport Trip Type vs Avg Duration",
            text_auto=".2f",
        )

        fig.update_traces(marker_color=COLOR_ORANGE, opacity=0.9)
        show_chart(fig, height=400)

    col5, col6 = st.columns(2)

    with col5:
        st.dataframe(
            styled_table(distance_impact.round(2)),
            use_container_width=True,
            height=260,
        )

    with col6:
        st.dataframe(
            styled_table(airport_impact.round(2)),
            use_container_width=True,
            height=260,
        )

    # -------------------------------
    # Model readiness checklist
    # -------------------------------
    chart_header(
        "Model Readiness Checklist",
        "Checks that connect EDA and feature engineering to model building.",
    )

    readiness_checklist = pd.DataFrame(
        {
            "Area": [
                "Problem Type",
                "Target Column",
                "Target Transformation",
                "Datetime Features",
                "Distance Features",
                "Airport Features",
                "Interaction Features",
                "Categorical Encoding",
                "Scaling",
                "Leakage Detection",
                "Final X/y Dataset",
            ],
            "Status": [
                "Regression",
                "trip_duration",
                "trip_duration_log planned in notebook",
                "Done",
                "Done",
                "Done",
                "Done",
                "Done in notebook",
                "Done in notebook",
                "Done",
                "Ready in feature engineering notebook",
            ],
        }
    )

    st.dataframe(
        styled_table(readiness_checklist),
        use_container_width=True,
        height=420,
    )

    summary_card(
        "Feature Impact Inference",
        "Trip duration is mainly influenced by distance, time patterns, airport movement, and interaction features. Final model-based importance should be checked after training regression models.",
    )

# ============================================================
# TAB 7 - DATA QUALITY
# ============================================================

with tab7:
    chart_header(
        "Data Quality",
        "Missing values, duplicates, summary statistics, correlation, and leakage check.",
    )

    missing_values = int(filtered_df.isnull().sum().sum())
    duplicate_rows = int(filtered_df.duplicated().sum())
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include="object").columns.tolist()

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        metric_card("Missing Values", f"{missing_values:,}", "Filtered data")

    with q2:
        metric_card("Duplicate Rows", f"{duplicate_rows:,}", "Filtered data")

    with q3:
        metric_card("Numeric Columns", f"{len(numeric_cols)}", "Available")

    with q4:
        metric_card("Categorical Columns", f"{len(categorical_cols)}", "Available")

    col1, col2 = st.columns(2)

    with col1:
        missing_summary = filtered_df.isnull().sum().reset_index()
        missing_summary.columns = ["Column", "Missing Values"]

        st.dataframe(
            styled_table(missing_summary),
            use_container_width=True,
            height=330,
        )

    with col2:
        st.dataframe(
            styled_table(filtered_df.describe().T.round(2)),
            use_container_width=True,
            height=330,
        )

    corr_cols = [
        "vendor_id",
        "passenger_count",
        "pickup_hour",
        "pickup_month",
        "trip_duration_minutes",
        "trip_distance_km",
        "is_weekend",
        "is_rush_hour",
        "is_office_hour",
        "is_late_night",
        "is_morning_peak",
        "is_evening_peak",
        "manhattan_distance_km",
        "distance_rush_interaction",
    ]

    available_corr_cols = [col for col in corr_cols if col in filtered_df.columns]

    corr_df = filtered_df[available_corr_cols].corr()

    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        title="Numerical Feature Correlation",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )

    show_chart(fig, height=470)

    chart_header(
        "Leakage Detection Before Model Building",
        "Target-derived and future-information columns must be checked before training.",
    )

    leakage_info = pd.DataFrame(
        {
            "Column": [
                "dropoff_datetime",
                "trip_duration",
                "trip_duration_log",
                "trip_duration_minutes",
                "trip_duration_hours",
                "average_speed_kmh",
                "speed_category",
                "duration_category",
                "trip_duration_capped",
                "trip_duration_zscore",
            ],
            "Risk": [
                "High",
                "High",
                "High",
                "High",
                "High",
                "High",
                "High",
                "High",
                "High",
                "High",
            ],
            "Reason": [
                "Trip end time is not available before trip completion",
                "Original target column",
                "Final transformed target, not an input feature",
                "Directly created from target",
                "Directly created from target",
                "Uses trip duration to calculate speed",
                "Created using speed",
                "Created from target duration",
                "Created from target outlier treatment",
                "Created from target distribution",
            ],
            "Action": [
                "Remove",
                "Use only as original target",
                "Use only as final target",
                "Remove",
                "Remove",
                "Remove",
                "Remove",
                "Remove",
                "Remove",
                "Remove",
            ],
        }
    )

    st.dataframe(
        styled_table(leakage_info),
        use_container_width=True,
        height=330,
    )

    l1, l2, l3 = st.columns(3)

    with l1:
        metric_card("Leakage Columns", "10", "Must be excluded")

    with l2:
        metric_card("Recommended Target", "trip_duration_log", "Use for modeling")

    with l3:
        metric_card("Model Status", "Feature Ready", "Notebook handles ML pipeline")

    insight_card(
        "Leakage Inference",
        "Before model building, remove columns created from trip_duration or future trip information. Otherwise, the model may learn answers directly and show fake high performance.",
    )
# ============================================================
# TAB 8 - INSIGHTS
# ============================================================

with tab8:
    chart_header(
        "Dashboard Inference",
        "Short conclusions from EDA and feature engineering.",
    )

    c1, c2 = st.columns(2)

    with c1:
        insight_card(
            "Time affects duration",
            "Trip duration changes by pickup hour. Rush-hour and time-slot features are useful.",
        )

        insight_card(
            "Distance is important",
            "Trip distance explains duration better than raw latitude and longitude.",
        )

        insight_card(
            "Outliers need review",
            "Very long trips may be real trips, traffic delays, or unusual records.",
        )

    with c2:
        insight_card(
            "Passenger count is weak alone",
            "Passenger count does not strongly explain trip duration by itself.",
        )

        insight_card(
            "Location matters",
            "Pickup and dropoff areas are concentrated, so location-based features are useful.",
        )

        insight_card(
            "Modeling direction",
            "Use datetime, distance, airport, location, interaction features, encoding, scaling, and leakage checks before modeling.",
        )

    summary_card(
        "Final Inference",
        "Trip duration is mainly affected by time, distance, location, airport movement, and traffic-related patterns.",
    )

    st.dataframe(
        styled_table(filtered_df.head(500)),
        use_container_width=True,
        height=380,
    )
