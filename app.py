from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Seoul Bike Demand Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "SeoulBikeData.csv"
MODEL_PATH = BASE / "WheelWise_Predictor.joblib"
RESULTS_PATH = BASE / "model_results.csv"
IMPORTANCE_PATH = BASE / "feature_importance.csv"

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #F8FCFE 0%, #EDF7FA 100%);
        color: #17324D;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #073B4C 0%, #0B6E69 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    h1, h2, h3, h4, p {
        color: #17324D;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #073B4C, #0B7A75);
        color: white;
        box-shadow: 0 14px 36px rgba(7, 59, 76, .18);
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: white !important;
        margin: 0;
    }

    .hero p {
        color: #DFF7F5 !important;
        margin: .45rem 0 0;
    }

    .card {
        background: white;
        border: 1px solid #D9EAF0;
        border-radius: 18px;
        padding: 1rem;
        min-height: 120px;
        box-shadow: 0 8px 24px rgba(23, 50, 77, .08);
    }

    .label {
        color: #607D8B;
        font-size: .8rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .value {
        color: #0B6E99;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: .25rem;
    }

    .note {
        color: #637A86;
        font-size: .85rem;
    }

    .panel {
        background: white;
        border: 1px solid #D9EAF0;
        border-radius: 20px;
        padding: 1.2rem;
        box-shadow: 0 8px 24px rgba(23, 50, 77, .07);
    }

    .low, .mid, .high {
        border-radius: 14px;
        padding: .9rem 1rem;
        font-weight: 700;
    }

    .low {
        background: #E8F7F1;
        color: #146C43;
        border: 1px solid #B9E5D1;
    }

    .mid {
        background: #FFF5DF;
        color: #8A5A00;
        border: 1px solid #F3D89A;
    }

    .high {
        background: #FDEBEC;
        color: #A53A43;
        border: 1px solid #F2BFC3;
    }

    /* Real Streamlit containers: no fake opening/closing HTML wrappers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, .94);
        border: 1px solid #D9EAF0 !important;
        border-radius: 20px !important;
        box-shadow: 0 9px 26px rgba(23, 50, 77, .07);
    }

    /* Make all input labels and values readable */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    div[data-testid="stSlider"] p,
    div[data-testid="stSlider"] span,
    div[data-testid="stDateInput"] p,
    div[data-testid="stTimeInput"] p {
        color: #17324D !important;
        opacity: 1 !important;
        font-weight: 600;
    }

    [data-baseweb="input"] input,
    [data-baseweb="select"] *,
    [data-baseweb="base-input"] input {
        color: #17324D !important;
        opacity: 1 !important;
    }

    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-baseweb="base-input"] {
        background: #FFFFFF !important;
        border-color: #BFD7E1 !important;
    }

    div[data-testid="stSlider"] [data-testid="stTickBar"] {
        color: #607D8B !important;
    }

    div[data-testid="stSlider"] [role="slider"] {
        background: #0B8F8A !important;
    }

    .stButton > button {
        width: 100%;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(90deg, #0B6E99, #13A89E);
        color: white !important;
        font-weight: 800;
        padding: .78rem 1rem;
        box-shadow: 0 6px 16px rgba(11, 110, 153, .20);
    }

    .stButton > button:hover {
        border: 0;
        color: white !important;
        transform: translateY(-1px);
    }

    .helper-row {
        display: flex;
        justify-content: space-between;
        color: #607D8B;
        font-size: .76rem;
        margin-top: -.35rem;
        margin-bottom: .75rem;
    }

    .slider-labels {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        margin-top: -.45rem;
        margin-bottom: .95rem;
        padding: 0 .15rem;
        color: #6B7F8A;
        font-size: .69rem;
        font-weight: 600;
        line-height: 1.15;
    }

    .slider-labels span {
        text-align: center;
        white-space: nowrap;
    }

    .slider-labels span:first-child {
        text-align: left;
    }

    .slider-labels span:last-child {
        text-align: right;
    }

    @media (max-width: 900px) {
        .slider-labels {
            font-size: .62rem;
        }
    }

    .auto-season {
        background: #E9F8F6;
        border: 1px solid #BCE8E2;
        color: #12645E;
        border-radius: 12px;
        padding: .8rem 1rem;
        font-weight: 700;
        margin: .25rem 0 .75rem;
    }

    .result-header {
        background: linear-gradient(90deg, #073B4C, #13A89E);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .result-header h2 {
        color: white !important;
        margin: 0;
    }

    .result-header p {
        color: #DDF7F4 !important;
        margin: .25rem 0 0;
    }

    .result-stat {
        background: #F7FCFD;
        border: 1px solid #CFE5EB;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 105px;
    }

    .result-stat-label {
        color: #607D8B;
        font-size: .78rem;
        text-transform: uppercase;
        font-weight: 700;
    }

    .result-stat-value {
        color: #0B6E99;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: .25rem;
    }

    .performance-card {
        background: #FFFFFF;
        border: 1px solid #CFE2E9;
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
        min-height: 118px;
        box-shadow: 0 8px 22px rgba(23, 50, 77, .08);
        margin-bottom: .7rem;
    }

    .performance-label {
        color: #607D8B;
        font-size: .8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .03em;
    }

    .performance-value {
        color: #0B6E99;
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: .25rem;
    }

    .performance-note {
        color: #6B818C;
        font-size: .82rem;
        margin-top: .2rem;
    }

    /* Sidebar navigation visibility */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"]:has(+ div[role="radiogroup"]) {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"] label span,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #FFFFFF !important;
        opacity: 1 !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: .18rem 0;
    }

    /* Streamlit metric visibility */
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span {
        color: #526E7A !important;
        opacity: 1 !important;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] {
        color: #0B6E99 !important;
        opacity: 1 !important;
        font-weight: 800;
    }

    [data-testid="stMetricDelta"] {
        color: #45606B !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(DATA_PATH, encoding="latin1")
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    return data[data["Functioning Day"].eq("Yes")].copy()

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def add_features(frame):
    result = frame.copy()
    result["peak_hour"] = result["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    result["comfort_index"] = result["temperature_c"] - 0.05 * result["humidity_pct"]
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    return result


def hour_label(hour):
    return f"{hour % 12 or 12}:00 {'AM' if hour < 12 else 'PM'}"


def season_from_date(selected_date):
    month = selected_date.month
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    return "Winter"


def seasonal_temperature_bounds(data, season):
    season_values = data.loc[data["Seasons"].eq(season), "Temperature(°C)"].dropna()
    if season_values.empty:
        return -20.0, 40.0, 15.0
    lower = float(np.floor(season_values.min()))
    upper = float(np.ceil(season_values.max()))
    typical = float(season_values.median())
    return lower, upper, typical


def demand_band(value, data):
    lower, upper = data["Rented Bike Count"].quantile([0.33, 0.67])
    if value < lower:
        return "Low", "low", "Normal availability should be sufficient."
    if value < upper:
        return "Moderate", "mid", "Review bicycle availability before this hour."
    return "High", "high", "Prepare extra bicycles and redistribution support."

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run the notebook first.")
    st.stop()

data = load_data()
bundle = load_bundle()
model = bundle["model"]

st.sidebar.markdown("## WheelWise")
st.sidebar.caption("Hourly bike-demand predictor")
page = st.sidebar.radio(
    "",
    ["Overview", "Demand Explorer", "Bike Predictor", "Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Predictions apply only to normal functioning periods.")

if page == "Overview":
    st.markdown(
        '<div class="hero"><h1>Seoul Bike Demand Predictor</h1><p>Explore hourly rental behaviour and prepare bicycles before demand arrives.</p></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    cards = [
        ("Operating Records", f"{len(data):,}", "Functioning hourly records"),
        ("Average Demand", f"{data['Rented Bike Count'].mean():,.0f}", "Bikes rented per hour"),
        ("Busiest Hour", hour_label(int(data.groupby("Hour")["Rented Bike Count"].mean().idxmax())), "Highest average demand"),
        ("Best Test R²", f"{bundle['metrics']['R2']:.3f}", "Variation explained"),
    ]
    for column, (label, value, note) in zip(columns, cards):
        column.markdown(
            f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Average demand throughout the day")
        hourly = data.groupby("Hour", as_index=False)["Rented Bike Count"].mean()
        figure = px.line(hourly, x="Hour", y="Rented Bike Count", markers=True)
        figure.update_layout(height=410, hovermode="x unified")
        st.plotly_chart(figure, use_container_width=True)
    with right:
        st.subheader("Seasonal demand")
        seasonal = data.groupby("Seasons", as_index=False)["Rented Bike Count"].mean()
        figure = px.bar(seasonal, x="Seasons", y="Rented Bike Count")
        figure.update_layout(height=410, showlegend=False)
        st.plotly_chart(figure, use_container_width=True)

elif page == "Demand Explorer":
    st.markdown(
        '<div class="hero"><h1>Interactive Demand Explorer</h1>'
        '<p>Compare demand by time, season, temperature, and rain conditions using clearly separated views.</p></div>',
        unsafe_allow_html=True,
    )

    filter_left, filter_right = st.columns(2, gap="large")

    with filter_left:
        seasons = st.multiselect(
            "Select season",
            sorted(data["Seasons"].unique()),
            default=sorted(data["Seasons"].unique()),
            help="Choose one or more seasons to compare.",
        )

    with filter_right:
        hour_range = st.slider(
            "Select hour range",
            min_value=0,
            max_value=23,
            value=(0, 23),
            help="Limit all charts to a specific period of the day.",
        )

    filtered = data[
        data["Seasons"].isin(seasons)
        & data["Hour"].between(hour_range[0], hour_range[1])
    ].copy()

    if filtered.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    summary_1, summary_2, summary_3 = st.columns(3)
    summary_1.metric("Filtered records", f"{len(filtered):,}")
    summary_2.metric("Average hourly rentals", f"{filtered['Rented Bike Count'].mean():,.0f}")
    summary_3.metric("Highest hourly rentals", f"{filtered['Rented Bike Count'].max():,.0f}")

    tab1, tab2 = st.tabs([
        "Demand by Hour",
        "Rain Impact",
    ])

    with tab1:
        st.subheader("How demand changes throughout the day")
        st.caption(
            "Each line shows the average number of bicycles rented during each hour for the selected seasons."
        )
        hourly = (
            filtered.groupby(["Hour", "Seasons"], as_index=False)["Rented Bike Count"]
            .mean()
            .rename(columns={"Rented Bike Count": "Average Rentals"})
        )
        figure = px.line(
            hourly,
            x="Hour",
            y="Average Rentals",
            color="Seasons",
            markers=True,
            labels={
                "Hour": "Hour of day",
                "Seasons": "Season",
            },
        )
        figure.update_layout(
            height=470,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(figure, use_container_width=True)


    with tab2:
        st.subheader("Demand during rain and no-rain hours")
        st.caption(
            "Rainfall is used only in this view. It compares normal dry hours with hours where measurable rain was recorded."
        )
        rain_view = filtered.copy()
        rain_view["Rain Condition"] = np.where(
            rain_view["Rainfall(mm)"] > 0,
            "Rain",
            "No Rain",
        )
        rain_summary = (
            rain_view.groupby(["Rain Condition", "Seasons"], as_index=False)["Rented Bike Count"]
            .mean()
            .rename(columns={"Rented Bike Count": "Average Rentals"})
        )
        figure = px.bar(
            rain_summary,
            x="Rain Condition",
            y="Average Rentals",
            color="Seasons",
            barmode="group",
            labels={
                "Rain Condition": "Weather condition",
                "Seasons": "Season",
            },
        )
        figure.update_layout(
            height=470,
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(figure, use_container_width=True)


elif page == "Bike Predictor":
    st.markdown(
        '<div class="hero"><h1>Hourly Bike Demand Predictor</h1>'
        '<p>Select a historical date, time, and weather conditions to estimate hourly bicycle rentals.</p></div>',
        unsafe_allow_html=True,
    )

    ranges = bundle["manual_ranges"]
    minimum_date = data["Date"].min().date()
    maximum_date = data["Date"].max().date()
    default_date = min(
        max(pd.Timestamp("2018-07-15").date(), minimum_date),
        maximum_date,
    )

    with st.container(border=True):
        st.subheader("Prediction inputs")
        st.caption(
            f"Available dataset period: {minimum_date.strftime('%d %b %Y')} "
            f"to {maximum_date.strftime('%d %b %Y')}."
        )

        date_col, time_col = st.columns(2, gap="large")

        with date_col:
            selected_date = st.date_input(
                "Prediction date",
                value=default_date,
                min_value=minimum_date,
                max_value=maximum_date,
                help="Only dates covered by the dataset can be selected.",
            )

        with time_col:
            selected_time = st.time_input(
                "Prediction time",
                value=pd.Timestamp("08:00").time(),
                step=3600,
                help="The model predicts demand for one complete hourly period.",
            )

        season = season_from_date(selected_date)
        temperature_min, temperature_max, temperature_default = seasonal_temperature_bounds(
            data,
            season,
        )

        st.markdown(
            f'<div class="auto-season">📅 Detected Seoul season: {season}</div>',
            unsafe_allow_html=True,
        )

        first, second = st.columns(2, gap="large")

        with first:
            temperature = st.slider(
                f"Temperature during {season} (°C)",
                min_value=temperature_min,
                max_value=temperature_max,
                value=float(
                    np.clip(
                        temperature_default,
                        temperature_min,
                        temperature_max,
                    )
                ),
                step=0.5,
                key=f"temperature_{season}",
                help=(
                    f"Observed {season} range in the dataset: "
                    f"{temperature_min:.0f}°C to {temperature_max:.0f}°C."
                ),
            )
            st.markdown(
                '<div class="helper-row"><span>Colder</span><span>Warmer</span></div>',
                unsafe_allow_html=True,
            )

            humidity = st.slider(
                "Humidity (%)",
                min_value=int(ranges["humidity_pct"]["min"]),
                max_value=int(ranges["humidity_pct"]["max"]),
                value=55,
                step=1,
                help="Lower values mean drier air; higher values mean more humid air.",
            )
            st.markdown(
                '<div class="helper-row"><span>Dry air</span><span>Humid / damp air</span></div>',
                unsafe_allow_html=True,
            )

        with second:
            solar = st.slider(
                "Solar radiation (MJ/m²)",
                min_value=float(ranges["solar_radiation"]["min"]),
                max_value=float(ranges["solar_radiation"]["max"]),
                value=0.8,
                step=0.1,
                help="Sunlight energy recorded during the selected hour.",
            )
            st.markdown(
                '''
                <div class="slider-labels">
                    <span>Dark / night</span>
                    <span>Low light</span>
                    <span>Bright</span>
                    <span>Strong sunlight</span>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            rainfall = st.slider(
                "Rainfall (mm)",
                min_value=float(ranges["rainfall_mm"]["min"]),
                max_value=min(float(ranges["rainfall_mm"]["max"]), 35.0),
                value=0.0,
                step=0.5,
                help="Rainfall recorded during the selected hour.",
            )
            st.markdown(
                '''
                <div class="slider-labels">
                    <span>No rain</span>
                    <span>Light rain</span>
                    <span>Moderate rain</span>
                    <span>Heavy rain</span>
                </div>
                ''',
                unsafe_allow_html=True,
            )

        submitted = st.button(
            "Predict Bike Demand",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        hour = int(selected_time.hour)

        manual = pd.DataFrame([{
            "hour": hour,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "solar_radiation": solar,
            "rainfall_mm": rainfall,
            "season": season,
        }])

        model_input = add_features(manual)[bundle["model_features"]]
        prediction = max(0.0, float(model.predict(model_input)[0]))
        level, css_class, advice = demand_band(prediction, data)

        lower_cut, upper_cut = data["Rented Bike Count"].quantile([0.33, 0.67])
        maximum = max(2500, int(data["Rented Bike Count"].quantile(0.99)))

        with st.container(border=True):
            st.markdown(
                '<div class="result-header"><h2>Prediction Result</h2>'
                '<p>Estimated hourly bicycle demand under the selected conditions.</p></div>',
                unsafe_allow_html=True,
            )

            stat1, stat2, stat3 = st.columns(3, gap="large")

            stat1.markdown(
                f'<div class="result-stat">'
                f'<div class="result-stat-label">Predicted rentals</div>'
                f'<div class="result-stat-value">{prediction:,.0f} bikes</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            stat2.markdown(
                f'<div class="result-stat">'
                f'<div class="result-stat-label">Demand level</div>'
                f'<div class="result-stat-value">{level}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            stat3.markdown(
                f'<div class="result-stat">'
                f'<div class="result-stat-label">Selected period</div>'
                f'<div class="result-stat-value">{hour_label(hour)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                number={
                    "suffix": " bikes",
                    "font": {"size": 44, "color": "#17324D"},
                },
                delta={
                    "reference": float(data["Rented Bike Count"].mean()),
                    "relative": True,
                    "valueformat": ".0%",
                    "increasing": {"color": "#C97800"},
                    "decreasing": {"color": "#0B8F8A"},
                },
                title={
                    "text": (
                        f"Expected hourly demand<br>"
                        f"<span style='font-size:0.72em'>"
                        f"{selected_date.strftime('%d %b %Y')} at "
                        f"{hour_label(hour)} · {season}</span>"
                    ),
                    "font": {"color": "#17324D", "size": 20},
                },
                gauge={
                    "shape": "angular",
                    "axis": {
                        "range": [0, maximum],
                        "tickcolor": "#45606B",
                        "tickfont": {"color": "#45606B"},
                    },
                    "bar": {"color": "#13A89E", "thickness": 0.28},
                    "bgcolor": "#F2F8FA",
                    "bordercolor": "#BFD7E1",
                    "steps": [
                        {"range": [0, lower_cut], "color": "#DDF4ED"},
                        {"range": [lower_cut, upper_cut], "color": "#FBECC8"},
                        {"range": [upper_cut, maximum], "color": "#F7D9DD"},
                    ],
                    "threshold": {
                        "line": {"color": "#073B4C", "width": 4},
                        "thickness": 0.8,
                        "value": prediction,
                    },
                },
            ))

            gauge.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#17324D"},
                margin=dict(l=45, r=45, t=90, b=30),
            )

            st.plotly_chart(gauge, use_container_width=True)

            st.markdown(
                f'<div class="{css_class}">{level} demand — {advice}</div>',
                unsafe_allow_html=True,
            )



elif page == "Model Performance":
    st.markdown(
        '<div class="hero"><h1>Model Performance</h1>'
        '<p>Compare regression models and inspect the features driving predictions.</p></div>',
        unsafe_allow_html=True,
    )

    results = pd.read_csv(RESULTS_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    metric_1, metric_2, metric_3 = st.columns(3, gap="large")

    metric_1.markdown(
        f'<div class="performance-card">'
        f'<div class="performance-label">Test MAE</div>'
        f'<div class="performance-value">{bundle["metrics"]["MAE"]:.1f} bikes</div>'
        f'<div class="performance-note">Average absolute prediction error</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    metric_2.markdown(
        f'<div class="performance-card">'
        f'<div class="performance-label">Test RMSE</div>'
        f'<div class="performance-value">{bundle["metrics"]["RMSE"]:.1f} bikes</div>'
        f'<div class="performance-note">Error with larger mistakes weighted more</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    metric_3.markdown(
        f'<div class="performance-card">'
        f'<div class="performance-label">Test R²</div>'
        f'<div class="performance-value">{bundle["metrics"]["R2"]:.3f}</div>'
        f'<div class="performance-note">Variation explained on the test set</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        st.subheader("Model comparison")
        st.caption("Lower RMSE indicates better prediction performance.")
        comparison = results.sort_values("RMSE")
        figure = px.bar(
            comparison,
            x="Model",
            y="RMSE",
            text="RMSE",
            labels={"RMSE": "Root Mean Squared Error"},
        )
        figure.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        figure.update_layout(
            height=460,
            showlegend=False,
            margin=dict(l=10, r=10, t=25, b=10),
            yaxis_title="RMSE in bicycles",
        )
        st.plotly_chart(figure, use_container_width=True)

    with chart_right:
        st.subheader("Permutation feature importance")
        st.caption(
            "A longer bar means model accuracy falls more when that feature is shuffled."
        )
        shown = importance.sort_values("Importance Mean")
        figure = px.bar(
            shown,
            x="Importance Mean",
            y="Feature",
            orientation="h",
            labels={
                "Importance Mean": "Increase in prediction error",
                "Feature": "Feature",
            },
        )
        figure.update_layout(
            height=460,
            showlegend=False,
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(figure, use_container_width=True)
