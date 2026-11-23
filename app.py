
import json
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dashboard_data"
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

st.set_page_config(
    page_title="Karnataka Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DASHBOARD STYLING
# --------------------------------------------------

st.markdown("""
<style>
    /* Main application */
    .stApp {
        background-color: #F5F8FC;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #17365D;
        font-weight: 700;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E1EAF5;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(30, 70, 120, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #1976D2;
        font-weight: 700;
    }

    /* Charts and dataframes */
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1976D2;
        color: white;
        border-radius: 8px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1259A7;
        color: white;
        border: none;
    }

    /* Dividers */
    hr {
        border-color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    monthly_spatial = pd.read_csv(DATA_DIR / "spatial_monthly.csv")
    annual_spatial = pd.read_csv(DATA_DIR / "spatial_annual.csv")

    weekly = pd.read_csv(PROCESSED_DIR / "karnataka_weekly_rainfall.csv")
    predictions = pd.read_csv(RESULTS_DIR / "test_predictions.csv")

    with open(DATA_DIR / "karnataka.geojson", "r", encoding="utf-8") as f:
        boundary = json.load(f)

    with open(RESULTS_DIR / "evaluation_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return monthly_spatial, annual_spatial, weekly, predictions, boundary, metrics


try:
    monthly_spatial, annual_spatial, weekly, predictions, boundary, metrics = load_data()
except Exception as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.info("Check that the dashboard_data, processed and results folders contain the required files.")
    st.stop()


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def add_boundary(fig, geojson):
    """Draw Karnataka's GeoJSON outline on a Plotly geographic figure."""

    def draw_geometry(geometry):
        geometry_type = geometry["type"]
        coordinates = geometry["coordinates"]

        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            return

        for polygon in polygons:
            for ring in polygon:
                lons = [point[0] for point in ring]
                lats = [point[1] for point in ring]

                fig.add_trace(
                    go.Scattergeo(
                        lon=lons,
                        lat=lats,
                        mode="lines",
                        line=dict(color="black", width=1.5),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )

    if geojson["type"] == "FeatureCollection":
        for feature in geojson["features"]:
            if feature.get("geometry"):
                draw_geometry(feature["geometry"])

    elif geojson["type"] == "Feature":
        draw_geometry(geojson["geometry"])

    else:
        draw_geometry(geojson)


def get_metric(metric_name, default=None):
    """Read a metric from the saved evaluation JSON."""
    for section in metrics.values():
        if isinstance(section, dict) and metric_name in section:
            return section[metric_name]
    return default


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #1565C0, #42A5F5);
    padding: 22px 16px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
">
    <div style="font-size: 30px;">🌧️</div>
    <div style="font-size: 20px; font-weight: 700;">
        Karnataka Rainfall
    </div>
    <div style="font-size: 12px; opacity: 0.9;">
        Deep Learning Analytics
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard Overview",
        "Rainfall Distribution Map",
        "Rainfall Trends",
        "Model Performance",
        "Project Information"
    ]
)

st.sidebar.divider()
st.sidebar.caption("IMD rainfall data | 1991–2020")


# --------------------------------------------------
# 5. DASHBOARD OVERVIEW
# --------------------------------------------------


if page == "Dashboard Overview":

    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #1565C0, #42A5F5);
        padding: 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
    ">
        <div style="font-size: 14px; opacity: 0.9;">
            DEEP LEARNING • RAINFALL ANALYTICS
        </div>
        <h1 style="color:white; margin:10px 0;">
            Karnataka Rainfall Prediction
        </h1>
        <p style="font-size:17px; margin:0;">
            Weekly rainfall forecasting using a Hybrid CNN-BiLSTM
            Deep Learning Architecture
        </p>
        <p style="font-size:13px; margin-top:12px; opacity:0.9;">
            Historical IMD Data | 1991–2020
        </p>
    </div>
    """, unsafe_allow_html=True)

    weekly["Week_Ending_Sunday"] = pd.to_datetime(
        weekly["Week_Ending_Sunday"]
    )

    total_weeks = len(weekly)
    avg_rainfall = weekly["Weekly_Rainfall_mm"].mean()
    max_rainfall = weekly["Weekly_Rainfall_mm"].max()
    total_rainfall = weekly["Weekly_Rainfall_mm"].sum()

    st.subheader("Rainfall at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Weekly Observations", f"{total_weeks:,}")
    col2.metric("Average Weekly Rainfall", f"{avg_rainfall:.2f} mm")
    col3.metric("Maximum Weekly Rainfall", f"{max_rainfall:.2f} mm")
    col4.metric("Total Recorded Rainfall", f"{total_rainfall:,.0f} mm")

    st.divider()

    st.subheader("Historical Weekly Rainfall")
    st.caption("State-wide weekly rainfall from January 1991 to December 2020.")

    fig = px.area(
        weekly,
        x="Week_Ending_Sunday",
        y="Weekly_Rainfall_mm",
        labels={
            "Week_Ending_Sunday": "Week ending",
            "Weekly_Rainfall_mm": "Rainfall (mm)"
        }
    )

    fig.update_traces(
        line=dict(color="#1976D2", width=1.2),
        fillcolor="rgba(25,118,210,0.16)"
    )

    fig.update_layout(
        height=430,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=25, b=20),
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#E8EEF5")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("About This Project")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("""
        This project explores historical rainfall patterns across
        Karnataka and develops a deep learning model for weekly
        rainfall prediction.

        **Main objectives**
        - Analyse historical rainfall distribution.
        - Explore weekly, monthly and annual rainfall trends.
        - Develop a hybrid CNN-BiLSTM forecasting model.
        - Evaluate predictions against a persistence baseline.
        """)

    with col2:
        st.markdown("""
        <div style="
            background:#EAF3FF;
            padding:20px;
            border-radius:12px;
            border-left:4px solid #1976D2;
        ">
            <h4 style="color:#174A7E;">Research Details</h4>
            <p><b>Research thrust:</b><br>Sensor Data Analytics</p>
            <p><b>Dataset:</b><br>IMD 0.25° Daily Gridded Rainfall</p>
            <p><b>Model:</b><br>Hybrid CNN-BiLSTM</p>
            <p><b>Forecast:</b><br>State-wide weekly rainfall</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "The historical rainfall map displays spatial IMD observations. "
        "The CNN-BiLSTM produces state-wide weekly forecasts, not "
        "district-level or grid-level predictions."
    )



# --------------------------------------------------
# 6. SPATIAL RAINFALL MAP
# --------------------------------------------------


elif page == "Rainfall Distribution Map":

    st.title("Karnataka Rainfall Distribution")
    st.caption(
        "Explore the spatial distribution of historical IMD rainfall "
        "across 252 grid-cell centres in Karnataka."
    )

    st.divider()

    # Map selection
    col1, col2 = st.columns([1, 2])

    with col1:
        map_type = st.selectbox(
            "Rainfall analysis",
            ["Monthly rainfall", "Annual rainfall"]
        )

    if map_type == "Monthly rainfall":

        available_years = sorted(monthly_spatial["year"].unique())

        with col2:
            selected_year = st.selectbox(
                "Select year",
                available_years,
                index=len(available_years) - 1
            )

        month_names = {
            1: "January", 2: "February", 3: "March",
            4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September",
            10: "October", 11: "November", 12: "December"
        }

        selected_month = st.select_slider(
            "Select month",
            options=list(month_names.keys()),
            value=7,
            format_func=lambda x: month_names[x]
        )

        map_data = monthly_spatial[
            (monthly_spatial["year"] == selected_year) &
            (monthly_spatial["month"] == selected_month)
        ].copy()

        map_title = (
            f"{month_names[selected_month]} {selected_year}"
        )

    else:

        available_years = sorted(annual_spatial["year"].unique())

        with col2:
            selected_year = st.selectbox(
                "Select year",
                available_years,
                index=len(available_years) - 1
            )

        map_data = annual_spatial[
            annual_spatial["year"] == selected_year
        ].copy()

        map_title = f"Annual Rainfall — {selected_year}"

    if map_data.empty:
        st.warning("No rainfall data is available for this selection.")

    else:

        # Summary metrics
        avg = map_data["rainfall_mm"].mean()
        maximum = map_data["rainfall_mm"].max()
        minimum = map_data["rainfall_mm"].min()

        col1, col2, col3 = st.columns(3)

        col1.metric("Average Grid Rainfall", f"{avg:.2f} mm")
        col2.metric("Maximum Grid Rainfall", f"{maximum:.2f} mm")
        col3.metric("Minimum Grid Rainfall", f"{minimum:.2f} mm")

        st.divider()

        st.subheader(f"Spatial Rainfall Map — {map_title}")

        fig = px.scatter_geo(
            map_data,
            lat="latitude",
            lon="longitude",
            color="rainfall_mm",
            size="rainfall_mm",
            size_max=19,
            color_continuous_scale=[
                [0.0, "#EFF6FF"],
                [0.25, "#BFDBFE"],
                [0.5, "#60A5FA"],
                [0.75, "#2563EB"],
                [1.0, "#082F75"]
            ],
            hover_name="latitude",
            hover_data={
                "latitude": ":.3f",
                "longitude": ":.3f",
                "rainfall_mm": ":.2f",
                "valid_days": True
            },
            labels={
                "rainfall_mm": "Rainfall (mm)",
                "latitude": "Latitude",
                "longitude": "Longitude",
                "valid_days": "Valid days"
            }
        )

        add_boundary(fig, boundary)

        fig.update_geos(
            projection_type="mercator",
            lonaxis_range=[73.8, 78.8],
            lataxis_range=[11.3, 18.8],
            showcountries=True,
            showland=True,
            showocean=True,
            landcolor="#F8FAFC",
            oceancolor="#EAF3FF",
            showcoastlines=True,
            coastlinecolor="#94A3B8"
        )

        fig.update_layout(
            height=650,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="white",
            coloraxis_colorbar=dict(
                title="Rainfall<br>(mm)",
                thickness=18
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Each circle represents one IMD grid-cell centre. "
            "Colour indicates the rainfall total for the selected "
            "month or year. Circle sizes also reflect rainfall."
        )

        with st.expander("View grid-cell rainfall data"):
            display_data = map_data[
                ["latitude", "longitude", "rainfall_mm", "valid_days"]
            ].sort_values("rainfall_mm", ascending=False)

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )



# --------------------------------------------------
# 7. RAINFALL TRENDS
# --------------------------------------------------


elif page == "Rainfall Trends":

    st.title("Rainfall Trends & Seasonal Analysis")
    st.caption(
        "Explore weekly, monthly and annual rainfall patterns "
        "across Karnataka from 1991 to 2020."
    )

    trend_frequency = st.radio(
        "Select analysis period",
        ["Weekly", "Monthly", "Annual"],
        horizontal=True
    )

    trend_data = weekly.copy()
    trend_data["Week_Ending_Sunday"] = pd.to_datetime(
        trend_data["Week_Ending_Sunday"]
    )

    if trend_frequency == "Weekly":

        trend_data = trend_data.rename(columns={
            "Week_Ending_Sunday": "Period",
            "Weekly_Rainfall_mm": "Rainfall"
        })

    elif trend_frequency == "Monthly":

        trend_data["Period"] = (
            trend_data["Week_Ending_Sunday"].dt.to_period("M")
        )

        trend_data = (
            trend_data.groupby("Period")["Weekly_Rainfall_mm"]
            .sum()
            .reset_index()
        )

        trend_data["Period"] = trend_data["Period"].dt.to_timestamp()
        trend_data = trend_data.rename(
            columns={"Weekly_Rainfall_mm": "Rainfall"}
        )

    else:

        trend_data["Period"] = (
            trend_data["Week_Ending_Sunday"].dt.year
        )

        trend_data = (
            trend_data.groupby("Period")["Weekly_Rainfall_mm"]
            .sum()
            .reset_index()
        )

        trend_data = trend_data.rename(
            columns={"Weekly_Rainfall_mm": "Rainfall"}
        )

    # Year selection
    trend_data["Year"] = (
        pd.to_datetime(trend_data["Period"]).dt.year
        if trend_frequency != "Annual"
        else trend_data["Period"]
    )

    years = sorted(trend_data["Year"].unique())

    col1, col2 = st.columns(2)

    with col1:
        start_year = st.selectbox(
            "Start year",
            years,
            index=0,
            key="trend_start"
        )

    with col2:
        end_year = st.selectbox(
            "End year",
            years,
            index=len(years) - 1,
            key="trend_end"
        )

    if start_year > end_year:
        st.error("Start year must not be later than end year.")
        st.stop()

    filtered = trend_data[
        (trend_data["Year"] >= start_year) &
        (trend_data["Year"] <= end_year)
    ].copy()

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Rainfall",
        f"{filtered['Rainfall'].mean():.2f} mm"
    )

    col2.metric(
        "Maximum Rainfall",
        f"{filtered['Rainfall'].max():.2f} mm"
    )

    col3.metric(
        "Total Rainfall",
        f"{filtered['Rainfall'].sum():,.2f} mm"
    )

    st.divider()

    st.subheader(f"{trend_frequency} Rainfall Trend")

    fig = px.line(
        filtered,
        x="Period",
        y="Rainfall",
        markers=(trend_frequency == "Annual"),
        labels={
            "Period": trend_frequency,
            "Rainfall": "Rainfall (mm)"
        }
    )

    fig.update_traces(
        line=dict(color="#1976D2", width=2),
        marker=dict(size=7)
    )

    fig.update_layout(
        height=430,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#E8EEF5")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Monthly seasonal analysis
    st.divider()
    st.subheader("Monthly Rainfall Seasonality")

    weekly_monthly = weekly.copy()
    weekly_monthly["Week_Ending_Sunday"] = pd.to_datetime(
        weekly_monthly["Week_Ending_Sunday"]
    )

    weekly_monthly["Month"] = (
        weekly_monthly["Week_Ending_Sunday"].dt.month
    )

    seasonal = (
        weekly_monthly.groupby("Month")["Weekly_Rainfall_mm"]
        .mean()
        .reindex(range(1, 13))
        .reset_index()
    )

    seasonal["Month_Name"] = seasonal["Month"].apply(
        lambda x: pd.Timestamp(2000, x, 1).strftime("%B")
    )

    seasonal_fig = px.bar(
        seasonal,
        x="Month_Name",
        y="Weekly_Rainfall_mm",
        labels={
            "Month_Name": "Month",
            "Weekly_Rainfall_mm": "Average weekly rainfall (mm)"
        }
    )

    seasonal_fig.update_traces(
        marker_color="#1976D2"
    )

    seasonal_fig.update_layout(
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#E8EEF5")
    )

    st.plotly_chart(
        seasonal_fig,
        use_container_width=True
    )

    st.caption(
        "Seasonality is calculated as the average weekly rainfall "
        "for each calendar month across the study period."
    )



# --------------------------------------------------
# 8. MODEL PERFORMANCE
# --------------------------------------------------


elif page == "Model Performance":
    st.title("CNN-BiLSTM Model Performance")
    st.write(
        "Evaluation of the hybrid deep learning model on unseen "
        "weekly rainfall observations (2016–2020)."
    )

    # Load evaluation metrics
    cnn_metrics = metrics["CNN_BiLSTM"]
    baseline_metrics = metrics["Persistence_Baseline"]

    cnn_mae = cnn_metrics["MAE_mm"]
    cnn_rmse = cnn_metrics["RMSE_mm"]
    cnn_r2 = cnn_metrics["R2"]

    baseline_mae = baseline_metrics["MAE_mm"]
    baseline_rmse = baseline_metrics["RMSE_mm"]
    baseline_r2 = baseline_metrics["R2"]

    # Key performance indicators
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{cnn_mae:.2f} mm")
    c2.metric("RMSE", f"{cnn_rmse:.2f} mm")
    c3.metric("R² Score", f"{cnn_r2:.3f}")

    st.divider()

    # Actual vs predicted rainfall
    st.subheader("Actual vs Predicted Rainfall")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predictions["Week_Ending"],
        y=predictions["Actual_Rainfall_mm"],
        mode="lines",
        name="Actual rainfall"
    ))
    fig.add_trace(go.Scatter(
        x=predictions["Week_Ending"],
        y=predictions["Predicted_Rainfall_mm"],
        mode="lines",
        name="CNN-BiLSTM prediction"
    ))
    fig.update_layout(
        title="Weekly Actual and Predicted Rainfall",
        xaxis_title="Week ending",
        yaxis_title="Rainfall (mm)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model comparison
    st.subheader("Comparison with Persistence Baseline")

    comparison = pd.DataFrame({
        "Metric": ["MAE (mm)", "RMSE (mm)", "R²"],
        "CNN-BiLSTM": [cnn_mae, cnn_rmse, cnn_r2],
        "Persistence Baseline": [
            baseline_mae, baseline_rmse, baseline_r2
        ]
    })

    st.dataframe(comparison, hide_index=True, use_container_width=True)

    mae_improvement = (
        (baseline_mae - cnn_mae) / baseline_mae
    ) * 100
    rmse_improvement = (
        (baseline_rmse - cnn_rmse) / baseline_rmse
    ) * 100

    st.success(
        f"The CNN-BiLSTM reduced MAE by {mae_improvement:.1f}% "
        f"and RMSE by {rmse_improvement:.1f}% compared with "
        "the persistence baseline on the test set."
    )

    # Prediction scatter plot
    st.subheader("Prediction Accuracy")

    scatter_fig = px.scatter(
        predictions,
        x="Actual_Rainfall_mm",
        y="Predicted_Rainfall_mm",
        title="Actual vs Predicted Weekly Rainfall",
        labels={
            "Actual_Rainfall_mm": "Actual rainfall (mm)",
            "Predicted_Rainfall_mm": "Predicted rainfall (mm)"
        },
        opacity=0.65
    )

    max_value = max(
        predictions["Actual_Rainfall_mm"].max(),
        predictions["Predicted_Rainfall_mm"].max()
    )

    scatter_fig.add_trace(go.Scatter(
        x=[0, max_value],
        y=[0, max_value],
        mode="lines",
        name="Ideal prediction",
        line=dict(dash="dash")
    ))
    scatter_fig.update_layout(template="plotly_white")
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Residual analysis
    st.subheader("Prediction Error Distribution")

    residuals = (
        predictions["Actual_Rainfall_mm"]
        - predictions["Predicted_Rainfall_mm"]
    )

    residual_fig = px.histogram(
        x=residuals,
        nbins=30,
        title="Distribution of Prediction Errors",
        labels={"x": "Residual (Actual − Predicted), mm"},
        opacity=0.8
    )
    residual_fig.update_layout(template="plotly_white")
    st.plotly_chart(residual_fig, use_container_width=True)

    # Interpretation and limitations
    st.subheader("Interpretation")

    st.markdown(
        f"""
        - The model achieved a test MAE of **{cnn_mae:.2f} mm**
          and RMSE of **{cnn_rmse:.2f} mm**.
        - Its R² score was **{cnn_r2:.3f}**, indicating that it
          explains a portion of the variation in test rainfall.
        - Compared with the persistence baseline, the model
          improved MAE by **{mae_improvement:.1f}%**.
        - RMSE improvement was **{rmse_improvement:.1f}%**.
        """
    )

    st.warning(
        "Limitation: The model tends to underestimate extreme rainfall "
        "events. It uses historical state-wide rainfall only and predicts "
        "one state-wide weekly value. It does not provide district-level "
        "or grid-level forecasts."
    )

    st.caption(
        "Test period: July 2016 – December 2020. "
        "Metrics are calculated on the held-out test set."
    )


# --------------------------------------------------
# 9. PROJECT INFORMATION
# --------------------------------------------------



elif page == "Project Information":

    st.markdown("""
    <style>
    .project-hero {
        background: linear-gradient(120deg, #123B70, #1976D2);
        padding: 38px 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }
    .project-hero h1 {
        color: white !important;
        font-size: 31px !important;
        line-height: 1.3;
        margin-bottom: 12px;
    }
    .project-hero p {
        color: #EAF3FF;
        font-size: 16px;
        line-height: 1.7;
        margin-bottom: 0;
    }
    .eyebrow {
        color: #BBD9FF;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .info-card {
        background: white;
        border: 1px solid #E1EAF4;
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        margin-bottom: 12px;
    }
    .info-card h4 {
        color: #17365D;
        margin: 0 0 9px 0;
        font-size: 17px;
    }
    .info-card p {
        color: #52657A;
        font-size: 14px;
        line-height: 1.65;
        margin: 0;
    }
    .section-kicker {
        color: #1976D2;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .section-heading {
        color: #17365D;
        font-size: 25px;
        font-weight: 750;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .architecture-step {
        background: #F1F6FC;
        border: 1px solid #DCE8F5;
        border-radius: 11px;
        padding: 13px 10px;
        text-align: center;
        color: #17365D;
        font-size: 13px;
        font-weight: 650;
        min-height: 65px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .architecture-arrow {
        text-align: center;
        color: #1976D2;
        font-size: 23px;
        font-weight: bold;
        padding: 2px;
    }
    .sdg-card {
        background: #FFFFFF;
        border: 1px solid #E1EAF4;
        border-left: 5px solid #1976D2;
        border-radius: 12px;
        padding: 17px;
        min-height: 125px;
        margin-bottom: 12px;
    }
    .sdg-card h4 {
        color: #17365D;
        font-size: 16px;
        margin: 0 0 7px 0;
    }
    .sdg-card p {
        color: #52657A;
        font-size: 13px;
        line-height: 1.5;
        margin: 0;
    }
    .method-number {
        display: inline-block;
        background: #E6F1FF;
        color: #1768B2;
        border-radius: 9px;
        padding: 5px 10px;
        font-weight: 750;
        margin-bottom: 8px;
    }
    .closing-panel {
        background: #EAF3FF;
        border: 1px solid #CFE2F8;
        padding: 22px;
        border-radius: 14px;
        color: #17365D;
        line-height: 1.7;
    }
    </style>

    <div class="project-hero">
        <div class="eyebrow">Neural Networks & Deep Learning • Research Project</div>
        <h1>Karnataka Rainfall Prediction Using Hybrid CNN-BiLSTM Deep Learning Architecture</h1>
        <p>
            A data-driven approach to understanding historical rainfall
            patterns and forecasting weekly state-wide rainfall across
            Karnataka, India.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Project identity
    st.markdown('<div class="section-kicker">01 / Project identity</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Research at a glance</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Study period", "1991–2020")
    k2.metric("Weekly observations", "1,564")
    k3.metric("Spatial grid cells", "252")
    k4.metric("Forecast horizon", "1 week")

    st.markdown("""
    <div class="info-card">
        <h4>Project abstract</h4>
        <p>
        Rainfall prediction is important for agricultural planning,
        water resource management and climate-related risk assessment.
        This project investigates a hybrid deep learning approach
        combining Convolutional Neural Networks (CNN) and Bidirectional
        Long Short-Term Memory (BiLSTM) networks to forecast weekly
        rainfall across Karnataka. Daily rainfall observations from
        the India Meteorological Department's 0.25° gridded dataset
        for 1991–2020 are spatially aggregated using latitude-based
        area weighting and converted into complete Monday–Sunday
        weekly totals. A 12-week historical input sequence is used
        to train the hybrid model. The model is evaluated on a
        chronological held-out test set and compared with a
        persistence baseline using Mean Absolute Error, Root Mean
        Squared Error and the coefficient of determination. The
        resulting application combines historical spatial rainfall
        visualisation, temporal trend exploration and model
        evaluation in an interactive dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Research details
    st.markdown('<div class="section-kicker">02 / Research profile</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Project information</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h4>Academic details</h4>
            <p>
            <b>Course:</b> MDA507-5 Neural Networks and Deep Learning<br><br>
            <b>Research thrust:</b> Sensor Data Analytics<br><br>
            <b>Domain:</b> Environmental data analytics<br><br>
            <b>Study area:</b> Karnataka, India
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h4>Research objectives</h4>
            <p>
            1. Analyse long-term rainfall behaviour across Karnataka.<br><br>
            2. Prepare reliable weekly rainfall time-series data.<br><br>
            3. Develop a hybrid CNN-BiLSTM forecasting model.<br><br>
            4. Evaluate performance against a baseline model.<br><br>
            5. Present findings through an interactive dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # Dataset profile
    st.markdown('<div class="section-kicker">03 / Data foundation</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Dataset & study coverage</div>',
                unsafe_allow_html=True)

    d1, d2 = st.columns([1.15, 1])

    with d1:
        st.markdown("""
        <div class="info-card">
            <h4>Data source</h4>
            <p>
            <b>Provider:</b> India Meteorological Department (IMD)<br><br>
            <b>Dataset:</b> Daily gridded rainfall<br><br>
            <b>Spatial resolution:</b> 0.25° × 0.25°<br><br>
            <b>Temporal coverage:</b> 1 January 1991 – 31 December 2020<br><br>
            <b>Original unit:</b> Millimetres per day<br><br>
            <b>Study region:</b> Karnataka, India
            </p>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown("""
        <div class="info-card">
            <h4>Data preparation</h4>
            <p>
            • Karnataka boundary used for spatial grid selection.<br><br>
            • 252 grid-cell centres selected within the state boundary.<br><br>
            • Latitude-based area weighting used for spatial aggregation.<br><br>
            • Daily spatial coverage checked for completeness.<br><br>
            • Complete seven-day periods aggregated into weekly totals.<br><br>
            • Chronological train, validation and test partitions used.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "The dashboard's rainfall map displays historical spatial rainfall "
        "distribution. The CNN-BiLSTM model forecasts one state-wide weekly "
        "rainfall value; it does not produce spatial or district-level forecasts."
    )

    st.divider()

    # Methodology
    st.markdown('<div class="section-kicker">04 / Research workflow</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Methodology</div>',
                unsafe_allow_html=True)
    st.write(
        "The project follows a chronological data-processing, "
        "model-development and evaluation workflow."
    )

    workflow = [
        ("01", "Data acquisition",
         "Obtain IMD gridded rainfall observations and Karnataka boundary data."),
        ("02", "Spatial processing",
         "Select state grid cells and calculate area-weighted daily rainfall."),
        ("03", "Temporal aggregation",
         "Prepare complete Monday–Sunday weekly rainfall observations."),
        ("04", "Exploratory analysis",
         "Study rainfall distributions, annual variation and seasonal patterns."),
        ("05", "Sequence preparation",
         "Scale the series using training data and create 12-week input sequences."),
        ("06", "Deep learning",
         "Train the CNN-BiLSTM model using chronological training and validation data."),
        ("07", "Evaluation",
         "Assess test performance using MAE, RMSE and R² against a persistence baseline."),
        ("08", "Visualisation",
         "Develop an interactive dashboard for rainfall exploration and model results.")
    ]

    for i in range(0, len(workflow), 2):
        left, right = st.columns(2)
        for col, item in zip([left, right], workflow[i:i+2]):
            number, heading, description = item
            with col:
                st.markdown(f"""
                <div class="info-card">
                    <div class="method-number">{number}</div>
                    <h4>{heading}</h4>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Architecture
    st.markdown('<div class="section-kicker">05 / Deep learning model</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Hybrid CNN-BiLSTM architecture</div>',
                unsafe_allow_html=True)
    st.write(
        "The architecture combines convolutional feature extraction "
        "with bidirectional sequence learning to model patterns in "
        "the historical rainfall series."
    )

    architecture = [
        "Input sequence<br>(12 weeks × 1 feature)",
        "Conv1D<br>64 filters • ReLU",
        "MaxPooling1D<br>Pool size 2",
        "Bidirectional LSTM<br>64 units",
        "Dropout<br>0.2",
        "Dense<br>32 units • ReLU",
        "Output<br>1 weekly rainfall value"
    ]

    for i, layer in enumerate(architecture):
        st.markdown(
            f'<div class="architecture-step">{layer}</div>',
            unsafe_allow_html=True
        )
        if i < len(architecture) - 1:
            st.markdown(
                '<div class="architecture-arrow">↓</div>',
                unsafe_allow_html=True
            )

    st.write("")
    a1, a2, a3 = st.columns(3)
    a1.metric("Input window", "12 weeks")
    a2.metric("Optimiser", "Adam")
    a3.metric("Loss function", "MSE")

    st.caption(
        "The model uses historical state-wide weekly rainfall as its "
        "input feature and predicts the next weekly total."
    )

    st.divider()

    # Research impact
    st.markdown('<div class="section-kicker">06 / Research significance</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Relevance & potential impact</div>',
                unsafe_allow_html=True)

    impact1, impact2 = st.columns(2)

    with impact1:
        st.markdown("""
        <div class="info-card">
            <h4>Local and regional relevance</h4>
            <p>
            Provides rainfall insights relevant to Karnataka's
            agricultural activities, regional water planning and
            preparedness for rainfall variability.
            </p>
        </div>
        <div class="info-card">
            <h4>National relevance</h4>
            <p>
            Demonstrates the application of deep learning to
            Indian rainfall data and contributes to research
            in environmental data analytics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with impact2:
        st.markdown("""
        <div class="info-card">
            <h4>Global relevance</h4>
            <p>
            Contributes to broader research on data-driven
            rainfall forecasting and climate-related
            decision-support methods.
            </p>
        </div>
        <div class="info-card">
            <h4>Cross-cutting themes</h4>
            <p>
            Climate resilience, food security, water resource
            management, disaster risk reduction and
            responsible use of data-driven technologies.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Sustainable Development Goals")

    sdg1, sdg2 = st.columns(2)
    sdg3, sdg4 = st.columns(2)

    sdgs = [
        (sdg1, "SDG 2 — Zero Hunger",
         "Rainfall information may support agricultural planning and food security."),
        (sdg2, "SDG 6 — Clean Water and Sanitation",
         "Rainfall analysis can contribute to water resource planning."),
        (sdg3, "SDG 11 — Sustainable Cities and Communities",
         "Historical rainfall insights can support climate-risk awareness."),
        (sdg4, "SDG 13 — Climate Action",
         "Data-driven rainfall analysis contributes to climate resilience research.")
    ]

    for col, heading, description in sdgs:
        with col:
            st.markdown(f"""
            <div class="sdg-card">
                <h4>{heading}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Technologies
    st.markdown('<div class="section-kicker">07 / Implementation</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Technology stack</div>',
                unsafe_allow_html=True)

    tech = [
        ("Python", "Core programming language"),
        ("TensorFlow / Keras", "Deep learning and model training"),
        ("Pandas / NumPy", "Data preparation and numerical analysis"),
        ("Xarray / NetCDF4", "Gridded rainfall data processing"),
        ("Scikit-learn", "Scaling and evaluation"),
        ("Plotly", "Interactive charts and maps"),
        ("Streamlit", "Web dashboard"),
        ("GitHub", "Version control and source management")
    ]

    for i in range(0, len(tech), 4):
        cols = st.columns(4)
        for col, (name, purpose) in zip(cols, tech[i:i+4]):
            with col:
                st.markdown(f"""
                <div class="info-card">
                    <h4>{name}</h4>
                    <p>{purpose}</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Limitations
    st.markdown('<div class="section-kicker">08 / Scope & future direction</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Limitations and future improvements</div>',
                unsafe_allow_html=True)

    l1, l2 = st.columns(2)

    with l1:
        st.markdown("""
        <div class="info-card">
            <h4>Current limitations</h4>
            <p>
            • Forecast is a single state-wide weekly value.<br><br>
            • Historical rainfall is the model's input feature.<br><br>
            • Extreme rainfall events can be underestimated.<br><br>
            • The model does not provide district-level forecasts.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with l2:
        st.markdown("""
        <div class="info-card">
            <h4>Potential future improvements</h4>
            <p>
            • Incorporate atmospheric and meteorological variables.<br><br>
            • Explore longer seasonal input windows.<br><br>
            • Investigate spatial deep learning architectures.<br><br>
            • Evaluate extreme-rainfall-specific modelling methods.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="closing-panel">
        <b>Research contribution</b><br>
        This project demonstrates an end-to-end workflow for
        processing gridded rainfall observations, developing a
        hybrid deep learning forecasting model, evaluating its
        performance and communicating results through an
        interactive data visualisation application.
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Academic project • MDA507-5 Neural Networks and Deep Learning "
        "• Sensor Data Analytics"
    )
