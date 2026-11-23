
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import shape, Point
from shapely.ops import unary_union


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

NETCDF_FILE = BASE_DIR / "data" / "karnataka_rainfall_imd.nc"
GEOJSON_FILE = BASE_DIR / "data" / "karnataka.geojson"

OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_DAILY_COVERAGE = 0.80


# --------------------------------------------------
# 2. LOAD KARNATAKA GEOJSON BOUNDARY
# --------------------------------------------------

print("\nSTEP 1: Loading Karnataka boundary...")

with open(GEOJSON_FILE, "r", encoding="utf-8") as file:
    geojson = json.load(file)

if geojson["type"] == "FeatureCollection":
    geometries = [
        shape(feature["geometry"])
        for feature in geojson["features"]
        if feature.get("geometry") is not None
    ]
    karnataka_boundary = unary_union(geometries)

elif geojson["type"] == "Feature":
    karnataka_boundary = shape(geojson["geometry"])

else:
    karnataka_boundary = shape(geojson)

if karnataka_boundary.is_empty or not karnataka_boundary.is_valid:
    raise ValueError("Karnataka boundary is empty or invalid.")

print("Boundary loaded successfully.")
print("Boundary bounds:", karnataka_boundary.bounds)


# --------------------------------------------------
# 3. LOAD IMD NETCDF DATA
# --------------------------------------------------

print("\nSTEP 2: Loading IMD rainfall dataset...")

ds = xr.open_dataset(
    NETCDF_FILE,
    group="rain",
    engine="netcdf4"
)

rain = ds["rain"].transpose("time", "lat", "lon")

latitudes = ds["lat"].values
longitudes = ds["lon"].values
dates = pd.to_datetime(ds["time"].values)

print("Dataset dimensions:", dict(ds.sizes))
print("Rainfall units:", rain.attrs.get("units", "Not specified"))


# --------------------------------------------------
# 4. SELECT GRID CELLS INSIDE KARNATAKA
# --------------------------------------------------

print("\nSTEP 3: Selecting Karnataka grid cells...")

selected_cells = []

for i, latitude in enumerate(latitudes):
    for j, longitude in enumerate(longitudes):

        point = Point(float(longitude), float(latitude))

        if karnataka_boundary.covers(point):
            selected_cells.append((i, j, latitude))

if len(selected_cells) == 0:
    ds.close()
    raise ValueError(
        "No grid cells found inside Karnataka. "
        "Check the GeoJSON coordinate system and boundary."
    )

print("Selected grid cells:", len(selected_cells))

if len(selected_cells) < 20:
    ds.close()
    raise ValueError(
        "Too few grid cells selected. Check the boundary file "
        "and coordinate reference system."
    )


# --------------------------------------------------
# 5. EXTRACT RAINFALL AND CALCULATE AREA-WEIGHTED MEAN
# --------------------------------------------------

print("\nSTEP 4: Calculating daily regional rainfall...")

rain_values = rain.values

daily_rainfall = np.full(len(dates), np.nan)
daily_coverage = np.zeros(len(dates))

# Approximate grid-cell area weighting using cos(latitude)
weights = np.array([
    np.cos(np.deg2rad(latitude))
    for _, _, latitude in selected_cells
])

for day in range(len(dates)):

    values = np.array([
        rain_values[day, i, j]
        for i, j, _ in selected_cells
    ])

    valid = np.isfinite(values)

    total_weight = weights.sum()
    valid_weight = weights[valid].sum()

    coverage = valid_weight / total_weight
    daily_coverage[day] = coverage

    if coverage >= MIN_DAILY_COVERAGE and valid.any():

        daily_rainfall[day] = np.average(
            values[valid],
            weights=weights[valid]
        )

daily_df = pd.DataFrame({
    "Date": dates,
    "Daily_Rainfall_mm": daily_rainfall,
    "Spatial_Coverage": daily_coverage
})

daily_file = OUTPUT_DIR / "karnataka_daily_rainfall.csv"
daily_df.to_csv(daily_file, index=False)

print("Daily rainfall file saved:", daily_file)
print("Valid daily observations:", daily_df["Daily_Rainfall_mm"].notna().sum())
print("Invalid daily observations:", daily_df["Daily_Rainfall_mm"].isna().sum())


# --------------------------------------------------
# 6. AGGREGATE DAILY RAINFALL INTO MONDAY-SUNDAY WEEKS
# --------------------------------------------------

print("\nSTEP 5: Creating weekly rainfall totals...")

daily_series = daily_df.set_index("Date")["Daily_Rainfall_mm"]

weekly_rainfall = daily_series.resample(
    "W-SUN",
    label="right",
    closed="right"
).sum(min_count=7)

weekly_df = weekly_rainfall.dropna().reset_index()

weekly_df.columns = [
    "Week_Ending_Sunday",
    "Weekly_Rainfall_mm"
]

weekly_df["Week_Start_Monday"] = (
    weekly_df["Week_Ending_Sunday"] - pd.Timedelta(days=6)
)

weekly_df = weekly_df[
    [
        "Week_Start_Monday",
        "Week_Ending_Sunday",
        "Weekly_Rainfall_mm"
    ]
]

weekly_file = OUTPUT_DIR / "karnataka_weekly_rainfall.csv"
weekly_df.to_csv(weekly_file, index=False)


# --------------------------------------------------
# 7. PRINT FINAL DATASET SUMMARY
# --------------------------------------------------

print("\nSTEP 6: Final preprocessing summary")

print("\nDaily dataset:")
print(daily_df.head())

print("\nWeekly dataset:")
print(weekly_df.head())

print("\nTotal complete weeks:", len(weekly_df))

if len(weekly_df) > 0:

    print(
        "Weekly rainfall minimum:",
        weekly_df["Weekly_Rainfall_mm"].min()
    )

    print(
        "Weekly rainfall maximum:",
        weekly_df["Weekly_Rainfall_mm"].max()
    )

    print(
        "Weekly rainfall average:",
        weekly_df["Weekly_Rainfall_mm"].mean()
    )

    print(
        "Weekly dataset date range:",
        weekly_df["Week_Start_Monday"].min(),
        "to",
        weekly_df["Week_Ending_Sunday"].max()
    )

print("\nSaved files:")
print(daily_file)
print(weekly_file)

ds.close()

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
