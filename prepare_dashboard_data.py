
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, shape
from shapely.ops import unary_union

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NC_FILE = os.path.join(BASE_DIR, "data", "karnataka_rainfall_imd.nc")
GEOJSON_FILE = os.path.join(BASE_DIR, "data", "karnataka.geojson")
OUTPUT_DIR = os.path.join(BASE_DIR, "dashboard_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Karnataka boundary...")
with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
    boundary_data = json.load(f)

if boundary_data["type"] == "FeatureCollection":
    geometries = [
        shape(feature["geometry"])
        for feature in boundary_data["features"]
        if feature.get("geometry") is not None
    ]
    boundary = unary_union(geometries)

elif boundary_data["type"] == "Feature":
    boundary = shape(boundary_data["geometry"])

else:
    boundary = shape(boundary_data)

print("Opening IMD rainfall dataset...")
ds = xr.open_dataset(
    NC_FILE,
    group="rain",
    engine="netcdf4"
)

rain = ds["rain"]

# Select grid-cell centres located inside Karnataka
latitudes = rain["lat"].values
longitudes = rain["lon"].values

selected_cells = []

for i, lat in enumerate(latitudes):
    for j, lon in enumerate(longitudes):
        point = Point(float(lon), float(lat))

        if boundary.covers(point):
            selected_cells.append((i, j, float(lat), float(lon)))

print(f"Selected grid cells: {len(selected_cells)}")

# Extract selected grid cells
records = []

for i, j, lat, lon in selected_cells:
    values = rain[:, i, j].values

    cell_df = pd.DataFrame({
        "date": pd.to_datetime(rain["time"].values),
        "latitude": lat,
        "longitude": lon,
        "rainfall_mm": values
    })

    records.append(cell_df)

spatial_daily = pd.concat(records, ignore_index=True)

# Exclude invalid rainfall values
spatial_daily["rainfall_mm"] = pd.to_numeric(
    spatial_daily["rainfall_mm"], errors="coerce"
)

spatial_daily.loc[
    spatial_daily["rainfall_mm"] < 0, "rainfall_mm"
] = np.nan

# Monthly rainfall totals per grid cell
print("Calculating monthly spatial rainfall...")

spatial_daily["year"] = spatial_daily["date"].dt.year
spatial_daily["month"] = spatial_daily["date"].dt.month

monthly = (
    spatial_daily
    .groupby(["year", "month", "latitude", "longitude"])
    .agg(
        rainfall_mm=("rainfall_mm", lambda x: x.sum(min_count=1)),
        valid_days=("rainfall_mm", "count")
    )
    .reset_index()
)

monthly.to_csv(
    os.path.join(OUTPUT_DIR, "spatial_monthly.csv"),
    index=False
)

# Annual rainfall totals per grid cell
print("Calculating annual spatial rainfall...")

annual = (
    spatial_daily
    .groupby(["year", "latitude", "longitude"])
    .agg(
        rainfall_mm=("rainfall_mm", lambda x: x.sum(min_count=1)),
        valid_days=("rainfall_mm", "count")
    )
    .reset_index()
)

annual.to_csv(
    os.path.join(OUTPUT_DIR, "spatial_annual.csv"),
    index=False
)

# Save Karnataka boundary for the dashboard
with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
    geojson = json.load(f)

with open(
    os.path.join(OUTPUT_DIR, "karnataka.geojson"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(geojson, f)

ds.close()

print("\nDashboard data preparation completed!")
print(f"Monthly records: {len(monthly)}")
print(f"Annual records: {len(annual)}")
print(f"Output folder: {OUTPUT_DIR}")
