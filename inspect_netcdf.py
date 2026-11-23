
import xarray as xr
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "data" / "karnataka_rainfall_imd.nc"

# IMPORTANT: Open the nested rain group
ds = xr.open_dataset(
    FILE_PATH,
    group="rain",
    engine="netcdf4"
)

print("\n========== RAINFALL DATASET ==========")
print(ds)

# Dimensions
print("\nDimensions:")
print(dict(ds.sizes))

# Coordinate information
print("\nLatitude values:")
print(ds["lat"].values)

print("\nLongitude values:")
print(ds["lon"].values)

# Time information
print("\nTime attributes:")
print(ds["time"].attrs)

print("\nFirst 10 time values:")
print(ds["time"].values[:10])

print("\nLast 10 time values:")
print(ds["time"].values[-10:])

# Rainfall variable information
rain = ds["rain"]

print("\nRainfall attributes:")
print(rain.attrs)

print("\nRainfall shape:")
print(rain.shape)

# Missing and statistical values
print("\nRainfall statistics:")

print("Minimum:", float(rain.min(skipna=True).values))
print("Maximum:", float(rain.max(skipna=True).values))
print("Mean:", float(rain.mean(skipna=True).values))

print(
    "Missing values:",
    int(rain.isnull().sum().values)
)

# Check unusual values
print(
    "Negative values:",
    int((rain < 0).sum().values)
)

print(
    "Zero values:",
    int((rain == 0).sum().values)
)

ds.close()

print("\nInspection completed.")
