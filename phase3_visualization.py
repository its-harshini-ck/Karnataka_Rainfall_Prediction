
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DAILY_FILE = BASE_DIR / "processed" / "karnataka_daily_rainfall.csv"
WEEKLY_FILE = BASE_DIR / "processed" / "karnataka_weekly_rainfall.csv"

PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nSTEP 1: Loading processed rainfall data...")

daily = pd.read_csv(DAILY_FILE, parse_dates=["Date"])

weekly = pd.read_csv(
    WEEKLY_FILE,
    parse_dates=["Week_Start_Monday", "Week_Ending_Sunday"]
)

print("Daily records:", len(daily))
print("Weekly records:", len(weekly))


# --------------------------------------------------
# 3. DATA QUALITY VALIDATION
# --------------------------------------------------

print("\nSTEP 2: Validating data...")

print("Daily duplicate dates:", daily["Date"].duplicated().sum())
print("Daily missing rainfall:", daily["Daily_Rainfall_mm"].isna().sum())
print("Negative daily rainfall:", (daily["Daily_Rainfall_mm"] < 0).sum())

date_differences = daily["Date"].sort_values().diff().dropna()

print(
    "Daily date gaps:",
    (date_differences != pd.Timedelta(days=1)).sum()
)

print(
    "Minimum spatial coverage:",
    daily["Spatial_Coverage"].min()
)

print(
    "Maximum spatial coverage:",
    daily["Spatial_Coverage"].max()
)

print(
    "Average spatial coverage:",
    daily["Spatial_Coverage"].mean()
)

print(
    "Weekly missing rainfall:",
    weekly["Weekly_Rainfall_mm"].isna().sum()
)

print(
    "Weekly duplicate start dates:",
    weekly["Week_Start_Monday"].duplicated().sum()
)

print(
    "Invalid Monday starts:",
    (weekly["Week_Start_Monday"].dt.dayofweek != 0).sum()
)

print(
    "Invalid Sunday endings:",
    (weekly["Week_Ending_Sunday"].dt.dayofweek != 6).sum()
)

print(
    "Invalid week lengths:",
    (
        weekly["Week_Ending_Sunday"]
        - weekly["Week_Start_Monday"]
        != pd.Timedelta(days=6)
    ).sum()
)

if (daily["Daily_Rainfall_mm"] < 0).any():
    raise ValueError("Negative rainfall values found.")

if daily["Date"].duplicated().any():
    raise ValueError("Duplicate daily dates found.")

if daily["Daily_Rainfall_mm"].isna().any():
    raise ValueError("Missing daily rainfall values found.")

if weekly["Weekly_Rainfall_mm"].isna().any():
    raise ValueError("Missing weekly rainfall values found.")

if (weekly["Week_Start_Monday"].dt.dayofweek != 0).any():
    raise ValueError("Some weekly periods do not start on Monday.")

if (weekly["Week_Ending_Sunday"].dt.dayofweek != 6).any():
    raise ValueError("Some weekly periods do not end on Sunday.")


# --------------------------------------------------
# 4. PREPARE TIME-SERIES DATA
# --------------------------------------------------

daily = daily.sort_values("Date").set_index("Date")
weekly = weekly.sort_values("Week_Start_Monday")

monthly_rainfall = (
    daily["Daily_Rainfall_mm"]
    .resample("MS")
    .sum()
)

annual_rainfall = (
    daily["Daily_Rainfall_mm"]
    .resample("YS")
    .sum()
)

monthly_climatology = monthly_rainfall.groupby(
    monthly_rainfall.index.month
).mean()

monthly_table = monthly_rainfall.to_frame("Rainfall")
monthly_table["Year"] = monthly_table.index.year
monthly_table["Month"] = monthly_table.index.month

heatmap_data = monthly_table.pivot(
    index="Year",
    columns="Month",
    values="Rainfall"
)

# Use readable month names
month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


# --------------------------------------------------
# 5. PLOT 1: WEEKLY RAINFALL TREND
# --------------------------------------------------

print("\nSTEP 3: Generating visualizations...")

plt.figure(figsize=(15, 5))

plt.plot(
    weekly["Week_Ending_Sunday"],
    weekly["Weekly_Rainfall_mm"],
    linewidth=0.7
)

plt.title("Weekly Rainfall in Karnataka (1991–2020)")
plt.xlabel("Year")
plt.ylabel("Weekly Rainfall (mm)")
plt.tight_layout()

plt.savefig(PLOT_DIR / "01_weekly_rainfall_trend.png", dpi=300)
plt.show()
plt.close()


# --------------------------------------------------
# 6. PLOT 2: MONTHLY RAINFALL CLIMATOLOGY
# --------------------------------------------------

plt.figure(figsize=(11, 5))

plt.bar(
    month_names,
    monthly_climatology.values
)

plt.title("Average Monthly Rainfall in Karnataka (1991–2020)")
plt.xlabel("Month")
plt.ylabel("Average Monthly Rainfall (mm)")
plt.tight_layout()

plt.savefig(PLOT_DIR / "02_monthly_rainfall.png", dpi=300)
plt.show()
plt.close()


# --------------------------------------------------
# 7. PLOT 3: ANNUAL RAINFALL
# --------------------------------------------------

plt.figure(figsize=(13, 5))

plt.plot(
    annual_rainfall.index.year,
    annual_rainfall.values,
    marker="o",
    linewidth=1.5
)

plt.title("Annual Rainfall in Karnataka (1991–2020)")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.xticks(range(1991, 2021, 2), rotation=45)
plt.tight_layout()

plt.savefig(PLOT_DIR / "03_annual_rainfall.png", dpi=300)
plt.show()
plt.close()


# --------------------------------------------------
# 8. PLOT 4: YEAR-MONTH RAINFALL HEATMAP
# --------------------------------------------------

plt.figure(figsize=(14, 9))

sns.heatmap(
    heatmap_data,
    cmap="YlGnBu",
    linewidths=0.2,
    cbar_kws={"label": "Monthly Rainfall (mm)"}
)

plt.title("Monthly Rainfall Distribution Across Years")
plt.xlabel("Month")
plt.ylabel("Year")

plt.xticks(
    ticks=[i + 0.5 for i in range(12)],
    labels=month_names
)

plt.tight_layout()

plt.savefig(PLOT_DIR / "04_rainfall_heatmap.png", dpi=300)
plt.show()
plt.close()


# --------------------------------------------------
# 9. PLOT 5: WEEKLY RAINFALL DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    weekly["Weekly_Rainfall_mm"],
    bins=40,
    kde=True
)

plt.title("Distribution of Weekly Rainfall")
plt.xlabel("Weekly Rainfall (mm)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(PLOT_DIR / "05_weekly_rainfall_distribution.png", dpi=300)
plt.show()
plt.close()


# --------------------------------------------------
# 10. SUMMARY STATISTICS
# --------------------------------------------------

print("\nSTEP 4: Rainfall statistical summary")

print("\nDaily rainfall:")
print(daily["Daily_Rainfall_mm"].describe())

print("\nWeekly rainfall:")
print(weekly["Weekly_Rainfall_mm"].describe())

print("\nAverage monthly rainfall:")
print(monthly_climatology.round(2))

print("\nAnnual rainfall:")
print(annual_rainfall.round(2))

print("\nAll plots saved in:", PLOT_DIR)

print("\nPHASE 3 VISUALIZATION COMPLETED!")
