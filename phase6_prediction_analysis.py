
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "phase6_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_FILE = RESULTS_DIR / "test_predictions.csv"

# --------------------------------------------------
# 2. Load test predictions
# --------------------------------------------------

df = pd.read_csv(PREDICTION_FILE)
df["Week_Ending"] = pd.to_datetime(df["Week_Ending"])

actual = df["Actual_Rainfall_mm"].to_numpy()
predicted = df["Predicted_Rainfall_mm"].to_numpy()
baseline = df["Persistence_Baseline_mm"].to_numpy()

residuals = predicted - actual
absolute_errors = np.abs(residuals)

print("\nPHASE 6: PREDICTION ANALYSIS")
print("--------------------------------")
print("Test observations:", len(df))

# --------------------------------------------------
# 3. Calculate performance metrics
# --------------------------------------------------

def evaluate(y_true, y_pred):
    return {
        "MAE_mm": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

model_metrics = evaluate(actual, predicted)
baseline_metrics = evaluate(actual, baseline)

improvement = (
    (baseline_metrics["MAE_mm"] - model_metrics["MAE_mm"])
    / baseline_metrics["MAE_mm"]
) * 100

print("\nCNN-BiLSTM:", model_metrics)
print("Persistence:", baseline_metrics)
print(f"MAE improvement: {improvement:.2f}%")

# --------------------------------------------------
# 4. Prediction distribution and correlation
# --------------------------------------------------

distribution = {
    "Actual_mean_mm": float(np.mean(actual)),
    "Predicted_mean_mm": float(np.mean(predicted)),
    "Actual_std_mm": float(np.std(actual)),
    "Predicted_std_mm": float(np.std(predicted)),
    "Actual_median_mm": float(np.median(actual)),
    "Predicted_median_mm": float(np.median(predicted)),
    "Actual_max_mm": float(np.max(actual)),
    "Predicted_max_mm": float(np.max(predicted)),
    "Correlation": float(np.corrcoef(actual, predicted)[0, 1])
}

# --------------------------------------------------
# 5. Rainfall category analysis
# --------------------------------------------------

bins = [0, 10, 25, 50, 100, np.inf]

labels = [
    "0-10 mm",
    "10-25 mm",
    "25-50 mm",
    "50-100 mm",
    "100+ mm"
]

df["Rainfall_Category"] = pd.cut(
    actual,
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True
)

category_results = []

for category in labels:
    subset = df[df["Rainfall_Category"] == category]

    if len(subset) == 0:
        continue

    category_results.append({
        "Rainfall_Category": category,
        "Count": len(subset),
        "MAE_mm": mean_absolute_error(
            subset["Actual_Rainfall_mm"],
            subset["Predicted_Rainfall_mm"]
        ),
        "RMSE_mm": np.sqrt(mean_squared_error(
            subset["Actual_Rainfall_mm"],
            subset["Predicted_Rainfall_mm"]
        ))
    })

category_df = pd.DataFrame(category_results)

category_df.to_csv(
    OUTPUT_DIR / "rainfall_category_analysis.csv",
    index=False
)

print("\nRAIN​​FALL CATEGORY ANALYSIS")
print(category_df.to_string(index=False))

# --------------------------------------------------
# 6. Identify heavy rainfall prediction errors
# --------------------------------------------------

top_heavy = df.nlargest(
    10, "Actual_Rainfall_mm"
).copy()

top_heavy["Underprediction_mm"] = (
    top_heavy["Actual_Rainfall_mm"]
    - top_heavy["Predicted_Rainfall_mm"]
)

top_heavy["Underprediction_Percent"] = (
    top_heavy["Underprediction_mm"]
    / top_heavy["Actual_Rainfall_mm"].replace(0, np.nan)
) * 100

top_heavy.to_csv(
    OUTPUT_DIR / "top_heavy_rainfall_predictions.csv",
    index=False
)

print("\nTOP 10 HEAVY RAINFALL WEEKS")
print(
    top_heavy[
        [
            "Week_Ending",
            "Actual_Rainfall_mm",
            "Predicted_Rainfall_mm",
            "Underprediction_mm"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# 7. Plot 1: Actual vs predicted scatter plot
# --------------------------------------------------

plt.figure(figsize=(8, 7))

plt.scatter(
    actual, predicted,
    alpha=0.65
)

maximum = max(actual.max(), predicted.max())

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel("Actual rainfall (mm)")
plt.ylabel("Predicted rainfall (mm)")
plt.title("Actual vs Predicted Weekly Rainfall")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_actual_vs_predicted_scatter.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 8. Plot 2: Residual distribution
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    residuals,
    bins=25,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--",
    label="Zero error"
)

plt.xlabel("Prediction error (Predicted - Actual) mm")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Residuals")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_residual_distribution.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 9. Plot 3: Error by rainfall category
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    category_df["Rainfall_Category"].astype(str),
    category_df["MAE_mm"]
)

plt.xlabel("Actual rainfall category")
plt.ylabel("Mean Absolute Error (mm)")
plt.title("Prediction Error Across Rainfall Categories")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_error_by_rainfall_category.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 10. Plot 4: Heavy rainfall comparison
# --------------------------------------------------

heavy_plot = top_heavy.sort_values(
    "Actual_Rainfall_mm"
)

x = np.arange(len(heavy_plot))
width = 0.38

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    heavy_plot["Actual_Rainfall_mm"],
    width,
    label="Actual"
)

plt.bar(
    x + width / 2,
    heavy_plot["Predicted_Rainfall_mm"],
    width,
    label="Predicted"
)

plt.xticks(
    x,
    heavy_plot["Week_Ending"].dt.strftime("%d-%b-%Y"),
    rotation=45,
    ha="right"
)

plt.xlabel("Week ending")
plt.ylabel("Rainfall (mm)")
plt.title("Actual vs Predicted: Top 10 Heavy Rainfall Weeks")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_heavy_rainfall_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 11. Save complete analysis summary
# --------------------------------------------------

summary = {
    "test_observations": len(df),
    "CNN_BiLSTM_metrics": model_metrics,
    "Persistence_metrics": baseline_metrics,
    "MAE_improvement_percent": float(improvement),
    "prediction_distribution": distribution,
    "category_analysis": category_results
}

with open(
    OUTPUT_DIR / "phase6_analysis_summary.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)

print("\n--------------------------------")
print("PHASE 6 ANALYSIS COMPLETED")
print("Output folder:", OUTPUT_DIR)
print("Generated four analysis plots.")
print("Saved category analysis and heavy rainfall results.")
