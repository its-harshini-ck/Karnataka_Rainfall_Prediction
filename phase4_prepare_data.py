
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

WEEKLY_FILE = (
    BASE_DIR / "processed" / "karnataka_weekly_rainfall.csv"
)

OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

LOOKBACK = 12

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


# --------------------------------------------------
# 2. LOAD WEEKLY DATA
# --------------------------------------------------

print("\nSTEP 1: Loading weekly rainfall data...")

df = pd.read_csv(
    WEEKLY_FILE,
    parse_dates=["Week_Start_Monday", "Week_Ending_Sunday"]
)

df = df.sort_values("Week_Start_Monday").reset_index(drop=True)

values = df["Weekly_Rainfall_mm"].values.astype("float32")

dates = df["Week_Ending_Sunday"].values

print("Total weekly observations:", len(values))


# --------------------------------------------------
# 3. DEFINE CHRONOLOGICAL SPLITS
# --------------------------------------------------

print("\nSTEP 2: Creating chronological splits...")

n = len(values)

train_end = int(n * TRAIN_RATIO)

validation_end = int(
    n * (TRAIN_RATIO + VALIDATION_RATIO)
)

print("Training boundary:", train_end)
print("Validation boundary:", validation_end)
print("Testing boundary:", n)


# --------------------------------------------------
# 4. SCALE DATA USING TRAINING DATA ONLY
# --------------------------------------------------

print("\nSTEP 3: Scaling rainfall values...")

scaler = MinMaxScaler(feature_range=(0, 1))

# Fit only on the training period
scaler.fit(values[:train_end].reshape(-1, 1))

# Transform the complete series using training-fitted scaler
scaled_values = scaler.transform(
    values.reshape(-1, 1)
).astype("float32")

scaler_file = OUTPUT_DIR / "rainfall_scaler.pkl"

joblib.dump(scaler, scaler_file)

print("Scaler saved:", scaler_file)


# --------------------------------------------------
# 5. CREATE SEQUENCES
# --------------------------------------------------

print("\nSTEP 4: Creating 12-week sequences...")

X = []
y = []
target_indices = []

for i in range(LOOKBACK, n):

    X.append(
        scaled_values[i - LOOKBACK:i]
    )

    y.append(
        scaled_values[i]
    )

    target_indices.append(i)

X = np.array(X, dtype="float32")
y = np.array(y, dtype="float32")

target_indices = np.array(target_indices)

print("Complete input shape:", X.shape)
print("Complete target shape:", y.shape)


# --------------------------------------------------
# 6. SPLIT SEQUENCES CHRONOLOGICALLY
# --------------------------------------------------

train_mask = target_indices < train_end

validation_mask = (
    (target_indices >= train_end)
    & (target_indices < validation_end)
)

test_mask = target_indices >= validation_end

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[validation_mask]
y_val = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]

dates_train = dates[target_indices[train_mask]]
dates_val = dates[target_indices[validation_mask]]
dates_test = dates[target_indices[test_mask]]


# --------------------------------------------------
# 7. DISPLAY DATASET SHAPES
# --------------------------------------------------

print("\nSTEP 5: Dataset split summary")

print("\nTraining:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTesting:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 8. SAVE MODEL-READY DATA
# --------------------------------------------------

print("\nSTEP 6: Saving model-ready datasets...")

np.savez_compressed(
    OUTPUT_DIR / "cnn_bilstm_data.npz",

    X_train=X_train,
    y_train=y_train,

    X_val=X_val,
    y_val=y_val,

    X_test=X_test,
    y_test=y_test,

    dates_train=dates_train,
    dates_val=dates_val,
    dates_test=dates_test
)

print(
    "Saved:",
    OUTPUT_DIR / "cnn_bilstm_data.npz"
)


# --------------------------------------------------
# 9. PRINT DATE RANGES
# --------------------------------------------------

print("\nTarget date ranges:")

print("Training:")
print(pd.to_datetime(dates_train[[0, -1]]))

print("\nValidation:")
print(pd.to_datetime(dates_val[[0, -1]]))

print("\nTesting:")
print(pd.to_datetime(dates_test[[0, -1]]))

print("\nPHASE 4 DATA PREPARATION COMPLETED!")
