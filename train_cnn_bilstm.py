
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Dropout, Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_PATH = "processed/cnn_bilstm_data.npz"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load prepared data
data = np.load(DATA_PATH)

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)

# CNN-BiLSTM architecture
model = Sequential([
    Input(shape=(12, 1)),
    Conv1D(filters=64, kernel_size=3,
           activation="relu", padding="same"),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# Training callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-5
    ),
    ModelCheckpoint(
        filepath=os.path.join(
            RESULTS_DIR, "best_cnn_bilstm.keras"
        ),
        monitor="val_loss",
        save_best_only=True
    )
]

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save(os.path.join(RESULTS_DIR, "final_cnn_bilstm.keras"))

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("CNN-BiLSTM Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "training_loss.png"),
    dpi=300
)
plt.close()

# Evaluate best saved model
best_model = tf.keras.models.load_model(
    os.path.join(RESULTS_DIR, "best_cnn_bilstm.keras")
)

predictions = best_model.predict(X_test, verbose=0)

# Save scaled predictions for reference
np.save(
    os.path.join(RESULTS_DIR, "test_predictions_scaled.npy"),
    predictions
)

print("\nTraining and evaluation completed.")
print("Best model saved in:", RESULTS_DIR)
