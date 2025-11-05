# run_energy_forecast.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------- Helpers ----------
def create_sequences(data, look_back=24):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

print("📥 Looking for dataset 'energy.csv' in current folder...")
if not os.path.exists("energy.csv"):
    print("⛔ File 'energy.csv' not found. Please add your CSV and ensure it has a numeric column named 'consumption' or 'Global_active_power'.")
    raise SystemExit(1)

# ---------- Data loading & target selection (handles semicolons) ----------
print("🔎 Reading CSV...")
# Try semicolon CSV first (UCI/Kaggle power dataset is ';' delimited)
df = pd.read_csv("energy.csv", sep=';', low_memory=False)

# If it collapsed into one column (still semicolon inside), try automatic delimiter inference
if df.shape[1] == 1:
    try:
        df = pd.read_csv("energy.csv", sep=None, engine="python", low_memory=False)
    except Exception:
        pass

print("Raw data shape:", df.shape)
df.columns = [c.strip() for c in df.columns]  # tidy names
df.replace('?', np.nan, inplace=True)

# Combine Date + Time if present
if 'Date' in df.columns and 'Time' in df.columns:
    df['datetime'] = pd.to_datetime(
        df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip(),
        dayfirst=True, errors='coerce'
    )
    df = df.sort_values('datetime').reset_index(drop=True)

# Choose target column (prefer Global_active_power)
if 'Global_active_power' in df.columns:
    target_col = 'Global_active_power'
else:
    candidates = [c for c in df.columns
                  if 'global_active' in c.lower()
                  or 'active_power' in c.lower()
                  or 'power' in c.lower()
                  or 'load' in c.lower()
                  or 'consum' in c.lower()]
    if candidates:
        target_col = candidates[0]
        print(f"ℹ️ Using detected column '{target_col}' as target.")
    else:
        print("⛔ Could not find a suitable target column. Ensure your CSV has 'Global_active_power' or a numeric 'consumption' column.")
        raise SystemExit(1)

# Make target numeric and drop NaNs
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# Create the expected 'consumption' column
df.rename(columns={target_col: 'consumption'}, inplace=True)
print(f"✅ Using column 'consumption' (from '{target_col}')")
print("Cleaned data shape:", df.shape)

# ---------- Prepare series ----------
series = df['consumption'].astype('float32').values.reshape(-1, 1)

# Scale 0..1
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# Sequences
LOOK_BACK = 24  # use last 24 timesteps to predict the next one
X, y = create_sequences(series_scaled, LOOK_BACK)

# Train/Test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"📦 Sequences -> X: {X.shape}, y: {y.shape}")
print(f"🧪 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ---------- Model ----------
model = Sequential([
    LSTM(64, input_shape=(LOOK_BACK, 1), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()

EPOCHS = 20
BATCH = 64

print(f"🔁 Training LSTM for {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.1,
    verbose=2
)

# ---------- Evaluate ----------
print("🔎 Evaluating on test set...")
pred_scaled = model.predict(X_test).reshape(-1, 1)
y_test_scaled = y_test.reshape(-1, 1)

# Inverse scale back to original units
pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test_scaled)

rmse = np.sqrt(mean_squared_error(actual, pred))
print(f"✅ Test RMSE: {rmse:.4f}")

# ---------- Save artifacts ----------
model.save("energy_lstm.h5")
print("💾 Saved model: energy_lstm.h5")

# Training loss plot
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.close()
print("🖼️ Saved training_loss.png")

# Actual vs Predicted (first 200 points)
n = min(200, len(actual))
plt.figure(figsize=(10, 4))
plt.plot(actual[:n], label='Actual')
plt.plot(pred[:n], label='Predicted')
plt.title('Actual vs Predicted (first samples)')
plt.legend()
plt.tight_layout()
plt.savefig("forecast_plot.png", dpi=150)
plt.close()
print("🖼️ Saved forecast_plot.png")

print("🎯 Done. Files created: energy_lstm.h5, training_loss.png, forecast_plot.png")