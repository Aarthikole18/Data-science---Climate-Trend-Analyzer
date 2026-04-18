# ==========================================
# Climate Trend Analyzer (Final Polished)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# ------------------------------
# CREATE OUTPUT FOLDER
# ------------------------------
os.makedirs("outputs", exist_ok=True)

# ------------------------------
# 1. LOAD DATA
# ------------------------------
print("Loading dataset...")
df = pd.read_csv("data/climate.csv")

# ------------------------------
# 2. PREPROCESSING
# ------------------------------
print("Preprocessing data...")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())
df['Rainfall'] = df['Rainfall'].fillna(df['Rainfall'].mean())
df['CO2'] = df['CO2'].fillna(df['CO2'].mean())

df['Time_Index'] = np.arange(len(df))

# ------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------
print("Creating features...")

# ✅ FIX: smaller window for better accuracy
df['Rolling_Mean'] = df['Temperature'].rolling(window=3).mean()

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# ------------------------------
# 4. ANOMALY DETECTION
# ------------------------------
print("Detecting anomalies...")

mean_temp = df['Temperature'].mean()
std_temp = df['Temperature'].std()

df['Anomaly'] = ((df['Temperature'] > mean_temp + 2 * std_temp) |
                 (df['Temperature'] < mean_temp - 2 * std_temp))

anomalies = df[df['Anomaly'] == True]

print(f"Total anomalies detected: {len(anomalies)}")

# ------------------------------
# 5. ML MODEL (MULTI-FEATURE)
# ------------------------------
print("Training ML model...")

features = df[['Time_Index', 'Rainfall', 'CO2']]
target = df['Temperature']

model = LinearRegression()
model.fit(features, target)

df['Advanced_Trend'] = model.predict(features)

# ------------------------------
# 6. CORRELATION ANALYSIS
# ------------------------------
print("\nCorrelation Analysis:")

correlation = df[['Temperature', 'Rainfall', 'CO2']].corr()
print(correlation)

# ------------------------------
# 7. FORECASTING
# ------------------------------
print("\nForecasting...")

future_steps = 12
future_time = np.arange(len(df), len(df) + future_steps)

last_rainfall = df['Rainfall'].iloc[-1]
last_co2 = df['CO2'].iloc[-1]

future_features = pd.DataFrame({
    'Time_Index': future_time,
    'Rainfall': [last_rainfall] * future_steps,
    'CO2': [last_co2] * future_steps
})

future_predictions = model.predict(future_features)

future_dates = pd.date_range(
    start=df['Date'].iloc[-1],
    periods=future_steps + 1,
    freq='M'
)[1:]

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Temperature': future_predictions
})

# ------------------------------
# 8. VISUALIZATION
# ------------------------------
print("\nGenerating plots...")

plt.figure(figsize=(14, 7))

plt.plot(df['Date'], df['Temperature'], label='Temperature')
plt.plot(df['Date'], df['Rolling_Mean'], label='Rolling Mean', linewidth=2)
plt.plot(df['Date'], df['Advanced_Trend'], label='ML Trend', linestyle='--')

plt.scatter(anomalies['Date'], anomalies['Temperature'],
            color='red', label='Anomalies', s=80)

plt.plot(forecast_df['Date'], forecast_df['Predicted_Temperature'],
         label='Forecast', linestyle='dotted')

plt.title("Climate Trend Analysis (Advanced)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()

plt.savefig("outputs/climate_trend_advanced.png")
plt.show()

# ------------------------------
# 9. SAVE OUTPUTS
# ------------------------------
print("\nSaving outputs...")

df.to_csv("outputs/processed_data.csv", index=False)
forecast_df.to_csv("outputs/forecast.csv", index=False)

# ------------------------------
# 10. INSIGHTS
# ------------------------------
print("\n🔍 KEY INSIGHTS:")

print(f"Average Temp: {df['Temperature'].mean():.2f} °C")
print(f"Max Temp: {df['Temperature'].max():.2f} °C")
print(f"Min Temp: {df['Temperature'].min():.2f} °C")
print(f"Anomalies: {len(anomalies)}")

if df['Advanced_Trend'].iloc[-1] > df['Advanced_Trend'].iloc[0]:
    print("📈 Increasing Temperature Trend")
else:
    print("📉 Decreasing Temperature Trend")

print("\n✅ Everything working perfectly!")