# ==============================
# Climate Trend Analyzer - main.py
# ==============================

# ------------------------------
# IMPORT LIBRARIES
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# ------------------------------
# CREATE OUTPUT FOLDER (AUTO)
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

df['Time_Index'] = np.arange(len(df))

# ------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------
print("Creating features...")

df['Rolling_Mean'] = df['Temperature'].rolling(window=12).mean()

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
# 5. TREND ANALYSIS
# ------------------------------
print("Performing trend analysis...")

model = LinearRegression()
model.fit(df[['Time_Index']], df['Temperature'])

df['Trend_Line'] = model.predict(df[['Time_Index']])

# ------------------------------
# 6. FORECASTING
# ------------------------------
print("Forecasting future values...")

future_steps = 12
future_index = np.arange(len(df), len(df) + future_steps).reshape(-1, 1)

future_predictions = model.predict(future_index)

last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=future_steps+1, freq='M')[1:]

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Temperature': future_predictions
})

# ------------------------------
# 7. VISUALIZATION
# ------------------------------
print("Generating plots...")

plt.figure(figsize=(14, 7))

plt.plot(df['Date'], df['Temperature'], label='Temperature', alpha=0.6)
plt.plot(df['Date'], df['Rolling_Mean'], label='Rolling Mean', linewidth=3)
plt.plot(df['Date'], df['Trend_Line'], label='Trend Line (ML)', linestyle='--')

plt.scatter(anomalies['Date'], anomalies['Temperature'],
            color='red', label='Anomalies', s=80)

plt.plot(forecast_df['Date'], forecast_df['Predicted_Temperature'],
         label='Forecast (Next 12 Months)', linestyle='dotted', linewidth=3)

plt.title("Climate Trend Analysis", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()

# SAVE GRAPH
plt.savefig("outputs/climate_trend.png")

plt.show()

# ------------------------------
# 8. SAVE DATA
# ------------------------------
print("Saving outputs...")

df.to_csv("outputs/processed_data.csv", index=False)
forecast_df.to_csv("outputs/forecast.csv", index=False)

# ------------------------------
# 9. INSIGHTS
# ------------------------------
print("\n🔍 KEY INSIGHTS:")

avg_temp = df['Temperature'].mean()
max_temp = df['Temperature'].max()
min_temp = df['Temperature'].min()

print(f"Average Temperature: {avg_temp:.2f} °C")
print(f"Max Temperature: {max_temp:.2f} °C")
print(f"Min Temperature: {min_temp:.2f} °C")
print(f"Total Anomalies Detected: {len(anomalies)}")

if df['Trend_Line'].iloc[-1] > df['Trend_Line'].iloc[0]:
    print("📈 Overall Trend: Increasing Temperature (Warming)")
else:
    print("📉 Overall Trend: Decreasing Temperature")

print("\n✅ Project executed successfully!")