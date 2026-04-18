import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide"
)

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv("outputs/processed_data.csv")
forecast_df = pd.read_csv("outputs/forecast.csv")

df['Date'] = pd.to_datetime(df['Date'])
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

# ------------------------------
# TITLE
# ------------------------------
st.title("🌍 Climate Trend Analyzer Dashboard")
st.markdown("### Data-driven insights into climate patterns & future trends")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("🔧 Filters")

year = st.sidebar.selectbox("Select Year", sorted(df['Year'].unique()))
filtered_df = df[df['Year'] == year]

# ------------------------------
# METRICS (THIS IS 🔥 UI)
# ------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Temp (°C)", f"{filtered_df['Temperature'].mean():.2f}")
col2.metric("Max Temp (°C)", f"{filtered_df['Temperature'].max():.2f}")
col3.metric("Min Temp (°C)", f"{filtered_df['Temperature'].min():.2f}")
col4.metric("Anomalies", f"{filtered_df['Anomaly'].sum()}")

# ------------------------------
# MAIN GRAPH
# ------------------------------
st.subheader("📈 Climate Trend Analysis")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(filtered_df['Date'], filtered_df['Temperature'], label='Temperature')
ax.plot(filtered_df['Date'], filtered_df['Rolling_Mean'], label='Trend', linewidth=2)
ax.plot(filtered_df['Date'], filtered_df['Advanced_Trend'], label='ML Trend', linestyle='--')

# anomalies
anomalies = filtered_df[filtered_df['Anomaly'] == True]
ax.scatter(anomalies['Date'], anomalies['Temperature'],
           color='red', label='Anomalies', s=70)

# forecast
ax.plot(forecast_df['Date'], forecast_df['Predicted_Temperature'],
        label='Forecast', linestyle='dotted')

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid()

st.pyplot(fig)

# ------------------------------
# CORRELATION HEATMAP
# ------------------------------
st.subheader("🔥 Feature Correlation")

corr = df[['Temperature', 'Rainfall', 'CO2']].corr()

fig2, ax2 = plt.subplots()
cax = ax2.matshow(corr)
fig2.colorbar(cax)

ax2.set_xticks(range(len(corr.columns)))
ax2.set_yticks(range(len(corr.columns)))

ax2.set_xticklabels(corr.columns)
ax2.set_yticklabels(corr.columns)

st.pyplot(fig2)

# ------------------------------
# DATA TABLE
# ------------------------------
st.subheader("📋 Data Table")
st.dataframe(filtered_df)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("🚀 Built with Python, ML & Streamlit")