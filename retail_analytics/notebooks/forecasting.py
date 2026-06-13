import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv('data/cleaned_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print("✅ Data Loaded!")

# ================================
# 2. PREPARE TIME SERIES
# ================================
daily = df.groupby('Date')['Revenue'].sum().reset_index()
daily = daily.sort_values('Date')

# Feature Engineering
daily['Day'] = daily['Date'].dt.day
daily['Month'] = daily['Date'].dt.month
daily['Year'] = daily['Date'].dt.year
daily['Day_of_Week'] = daily['Date'].dt.dayofweek
daily['Quarter'] = daily['Date'].dt.quarter
daily['Lag_7'] = daily['Revenue'].shift(7)
daily['Lag_30'] = daily['Revenue'].shift(30)
daily['Rolling_7'] = daily['Revenue'].rolling(7).mean()
daily['Rolling_30'] = daily['Revenue'].rolling(30).mean()
daily.dropna(inplace=True)

print("✅ Features created!")

# ================================
# 3. TRAIN MODEL
# ================================
features = ['Day','Month','Year','Day_of_Week',
            'Quarter','Lag_7','Lag_30',
            'Rolling_7','Rolling_30']

X = daily[features]
y = daily['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

model = XGBRegressor(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ================================
# 4. EVALUATE MODEL
# ================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n📊 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# ================================
# 5. ACTUAL VS PREDICTED CHART
# ================================
plt.figure(figsize=(14,6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Days')
plt.ylabel('Revenue')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/forecast.png')
plt.show()
print("✅ Forecast chart saved!")

# ================================
# 6. PREDICT NEXT 90 DAYS
# ================================
last_date = daily['Date'].max()
future_dates = pd.date_range(start=last_date+pd.Timedelta(days=1), periods=90)

future = pd.DataFrame()
future['Date'] = future_dates
future['Day'] = future['Date'].dt.day
future['Month'] = future['Date'].dt.month
future['Year'] = future['Date'].dt.year
future['Day_of_Week'] = future['Date'].dt.dayofweek
future['Quarter'] = future['Date'].dt.quarter
future['Lag_7'] = daily['Revenue'].iloc[-7:].mean()
future['Lag_30'] = daily['Revenue'].iloc[-30:].mean()
future['Rolling_7'] = daily['Revenue'].iloc[-7:].mean()
future['Rolling_30'] = daily['Revenue'].iloc[-30:].mean()

future_pred = model.predict(future[features])

# ================================
# 7. FUTURE FORECAST CHART
# ================================
plt.figure(figsize=(14,6))
plt.plot(daily['Date'], daily['Revenue'], label='Historical', color='blue')
plt.plot(future_dates, future_pred, label='90-Day Forecast', color='green', linestyle='--')
plt.title('90-Day Revenue Forecast')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/future_forecast.png')
plt.show()
print("✅ Future forecast chart saved!")

# Save forecast
future['Predicted_Revenue'] = future_pred
future.to_csv('outputs/forecast_90days.csv', index=False)
print("\n🚀 Phase 3 Complete! Forecasting Done!")