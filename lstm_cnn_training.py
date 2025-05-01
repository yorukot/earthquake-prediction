import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# 參數
sequence_length = 7  # 用過去7天預測
feature_cols = ['latitude', 'longitude', 'depth']

# 載入資料
df = pd.read_csv('data/origin_mag_only.csv')
df = df.dropna(subset=['mag'])
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# 特徵標準化
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 建立時序資料
X, y, time_targets = [], [], []
for i in range(len(df) - sequence_length):
    X.append(df[feature_cols].iloc[i:i+sequence_length].values)
    y.append(df['mag'].iloc[i+sequence_length])
    time_targets.append(df['time'].iloc[i+sequence_length])
X = np.array(X)
y = np.array(y)
time_targets = np.array(time_targets)

# 切分訓練/測試集（以2024-08-01為界）
split_time = pd.to_datetime('2024-08-01').tz_localize('UTC')
split_idx = np.where(time_targets < split_time)[0][-1] + 1 if np.any(time_targets < split_time) else 0

X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# LSTM–CNN模型
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(sequence_length, len(feature_cols))),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss='mse')

# 訓練
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 預測
y_pred = model.predict(X_test)
print('測試集最大預測震度:', np.max(y_pred) if len(y_pred) > 0 else '無資料') 