import os
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 載入資料
df = pd.read_csv('data/origin_mag_only.csv')
df = df.dropna(subset=['mag'])
df['time'] = pd.to_datetime(df['time'])

# 訓練集：2024-08-01 之前
train_df = df[df['time'] < '2024-08-01']

# 測試集：2024-08-01 ~ 2024-08-31
test_df = df[(df['time'] >= '2024-08-01') & (df['time'] < '2024-09-01')]

X_train = train_df.drop(columns=['mag', 'time'])
y_train = train_df['mag']

X_test = test_df.drop(columns=['mag', 'time'])
y_test = test_df['mag']

# 訓練模型
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 印出8月預測最大震度
print('2024年8月預測最大震度:', np.max(y_pred) if len(y_pred) > 0 else '無資料')

# 計算均方誤差 (MSE)
mse = mean_squared_error(y_test, y_pred) if len(y_test) > 0 else float('nan')
print(f'Mean Squared Error: {mse}')

# 計算均方根誤差 (RMSE)
rmse = np.sqrt(mse) if not np.isnan(mse) else float('nan')
print(f'Root Mean Squared Error: {rmse}')

# === Save model and code to timestamped history folder ===
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
history_dir = os.path.join('history', timestamp)
os.makedirs(history_dir, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(history_dir, 'earthquake_magnitude_model.pkl'))

# Save a copy of this script
shutil.copy(__file__, os.path.join(history_dir, 'training.py'))
