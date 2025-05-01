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
# 不再需要整月最大值，改為每日預測

X_train = train_df.drop(columns=['mag', 'time'])
y_train = train_df['mag']

# 訓練模型
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# 預測 8/1 和 8/2（修正時區問題）
for day in ['2024-08-01', '2024-08-02']:
    day_start = pd.to_datetime(day).tz_localize('UTC')
    day_end = day_start + pd.Timedelta(days=1)
    day_df = df[(df['time'] >= day_start) & (df['time'] < day_end)]
    X_day = day_df.drop(columns=['mag', 'time'])
    y_pred_day = model.predict(X_day) if len(X_day) > 0 else []
    has_big_eq = np.any(y_pred_day > 6) if len(y_pred_day) > 0 else False
    max_pred = np.max(y_pred_day) if len(y_pred_day) > 0 else '無資料'
    print(f'{day} 預測有大於6級地震嗎？ {"有" if has_big_eq else "沒有"}，最大預測震度: {max_pred}')

# 計算均方誤差 (MSE) 及 RMSE 只針對 8 月份所有資料
# 這裡保留原本的測試集計算
# 測試集：2024-08-01 ~ 2024-08-31
test_df = df[(df['time'] >= pd.to_datetime('2024-08-01').tz_localize('UTC')) & (df['time'] < pd.to_datetime('2024-09-01').tz_localize('UTC'))]
X_test = test_df.drop(columns=['mag', 'time'])
y_test = test_df['mag']
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred) if len(y_test) > 0 else float('nan')
print(f'Mean Squared Error: {mse}')
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
