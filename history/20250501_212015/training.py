import os
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 載入資料
df = pd.read_csv('data/origin_mag_only.csv')
df = df.dropna(subset=['mag'])  # Drop rows where mag is NaN
X = df.drop(columns=['mag', 'time'])  # 刪除 'mag' 和 'time' 欄位，只保留數值特徵
y = df['mag']  # 設定 'mag' 為目標變數

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 XGBoost 模型
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 計算均方誤差 (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 計算均方根誤差 (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# === Save model and code to timestamped history folder ===
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
history_dir = os.path.join('history', timestamp)
os.makedirs(history_dir, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(history_dir, 'earthquake_magnitude_model.pkl'))

# Save a copy of this script
shutil.copy(__file__, os.path.join(history_dir, 'training.py'))
