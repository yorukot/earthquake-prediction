import pandas as pd

# 讀取資料
df = pd.read_csv('data/origin_with_mw.csv')

# 只保留 time, latitude, longitude, depth, mag
df = df[['time', 'latitude', 'longitude', 'depth', 'mag']]

# 存檔
df.to_csv('data/origin_mag_only.csv', index=False)
