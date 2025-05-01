import pandas as pd
import numpy as np

def convert_mag_and_type(row):
    mag = row['mag']
    mag_type = str(row['magType']).lower() if pd.notnull(row['magType']) else None

    if pd.isnull(mag):
        return pd.Series([np.nan, np.nan])

    if mag_type in ['mw', 'mwr', 'mww']:
        return pd.Series([round(mag, 1), 'mw'])
    elif mag_type == 'mb':
        return pd.Series([round(0.85 * mag + 1.03, 1), 'mw'])
    elif mag_type == 'ms':
        return pd.Series([round(0.67 * mag + 2.07, 1), 'mw'])
    elif mag_type == 'ml':
        return pd.Series([round(0.85 * mag + 0.73, 1), 'mw'])
    else:
        return pd.Series([np.nan, np.nan])

# 讀取資料
df = pd.read_csv('data/origin.csv')

# 套用轉換
df[['mag', 'magType']] = df.apply(convert_mag_and_type, axis=1)

# 存檔
df.to_csv('data/origin_with_mw.csv', index=False)
