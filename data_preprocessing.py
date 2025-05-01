import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 載入資料
df = pd.read_csv('data/origin.csv')

# 刪除 '編號' 以及 '位置' 欄位
if '編號' in df.columns and '位置' in df.columns:
    df = df.drop(columns=['編號', '位置', '深度', '最大震度'])

# 處理缺失值 (填補或刪除)
df = df.dropna()  # 這裡選擇刪除有缺失的行

# 處理欄位名稱
# 將中文欄位名稱轉換為英文
column_mapping = {
    '地震時間': 'time',
    '經度': 'longitude',
    '緯度': 'latitude',
    '規模': 'mag',
}

# 重命名欄位
df = df.rename(columns=column_mapping)

# 確保有必要的欄位
required_columns = ['time', 'longitude', 'latitude', 'mag']
for col in required_columns:
    if col not in df.columns:
        print(f"警告: 缺少必要欄位 '{col}'")

# 轉換時間欄位為日期時間格式
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])

# 轉換數值欄位確保為數值型
numeric_columns = ['longitude', 'latitude', 'mag']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 再次處理轉換後可能出現的缺失值
df = df.dropna(subset=numeric_columns)

# 保存處理後的資料
df.to_csv('data/processed_data.csv', index=False)

print(f"資料處理完成，共有 {len(df)} 筆資料")
print(f"已儲存為 'data/processed_data.csv'")