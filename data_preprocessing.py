from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 處理缺失值 (填補或刪除)
df = df.dropna()  # 這裡選擇刪除有缺失的行

# 分離特徵和目標
X = df.drop(columns=['mag'])  # 特徵 (刪除 mag 欄位)
y = df['mag']  # 目標 (預測 mag)

# 類別編碼 (對於類別變數如 magType)
encoder = LabelEncoder()
X['magType'] = encoder.fit_transform(X['magType'])

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
