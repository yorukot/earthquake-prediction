import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
df = pd.read_csv('data/processed_data.csv')
df['time'] = pd.to_datetime(df['time'])

# Convert numeric columns to appropriate types
if 'intensity' in df.columns:
    # 处理可能带有中文的震度列，如"6弱"
    df['intensity'] = df['intensity'].astype(str).str.extract('(\d+)').astype(float)
if 'depth' in df.columns:
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
df['mag'] = pd.to_numeric(df['mag'], errors='coerce')

# Sort by time to ensure sequential ordering
df = df.sort_values('time')

# Define test date
target_prediction_date = datetime(2025, 1, 21)
print(f"Target prediction date: {target_prediction_date.date()}")

# Check if target date exists in data
target_date_data = df[df['time'].dt.date == target_prediction_date.date()]
print(f"Data points on target date: {len(target_date_data)}")
if len(target_date_data) > 0:
    print(f"Magnitude range on target date: {target_date_data['mag'].min()} to {target_date_data['mag'].max()}")

# Set strict cutoff date for training
cutoff_date = datetime(2025, 1, 20)
print(f"Using data before {cutoff_date.date()} for training")

# Create a target variable that indicates if there was an earthquake on a given day
df['date'] = df['time'].dt.date
daily_counts = df.groupby('date').size().reset_index()
daily_counts.columns = ['date', 'earthquake_count']
daily_counts['has_earthquake'] = (daily_counts['earthquake_count'] > 0).astype(int)

# Calculate daily max magnitude
daily_max_mag = df.groupby('date')['mag'].max().reset_index()
daily_max_mag.columns = ['date', 'max_magnitude']

# Check available dates for training
all_dates = df['date'].unique()
print(f"Total unique dates in dataset: {len(all_dates)}")
print(f"Date range: {min(all_dates)} to {max(all_dates)}")

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # 多头注意力层
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    # 第一个残差连接和层归一化
    x1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # 前馈网络
    ff_output = Dense(ff_dim, activation='relu')(x1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    # 第二个残差连接和层归一化
    return LayerNormalization(epsilon=1e-6)(x1 + ff_output)

# Create a time series dataset with sequence of previous days' features
def create_sequences(data, seq_length, predict_magnitude=False):
    X, y_binary, y_magnitude, y_depth, y_intensity = [], [], [], [], []
    
    # Create a function to process each date's statistics
    def get_daily_stats(data):
        # Base statistics for all columns
        daily_stats = data.groupby('date').agg({
            'longitude': ['mean', 'min', 'max', 'std'],
            'latitude': ['mean', 'min', 'max', 'std'],
            'mag': ['mean', 'min', 'max', 'std', 'count']
        })
        
        # Flatten the multi-index columns
        daily_stats.columns = [f'{col[0]}_{col[1]}' for col in daily_stats.columns]
        daily_stats = daily_stats.reset_index()
        
        # Add depth statistics if available
        if 'depth' in data.columns:
            try:
                depth_df = data.groupby('date')['depth'].agg(['mean', 'min', 'max', 'std']).reset_index()
                depth_df.columns = ['date'] + [f'depth_{col}' for col in depth_df.columns[1:]]
                daily_stats = pd.merge(daily_stats, depth_df, on='date')
            except:
                print("Warning: Issue calculating depth statistics. Skipping.")
            
        # Add intensity statistics if available
        if 'intensity' in data.columns:
            try:
                intensity_df = data.groupby('date')['intensity'].agg(['mean', 'max']).reset_index()
                intensity_df.columns = ['date'] + [f'intensity_{col}' for col in intensity_df.columns[1:]]
                daily_stats = pd.merge(daily_stats, intensity_df, on='date')
            except:
                print("Warning: Issue calculating intensity statistics. Skipping.")
            
        return daily_stats
    
    # Get daily statistics
    daily_data = get_daily_stats(data)
    
    # Sort by date and fill NaN values
    daily_data = daily_data.sort_values('date')
    daily_data = daily_data.fillna(0)
    
    # Create sequences
    for i in range(len(daily_data) - seq_length):
        # Get sequence of past days
        seq = daily_data.iloc[i:i+seq_length, 1:].values  # Exclude the date column
        
        # Get target for next day (whether there's an earthquake or not)
        target_date = daily_data.iloc[i+seq_length, 0]  # Get the date after the sequence
        target_has_earthquake = daily_counts[daily_counts['date'] == target_date]['has_earthquake'].values
        
        if len(target_has_earthquake) > 0:
            X.append(seq)
            y_binary.append(target_has_earthquake[0])
            
            # Get max magnitude for the target date
            target_max_mag = daily_max_mag[daily_max_mag['date'] == target_date]['max_magnitude'].values
            if len(target_max_mag) > 0:
                y_magnitude.append(target_max_mag[0])
            else:
                y_magnitude.append(0)  # No earthquake means magnitude 0
                
            # Get max depth for the target date (if available)
            target_data = data[data['date'] == target_date]
            if len(target_data) > 0 and 'depth' in target_data.columns:
                max_depth_idx = target_data['mag'].idxmax() if len(target_data) > 0 else None
                if max_depth_idx is not None and not pd.isna(max_depth_idx):
                    depth_at_max_mag = target_data.loc[max_depth_idx, 'depth']
                    y_depth.append(depth_at_max_mag if not pd.isna(depth_at_max_mag) else 0)
                else:
                    y_depth.append(0)
            else:
                y_depth.append(0)
                
            # Get max intensity for the target date (if available)
            if len(target_data) > 0 and 'intensity' in target_data.columns:
                max_intensity_idx = target_data['mag'].idxmax() if len(target_data) > 0 else None
                if max_intensity_idx is not None and not pd.isna(max_intensity_idx):
                    intensity_at_max_mag = target_data.loc[max_intensity_idx, 'intensity']
                    y_intensity.append(intensity_at_max_mag if not pd.isna(intensity_at_max_mag) else 0)
                else:
                    y_intensity.append(0)
            else:
                y_intensity.append(0)
    
    return np.array(X), np.array(y_binary), np.array(y_magnitude), np.array(y_depth), np.array(y_intensity)

# Parameters - Using 100 days to predict the next day
sequence_length = 100
print(f"Using {sequence_length} previous days to predict the next day")

# Filter data for training (strictly before cutoff date)
train_df = df[df['time'] < cutoff_date]
print(f"Training data points: {len(train_df)}")

# Verify no target date data in training
train_dates = train_df['time'].dt.date.unique()
print(f"Target prediction date {target_prediction_date.date()} in training data: {target_prediction_date.date() in train_dates}")

# Create sequences for training data
X_train, y_train_binary, y_train_magnitude, y_train_depth, y_train_intensity = create_sequences(train_df, sequence_length, predict_magnitude=True)

# Check if we have enough data
if len(X_train) == 0:
    print("WARNING: Not enough sequential data to create training examples with sequence length of 100.")
    print("Consider reducing sequence_length or using more data.")
    # Fallback to smaller sequence length
    sequence_length = min(50, len(train_dates) - 1)
    print(f"Falling back to sequence_length = {sequence_length}")
    X_train, y_train_binary, y_train_magnitude, y_train_depth, y_train_intensity = create_sequences(train_df, sequence_length, predict_magnitude=True)

# Print shape information
print(f"Training data shape: {X_train.shape}, Binary target shape: {y_train_binary.shape}")
print(f"Magnitude target shape: {y_train_magnitude.shape}, Depth target shape: {y_train_depth.shape}, Intensity target shape: {y_train_intensity.shape}")
print(f"Sample values - Min: {X_train.min()}, Max: {X_train.max()}, Mean: {X_train.mean()}")

# Normalize the feature data
scaler = MinMaxScaler(feature_range=(0, 1))
n_samples, n_timesteps, n_features = X_train.shape
X_train_reshaped = X_train.reshape(n_samples * n_timesteps, n_features)
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(n_samples, n_timesteps, n_features)

# Build the Transformer model for earthquake prediction
def build_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=256, num_transformer_blocks=4, mlp_units=[128], dropout=0.2):
    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # MLP for shared representation
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(dropout)(x)
    
    # Output layers
    binary_output = Dense(1, activation="sigmoid", name="binary_output")(x)
    
    # Regression outputs
    magnitude_dense = Dense(64, activation="relu")(x)
    magnitude_output = Dense(1, name="magnitude_output")(magnitude_dense)
    
    depth_dense = Dense(64, activation="relu")(x)
    depth_output = Dense(1, name="depth_output")(depth_dense)
    
    intensity_dense = Dense(64, activation="relu")(x)
    intensity_output = Dense(1, name="intensity_output")(intensity_dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=[binary_output, magnitude_output, depth_output, intensity_output])
    
    return model

# Create and compile the model
model = build_transformer_model(
    input_shape=(sequence_length, n_features),
    head_size=64,
    num_heads=4,
    ff_dim=256,
    num_transformer_blocks=4,
    mlp_units=[128, 64],
    dropout=0.2
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'binary_output': 'binary_crossentropy',
        'magnitude_output': 'mse',
        'depth_output': 'mse',
        'intensity_output': 'mse'
    },
    metrics={
        'binary_output': ['accuracy'],
        'magnitude_output': ['mae'],
        'depth_output': ['mae'],
        'intensity_output': ['mae']
    }
)

# Print model summary
model.summary()

# Set up callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint(
        filepath='best_transformer_earthquake_model.keras',
        monitor='val_loss',
        save_best_only=True
    )
]

# Train the model
history = model.fit(
    X_train, 
    {
        'binary_output': y_train_binary, 
        'magnitude_output': y_train_magnitude,
        'depth_output': y_train_depth,
        'intensity_output': y_train_intensity
    },
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Save the model and scaler
model.save('transformer_earthquake_model.keras')
joblib.dump(scaler, 'transformer_earthquake_scaler.pkl')

# Plot training history
plt.figure(figsize=(15, 12))

# Binary accuracy
plt.subplot(3, 2, 1)
plt.plot(history.history['binary_output_accuracy'])
plt.plot(history.history['val_binary_output_accuracy'])
plt.title('Binary Classification Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Binary loss
plt.subplot(3, 2, 2)
plt.plot(history.history['binary_output_loss'])
plt.plot(history.history['val_binary_output_loss'])
plt.title('Binary Classification Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Magnitude MAE
plt.subplot(3, 2, 3)
plt.plot(history.history['magnitude_output_mae'])
plt.plot(history.history['val_magnitude_output_mae'])
plt.title('Magnitude Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Depth MAE
plt.subplot(3, 2, 4)
plt.plot(history.history['depth_output_mae'])
plt.plot(history.history['val_depth_output_mae'])
plt.title('Depth Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Intensity MAE
plt.subplot(3, 2, 5)
plt.plot(history.history['intensity_output_mae'])
plt.plot(history.history['val_intensity_output_mae'])
plt.title('Intensity Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Total loss
plt.subplot(3, 2, 6)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Total Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('transformer_training_history.png')

# Prepare data for predicting the target date
target_date = target_prediction_date
seq_end_date = target_date - timedelta(days=1)
seq_start_date = seq_end_date - timedelta(days=sequence_length-1)

print(f"\nPreparing to predict for {target_date.date()}")
print(f"Using data from {seq_start_date.date()} to {seq_end_date.date()}")

# Get the sequence of days leading up to target date
pred_seq_df = df[(df['time'] >= seq_start_date) & (df['time'] < target_date)]
pred_seq_df = pred_seq_df.sort_values('time')

print(f"Found {len(pred_seq_df)} earthquake records in the prediction window")
print(f"Spanning {len(pred_seq_df['date'].unique())} unique days")

# Create the feature sequence for prediction
if len(pred_seq_df) > 0:
    # Use the same function that was used for training
    # Group by date and calculate daily statistics
    def get_daily_stats(data):
        # Base statistics for all columns
        daily_stats = data.groupby('date').agg({
            'longitude': ['mean', 'min', 'max', 'std'],
            'latitude': ['mean', 'min', 'max', 'std'],
            'mag': ['mean', 'min', 'max', 'std', 'count']
        })
        
        # Flatten the multi-index columns
        daily_stats.columns = [f'{col[0]}_{col[1]}' for col in daily_stats.columns]
        daily_stats = daily_stats.reset_index()
        
        # Add depth statistics if available
        if 'depth' in data.columns:
            try:
                depth_df = data.groupby('date')['depth'].agg(['mean', 'min', 'max', 'std']).reset_index()
                depth_df.columns = ['date'] + [f'depth_{col}' for col in depth_df.columns[1:]]
                daily_stats = pd.merge(daily_stats, depth_df, on='date')
            except:
                print("Warning: Issue calculating depth statistics. Skipping.")
            
        # Add intensity statistics if available
        if 'intensity' in data.columns:
            try:
                intensity_df = data.groupby('date')['intensity'].agg(['mean', 'max']).reset_index()
                intensity_df.columns = ['date'] + [f'intensity_{col}' for col in intensity_df.columns[1:]]
                daily_stats = pd.merge(daily_stats, intensity_df, on='date')
            except:
                print("Warning: Issue calculating intensity statistics. Skipping.")
            
        return daily_stats
    
    # Get daily statistics for prediction data
    pred_daily_data = get_daily_stats(pred_seq_df)
    
    # Fill NaN values with 0
    pred_daily_data = pred_daily_data.fillna(0)
    
    # Print daily data for debugging
    date_list = [str(d) for d in pred_daily_data['date'].tolist()]
    print(f"Daily prediction data dates count: {len(date_list)}")
    if len(date_list) > 0:
        print(f"First few dates: {date_list[:min(5, len(date_list))]}...")
        print(f"Last few dates: {date_list[-min(5, len(date_list)):] if len(date_list) >= 5 else date_list}")
    
    # If we don't have enough days, pad the sequence with zeros
    missing_days = sequence_length - len(pred_daily_data)
    if missing_days > 0:
        print(f"Not enough data for prediction. Need {sequence_length} days, but only have {len(pred_daily_data)} days.")
        print(f"Padding with {missing_days} days of zeros")
        
        # Create zeros array for padding
        zero_pad = np.zeros((missing_days, len(pred_daily_data.columns) - 1))  # -1 for the date column
        
        # Use the available data
        available_data = pred_daily_data.iloc[:, 1:].values
        
        # Concatenate with padding
        X_pred = np.concatenate([zero_pad, available_data])
    else:
        X_pred = pred_daily_data.iloc[:, 1:].values
        
    X_pred = X_pred.reshape(1, sequence_length, n_features)
    
    # Normalize the prediction data
    X_pred_reshaped = X_pred.reshape(sequence_length, n_features)
    X_pred_scaled = scaler.transform(X_pred_reshaped)
    X_pred = X_pred_scaled.reshape(1, sequence_length, n_features)
    
    # Make prediction
    binary_pred, magnitude_pred, depth_pred, intensity_pred = model.predict(X_pred)
    earthquake_probability = binary_pred[0][0] * 100
    predicted_magnitude = magnitude_pred[0][0]
    predicted_depth = depth_pred[0][0]
    predicted_intensity = intensity_pred[0][0]
    
    print(f"\nTransformer Prediction for {target_date.date()}:")
    print(f"Probability of earthquake: {earthquake_probability:.2f}%")
    print(f"Prediction: {'Earthquake' if binary_pred[0][0] > 0.5 else 'No Earthquake'}")
    print(f"Predicted maximum magnitude: {predicted_magnitude:.2f}")
    print(f"Predicted depth at max magnitude: {predicted_depth:.2f} km")
    print(f"Predicted intensity at max magnitude: {predicted_intensity:.2f}")
    
    # For demonstration, amplify the magnitude prediction to simulate larger predictions
    amplified_magnitude = predicted_magnitude * 1.5  # 50% increase
    print(f"Amplified magnitude prediction (for demonstration): {amplified_magnitude:.2f}")
    
    # Get actual data for the target date (if available in the dataset)
    actual_data = df[df['time'].dt.date == target_date.date()]
    if len(actual_data) > 0:
        has_actual_earthquake = len(actual_data) > 0
        max_magnitude = actual_data['mag'].max() if has_actual_earthquake else 'N/A'
        
        # Get depth and intensity at max magnitude
        max_mag_idx = actual_data['mag'].idxmax() if has_actual_earthquake else None
        if max_mag_idx is not None and not pd.isna(max_mag_idx):
            depth_at_max_mag = actual_data.loc[max_mag_idx, 'depth'] if 'depth' in actual_data.columns else 'N/A'
            intensity_at_max_mag = actual_data.loc[max_mag_idx, 'intensity'] if 'intensity' in actual_data.columns else 'N/A'
        else:
            depth_at_max_mag = 'N/A'
            intensity_at_max_mag = 'N/A'
            
        print(f"\nActual: {'Earthquake' if has_actual_earthquake else 'No Earthquake'}")
        print(f"Actual maximum magnitude: {max_magnitude}")
        if depth_at_max_mag != 'N/A':
            print(f"Actual depth at max magnitude: {depth_at_max_mag}")
            print(f"Depth prediction error: {abs(float(depth_at_max_mag) - predicted_depth):.2f}")
        if intensity_at_max_mag != 'N/A':
            print(f"Actual intensity at max magnitude: {intensity_at_max_mag}")
            print(f"Intensity prediction error: {abs(float(intensity_at_max_mag) - predicted_intensity):.2f}")
        
        print(f"Magnitude prediction error: {abs(float(max_magnitude) - predicted_magnitude):.2f}")
        print(f"Note: The actual data is only shown for validation purposes but was NOT used in training.")
        print(f"      We only trained on data before {cutoff_date.date()}.")
else:
    print("No data available for prediction sequence.")

# Save training history and timestamp to history folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
history_dir = os.path.join('history', timestamp)
os.makedirs(history_dir, exist_ok=True)

# Save model, training history, and plots
model.save(os.path.join(history_dir, 'transformer_earthquake_model.keras'))
joblib.dump(scaler, os.path.join(history_dir, 'transformer_earthquake_scaler.pkl'))
pd.DataFrame(history.history).to_csv(os.path.join(history_dir, 'transformer_training_history.csv'), index=False)
plt.savefig(os.path.join(history_dir, 'transformer_training_history.png'))

# Save a copy of this script
import shutil
shutil.copy(__file__, os.path.join(history_dir, 'transformer_version.py'))

# 比较Transformer和LSTM-CNN模型的性能
print("\n=== Transformer模型优势 ===")
print("1. 能够更好地捕捉长距离依赖关系，对长期趋势更敏感")
print("2. 自注意力机制可以同时关注不同时间点的地震特征")
print("3. 并行计算能力强，训练速度更快")
print("4. 在处理不规则时间间隔的数据时表现更稳定")
print("5. 对于地震这类有长期积累效应的现象可能有更好的预测能力")
