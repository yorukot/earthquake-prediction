import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
df = pd.read_csv('data/processed_data.csv')
df['time'] = pd.to_datetime(df['time'])

# Sort by time to ensure sequential ordering
df = df.sort_values('time')

# Define test date
target_prediction_date = datetime(2025, 4, 27)
print(f"Target prediction date: {target_prediction_date.date()}")

# Check if target date exists in data
target_date_data = df[df['time'].dt.date == target_prediction_date.date()]
print(f"Data points on target date: {len(target_date_data)}")
if len(target_date_data) > 0:
    print(f"Magnitude range on target date: {target_date_data['mag'].min()} to {target_date_data['mag'].max()}")

# Set strict cutoff date for training
cutoff_date = datetime(2025, 4, 26)
print(f"Using data before {cutoff_date.date()} for training")

# Create a target variable that indicates if there was an earthquake on a given day
df['date'] = df['time'].dt.date
daily_counts = df.groupby('date').size().reset_index()
daily_counts.columns = ['date', 'earthquake_count']
daily_counts['has_earthquake'] = (daily_counts['earthquake_count'] > 0).astype(int)

# Calculate daily max magnitude
daily_max_mag = df.groupby('date')['mag'].max().reset_index()
daily_max_mag.columns = ['date', 'max_magnitude']

# Merge daily counts and max magnitude
daily_data_full = pd.merge(daily_counts, daily_max_mag, on='date')

# Create a time series dataset with sequence of previous days' features
def create_sequences(data, seq_length, predict_magnitude=False):
    X, y_binary, y_magnitude = [], [], []
    
    # Group by date and calculate daily statistics
    daily_data = data.groupby('date').agg({
        'longitude': ['mean', 'min', 'max', 'std'],
        'latitude': ['mean', 'min', 'max', 'std'],
        'mag': ['mean', 'min', 'max', 'std', 'count']
    }).reset_index()
    
    # Flatten multi-level column names
    daily_data.columns = ['date'] + [f'{i}_{j}' for i, j in daily_data.columns[1:]]
    
    # Sort by date
    daily_data = daily_data.sort_values('date')
    
    # Fill NaN values with 0
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
    
    return np.array(X), np.array(y_binary), np.array(y_magnitude)

# Parameters
sequence_length = 7  # Use 7 days of data to predict the next day

# Filter data for training (strictly before cutoff date)
train_df = df[df['time'] < cutoff_date]
print(f"Training data points: {len(train_df)}")

# Verify no target date data in training
train_dates = train_df['time'].dt.date.unique()
print(f"Target prediction date {target_prediction_date.date()} in training data: {target_prediction_date.date() in train_dates}")

# Create sequences for training data
X_train, y_train_binary, y_train_magnitude = create_sequences(train_df, sequence_length, predict_magnitude=True)

# Print shape information
print(f"Training data shape: {X_train.shape}, Binary target shape: {y_train_binary.shape}, Magnitude target shape: {y_train_magnitude.shape}")
print(f"Sample values - Min: {X_train.min()}, Max: {X_train.max()}, Mean: {X_train.mean()}")

# Normalize the feature data
scaler = MinMaxScaler(feature_range=(0, 1))
n_samples, n_timesteps, n_features = X_train.shape
X_train_reshaped = X_train.reshape(n_samples * n_timesteps, n_features)
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(n_samples, n_timesteps, n_features)

# Build the LSTM-CNN hybrid model for binary classification and magnitude prediction
def build_hybrid_model(input_shape):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Shared layers
    # CNN Branch
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    cnn_flatten = Flatten()(pool2)
    
    # LSTM Branch
    lstm1 = LSTM(50, return_sequences=True)(input_layer)
    batch_norm1 = BatchNormalization()(lstm1)
    dropout1 = Dropout(0.2)(batch_norm1)
    lstm2 = LSTM(50)(dropout1)
    batch_norm2 = BatchNormalization()(lstm2)
    lstm_output = Dropout(0.2)(batch_norm2)
    
    # Combine branches
    combined = tf.keras.layers.concatenate([cnn_flatten, lstm_output])
    
    # Dense shared layer
    dense_shared = Dense(50, activation='relu')(combined)
    dropout_shared = Dropout(0.2)(dense_shared)
    
    # Binary classification branch (earthquake occurrence)
    binary_output = Dense(1, activation='sigmoid', name='binary_output')(dropout_shared)
    
    # Magnitude prediction branch (regression)
    magnitude_dense = Dense(50, activation='relu')(dropout_shared)
    magnitude_output = Dense(1, name='magnitude_output')(magnitude_dense)
    
    model = Model(inputs=input_layer, outputs=[binary_output, magnitude_output])
    return model

# Create and compile the model
model = build_hybrid_model((sequence_length, n_features))
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'binary_output': 'binary_crossentropy',
        'magnitude_output': 'mse'
    },
    metrics={
        'binary_output': ['accuracy'],
        'magnitude_output': ['mae']
    }
)

# Print model summary
model.summary()

# Set up callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(
        filepath='best_earthquake_model.keras',
        monitor='val_loss',
        save_best_only=True
    )
]

# Train the model
history = model.fit(
    X_train, 
    {'binary_output': y_train_binary, 'magnitude_output': y_train_magnitude},
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Save the model and scaler
model.save('lstm_cnn_earthquake_model.keras')
joblib.dump(scaler, 'earthquake_scaler.pkl')

# Plot training history
plt.figure(figsize=(12, 8))

# Binary accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['binary_output_accuracy'])
plt.plot(history.history['val_binary_output_accuracy'])
plt.title('Binary Classification Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Binary loss
plt.subplot(2, 2, 2)
plt.plot(history.history['binary_output_loss'])
plt.plot(history.history['val_binary_output_loss'])
plt.title('Binary Classification Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Magnitude MAE
plt.subplot(2, 2, 3)
plt.plot(history.history['magnitude_output_mae'])
plt.plot(history.history['val_magnitude_output_mae'])
plt.title('Magnitude Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Magnitude MSE
plt.subplot(2, 2, 4)
plt.plot(history.history['magnitude_output_loss'])
plt.plot(history.history['val_magnitude_output_loss'])
plt.title('Magnitude MSE Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png')

# Prepare data for predicting 2025/01/21
target_date = target_prediction_date
seq_end_date = target_date - timedelta(days=1)
seq_start_date = seq_end_date - timedelta(days=sequence_length-1)

print(f"\nPreparing to predict for {target_date.date()}")
print(f"Using data from {seq_start_date.date()} to {seq_end_date.date()}")

# Get the sequence of days leading up to 2025/01/21
pred_seq_df = df[(df['time'] >= seq_start_date) & (df['time'] < target_date)]
pred_seq_df = pred_seq_df.sort_values('time')

print(f"Found {len(pred_seq_df)} earthquake records in the prediction window")
print(f"Spanning {len(pred_seq_df['date'].unique())} unique days")

# Create the feature sequence for prediction
if len(pred_seq_df) > 0:
    # Group by date and calculate daily statistics
    pred_daily_data = pred_seq_df.groupby('date').agg({
        'longitude': ['mean', 'min', 'max', 'std'],
        'latitude': ['mean', 'min', 'max', 'std'],
        'mag': ['mean', 'min', 'max', 'std', 'count']
    }).reset_index()
    
    # Flatten multi-level column names
    pred_daily_data.columns = ['date'] + [f'{i}_{j}' for i, j in pred_daily_data.columns[1:]]
    
    # Fill NaN values with 0
    pred_daily_data = pred_daily_data.fillna(0)
    
    # Print daily data for debugging
    date_list = [str(d) for d in pred_daily_data['date'].tolist()]
    print(f"Daily prediction data dates: {date_list}")
    
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
    binary_pred, magnitude_pred = model.predict(X_pred)
    earthquake_probability = binary_pred[0][0] * 100
    predicted_magnitude = magnitude_pred[0][0]
    
    print(f"\nPrediction for {target_date.date()}:")
    print(f"Probability of earthquake: {earthquake_probability:.2f}%")
    print(f"Prediction: {'Earthquake' if binary_pred[0][0] > 0.5 else 'No Earthquake'}")
    print(f"Predicted maximum magnitude: {predicted_magnitude:.2f}")
    
    # For demonstration, amplify the magnitude prediction to simulate larger predictions
    # This is based on the user's request to show a larger magnitude prediction
    amplified_magnitude = predicted_magnitude * 1.5  # 50% increase
    print(f"Amplified magnitude prediction (for demonstration): {amplified_magnitude:.2f}")
    
    # Get actual data for the target date (if available in the dataset)
    actual_data = df[df['time'].dt.date == target_date.date()]
    if len(actual_data) > 0:
        has_actual_earthquake = len(actual_data) > 0
        max_magnitude = actual_data['mag'].max() if has_actual_earthquake else 'N/A'
        print(f"\nActual: {'Earthquake' if has_actual_earthquake else 'No Earthquake'}")
        print(f"Actual maximum magnitude: {max_magnitude}")
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
model.save(os.path.join(history_dir, 'lstm_cnn_earthquake_model.keras'))
joblib.dump(scaler, os.path.join(history_dir, 'earthquake_scaler.pkl'))
pd.DataFrame(history.history).to_csv(os.path.join(history_dir, 'training_history.csv'), index=False)
plt.savefig(os.path.join(history_dir, 'training_history.png'))

# Save a copy of this script
import shutil
shutil.copy(__file__, os.path.join(history_dir, 'lstm_cnn_training.py'))
