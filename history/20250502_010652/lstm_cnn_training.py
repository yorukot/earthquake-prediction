import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input, Model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
df = pd.read_csv('data/processed_data.csv')
df['time'] = pd.to_datetime(df['time'])

# Sort by time to ensure sequential ordering
df = df.sort_values('time')

# Create a target variable that indicates if there was an earthquake on a given day
df['date'] = df['time'].dt.date
daily_counts = df.groupby('date').size().reset_index()
daily_counts.columns = ['date', 'earthquake_count']
daily_counts['has_earthquake'] = (daily_counts['earthquake_count'] > 0).astype(int)

# Create a time series dataset with sequence of previous days' features
def create_sequences(data, seq_length):
    X, y = [], []
    
    # Group by date and calculate daily statistics
    daily_data = data.groupby('date').agg({
        'longitude': ['mean', 'min', 'max', 'std'],
        'latitude': ['mean', 'min', 'max', 'std'],
        'mag': ['mean', 'min', 'max', 'std', 'count']
    }).reset_index()
    
    # Flatten multi-level column names
    daily_data.columns = ['date'] + [f'{i}_{j}' for i, j in daily_data.columns[1:]]
    
    # Convert date to pandas datetime
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    # Sort by date
    daily_data = daily_data.sort_values('date')
    
    # Create sequences
    for i in range(len(daily_data) - seq_length):
        # Get sequence of past days
        seq = daily_data.iloc[i:i+seq_length, 1:].values  # Exclude the date column
        
        # Get target for next day (whether there's an earthquake or not)
        target_date = daily_data.iloc[i+seq_length, 0]  # Get the date after the sequence
        target_has_earthquake = daily_counts[daily_counts['date'] == target_date.date()]['has_earthquake'].values
        
        if len(target_has_earthquake) > 0:
            X.append(seq)
            y.append(target_has_earthquake[0])
    
    return np.array(X), np.array(y)

# Parameters
sequence_length = 14  # Use 14 days of data to predict the next day
cutoff_date = datetime(2025, 1, 20)  # Train only on data before this date

# Filter data for training (before cutoff date)
train_df = df[df['time'] < cutoff_date]

# Create sequences for training data
X_train, y_train = create_sequences(train_df, sequence_length)

# Normalize the feature data
scaler = MinMaxScaler()
n_samples, n_timesteps, n_features = X_train.shape
X_train_reshaped = X_train.reshape(n_samples * n_timesteps, n_features)
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(n_samples, n_timesteps, n_features)

# Build the LSTM-CNN hybrid model
def build_hybrid_model(input_shape):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # CNN Branch
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    cnn_flatten = Flatten()(pool2)
    
    # LSTM Branch
    lstm1 = LSTM(100, return_sequences=True)(input_layer)
    batch_norm1 = BatchNormalization()(lstm1)
    dropout1 = Dropout(0.3)(batch_norm1)
    lstm2 = LSTM(100)(dropout1)
    batch_norm2 = BatchNormalization()(lstm2)
    lstm_output = Dropout(0.3)(batch_norm2)
    
    # Combine branches
    combined = tf.keras.layers.concatenate([cnn_flatten, lstm_output])
    
    # Dense layers for classification
    dense1 = Dense(100, activation='relu')(combined)
    dropout3 = Dropout(0.3)(dense1)
    output = Dense(1, activation='sigmoid')(dropout3)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

# Create and compile the model
model = build_hybrid_model((sequence_length, n_features))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Set up callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(
        filepath='best_earthquake_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Save the model and scaler
model.save('lstm_cnn_earthquake_model.h5')
joblib.dump(scaler, 'earthquake_scaler.pkl')

# Prepare data for predicting 2025/01/21
target_date = datetime(2025, 1, 21)
seq_end_date = target_date - timedelta(days=1)
seq_start_date = seq_end_date - timedelta(days=sequence_length-1)

# Get the sequence of days leading up to 2025/01/21
pred_seq_df = df[(df['time'] >= seq_start_date) & (df['time'] <= seq_end_date)]
pred_seq_df = pred_seq_df.sort_values('time')

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
    
    # Check if we have enough days in our sequence
    if len(pred_daily_data) == sequence_length:
        X_pred = pred_daily_data.iloc[:, 1:].values  # Exclude the date column
        X_pred = X_pred.reshape(1, sequence_length, n_features)
        
        # Normalize the prediction data
        X_pred_reshaped = X_pred.reshape(sequence_length, n_features)
        X_pred_scaled = scaler.transform(X_pred_reshaped)
        X_pred = X_pred_scaled.reshape(1, sequence_length, n_features)
        
        # Make prediction
        prediction = model.predict(X_pred)[0][0]
        earthquake_probability = prediction * 100
        
        print(f"Prediction for {target_date.date()}:")
        print(f"Probability of earthquake: {earthquake_probability:.2f}%")
        print(f"Prediction: {'Earthquake' if prediction > 0.5 else 'No Earthquake'}")
        
        # Get actual data for the target date (if available)
        actual_data = df[df['time'].dt.date == target_date.date()]
        if len(actual_data) > 0:
            has_actual_earthquake = len(actual_data) > 0
            max_magnitude = actual_data['mag'].max() if has_actual_earthquake else 'N/A'
            print(f"Actual: {'Earthquake' if has_actual_earthquake else 'No Earthquake'}")
            print(f"Actual maximum magnitude: {max_magnitude}")
    else:
        print(f"Not enough data for prediction. Need {sequence_length} days, but only have {len(pred_daily_data)} days.")
else:
    print("No data available for prediction sequence.")

# Save training history and timestamp to history folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
history_dir = os.path.join('history', timestamp)
os.makedirs(history_dir, exist_ok=True)

# Save model and training history
model.save(os.path.join(history_dir, 'lstm_cnn_earthquake_model.h5'))
joblib.dump(scaler, os.path.join(history_dir, 'earthquake_scaler.pkl'))
pd.DataFrame(history.history).to_csv(os.path.join(history_dir, 'training_history.csv'), index=False)

# Save a copy of this script
import shutil
shutil.copy(__file__, os.path.join(history_dir, 'lstm_cnn_training.py'))
