import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Set non-interactive backend
matplotlib.use('Agg')

def load_and_prepare_data(file_path, target_column='consumption_scaled', seq_len=10):
    """Loads and prepares data for the LSTM model."""
    data = pd.read_csv(file_path)

    # Add time-based features if available
    if 'hour' in data.columns:
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    if 'day_of_week' in data.columns:
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    # Handle NaN and infinite values
    data.dropna(inplace=True)
    data = data[np.isfinite(data.select_dtypes(include=[np.number])).all(axis=1)]
    print("Initial Data Info:", data.info())

    # Extract and scale the target column
    target_data = data[target_column].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target_data.reshape(-1, 1))

    # Create sequences for the LSTM model
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Plot training history
def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Training Loss Over Epochs')
    plt.legend()
    plt.savefig('results/training_loss_optimized.png')
    print("Training loss plot saved as 'training_loss_optimized.png'.")

# Main function
def main():
    data_path = 'data/processed_data.csv'
    print("Loading and preparing data...")
    X, y, scaler = load_and_prepare_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Building the LSTM model...")
    model = build_lstm_model((X_train.shape[1], 1))

    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('models/lstm_best_model.keras', monitor='loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0001)

    print("Training the model...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=[early_stopping, checkpoint, reduce_lr])

    plot_history(history)
    model.save('models/lstm_model_optimized.h5')
    print("Model saved to 'models/lstm_model_output.h5'")

    print("Evaluating the model...")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    print(f"Root Mean Squared Error: {rmse:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_scaled, color='blue', label='Actual Consumption')
    plt.plot(predictions, color='red', label='Predicted Consumption')
    plt.xlabel('Time')
    plt.ylabel('Electricity Consumption')
    plt.legend()
    plt.savefig('results/predictions_vs_actual_optimized.png')
    print("Predictions vs Actual plot saved as 'predictions_vs_actual_optimized.png'.")

if __name__ == "__main__":
    main()
