"""
Unified 1D CNN Inference Script
================================
Load the trained unified CNN model and make predictions on new sensor data
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import joblib
from datetime import datetime


class UnifiedCNNPredictor:
    """Wrapper class for making predictions with the trained unified CNN model"""

    def __init__(self, model_dir="artifacts/unified_cnn"):
        """
        Initialize predictor by loading model and artifacts

        Args:
            model_dir: Directory containing saved model and artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.config = None
        self.scalers = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, configuration, and scalers"""
        print(f"Loading model artifacts from {self.model_dir}...")

        # Load model
        model_path = os.path.join(self.model_dir, 'unified_cnn_model.keras')
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, 'best_model.keras')

        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")

        # Load configuration
        config_path = os.path.join(self.model_dir, 'model_config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print(f"✓ Configuration loaded")

        # Load scalers
        scalers_path = os.path.join(self.model_dir, 'scalers.pkl')
        self.scalers = joblib.load(scalers_path)
        print(f"✓ Scalers loaded")

        print(f"Model expects {self.config['n_sensors']} sensors with window size {self.config['window_size']}")

    def preprocess_dataframe(self, df):
        """
        Preprocess raw sensor data into the format expected by the model

        Args:
            df: DataFrame with columns [timestamp, sensor_type, sensor_id, value]

        Returns:
            Normalized pivot table
        """
        # Create sensor key
        df['sensor_key'] = df['sensor_type'] + '_' + df['sensor_id']

        # Pivot
        pivot = df.pivot_table(
            index='timestamp',
            columns='sensor_key',
            values='value',
            aggfunc='first'
        )

        # Ensure we have all expected sensors
        expected_sensors = self.config['sensor_columns']
        for sensor in expected_sensors:
            if sensor not in pivot.columns:
                pivot[sensor] = np.nan

        # Reorder columns to match training
        pivot = pivot[expected_sensors]

        # Fill missing values
        pivot = pivot.ffill().bfill().fillna(0)

        # Normalize using saved scalers
        normalized = pivot.copy()
        for col in pivot.columns:
            if col in self.scalers:
                normalized[col] = self.scalers[col].transform(pivot[[col]])

        return normalized

    def create_sequences(self, data, window_size=None, stride=None):
        """
        Create sliding window sequences from data

        Args:
            data: DataFrame or numpy array
            window_size: Window size (uses config default if None)
            stride: Stride for sliding window (defaults to 1)

        Returns:
            numpy array of sequences
        """
        if window_size is None:
            window_size = self.config['window_size']
        if stride is None:
            stride = 1

        if isinstance(data, pd.DataFrame):
            data = data.values

        sequences = []
        for i in range(0, len(data) - window_size + 1, stride):
            sequences.append(data[i:i+window_size])

        return np.array(sequences)

    def predict(self, df, return_proba=True, stride=1):
        """
        Make anomaly predictions on sensor data

        Args:
            df: DataFrame with sensor data
            return_proba: If True, return probabilities; else return binary predictions
            stride: Stride for creating sequences

        Returns:
            predictions: Array of predictions
            timestamps: Corresponding timestamps (end of each window)
        """
        # Preprocess
        normalized = self.preprocess_dataframe(df)

        # Create sequences
        X = self.create_sequences(normalized, stride=stride)

        if len(X) == 0:
            return np.array([]), np.array([])

        # Predict
        predictions = self.model.predict(X, verbose=0).flatten()

        if not return_proba:
            predictions = (predictions > 0.5).astype(int)

        # Get timestamps (end of each window)
        timestamps = normalized.index[self.config['window_size']-1::stride][:len(predictions)]

        return predictions, timestamps

    def predict_single_window(self, df):
        """
        Predict on a single window of data

        Args:
            df: DataFrame with exactly window_size rows of sensor data

        Returns:
            probability: Anomaly probability
        """
        if len(df) != self.config['window_size']:
            raise ValueError(f"Expected {self.config['window_size']} rows, got {len(df)}")

        normalized = self.preprocess_dataframe(df)
        X = normalized.values.reshape(1, self.config['window_size'], -1)

        probability = self.model.predict(X, verbose=0)[0][0]

        return float(probability)


def main():
    """Example usage of the predictor"""
    print("="*70)
    print("UNIFIED CNN PREDICTOR - DEMO")
    print("="*70)

    # Initialize predictor
    predictor = UnifiedCNNPredictor()

    # Load test data
    print("\nLoading test data...")
    df = pd.read_csv("data/ipmb_5sensors_30min_1_to_10hz.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Take a subset for demo (e.g., first 5000 rows)
    df_test = df.head(5000).copy()

    print(f"Test data: {len(df_test)} rows")

    # Make predictions
    print("\nMaking predictions...")
    predictions, timestamps = predictor.predict(df_test, return_proba=True, stride=10)

    print(f"Generated {len(predictions)} predictions")

    # Show results
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'anomaly_probability': predictions,
        'prediction': (predictions > 0.5).astype(int)
    })

    print("\nFirst 10 predictions:")
    print(results_df.head(10))

    print("\nLast 10 predictions:")
    print(results_df.tail(10))

    print(f"\nTotal anomalies detected: {(results_df['prediction'] == 1).sum()}")
    print(f"Anomaly rate: {(results_df['prediction'] == 1).sum() / len(results_df) * 100:.2f}%")

    # Save predictions
    output_path = "artifacts/unified_cnn/predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    # Show high-confidence anomalies
    anomalies = results_df[results_df['anomaly_probability'] > 0.8].sort_values('anomaly_probability', ascending=False)
    if len(anomalies) > 0:
        print(f"\nHigh-confidence anomalies (probability > 0.8): {len(anomalies)}")
        print(anomalies.head(10))


if __name__ == "__main__":
    main()
