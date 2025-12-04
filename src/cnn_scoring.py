"""
src/cnn_scoring.py
CNN Model Scoring Module
Loads trained 1D CNN models (Detection and Predictive) and computes risk scores
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    import joblib
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN scoring will be disabled.")


class CNNScorer:
    """
    Loads and manages trained 1D CNN models for anomaly detection and prediction.
    Supports two models:
    - Detection Model: Real-time anomaly detection
    - Predictive Model: Future anomaly forecasting with risk scores
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        """
        Initialize CNN scorer with paths to trained models

        Args:
            artifacts_dir: Path to artifacts directory
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN scoring")

        self.artifacts_dir = artifacts_dir
        self.detection_model = None
        self.predictive_model = None
        self.detection_config = None
        self.predictive_config = None
        self.detection_scalers = None
        self.predictive_scalers = None

        # Load models
        self._load_detection_model()
        self._load_predictive_model()


    def _load_detection_model(self):
        """Load Detection CNN model and artifacts"""
        model_dir = os.path.join(self.artifacts_dir, "unified_cnn")
        model_path = os.path.join(model_dir, "unified_cnn_model.keras")
        config_path = os.path.join(model_dir, "model_config.json")
        scalers_path = os.path.join(model_dir, "scalers.pkl")

        if not os.path.exists(model_path):
            print(f"⚠️  Detection model not found: {model_path}")
            return

        try:
            self.detection_model = keras.models.load_model(model_path)
            print(f"✅ Loaded Detection CNN model")

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.detection_config = json.load(f)
                print(f"✅ Loaded Detection CNN config")

            if os.path.exists(scalers_path):
                import joblib
                self.detection_scalers = joblib.load(scalers_path)
                print(f"✅ Loaded Detection CNN scalers")

        except Exception as e:
            print(f"❌ Error loading Detection model: {e}")


    def _load_predictive_model(self):
        """Load Predictive CNN model and artifacts"""
        model_dir = os.path.join(self.artifacts_dir, "predictive_cnn")
        model_path = os.path.join(model_dir, "predictive_cnn_model.keras")
        config_path = os.path.join(model_dir, "predictive_config.json")
        scalers_path = os.path.join(model_dir, "scalers.pkl")

        if not os.path.exists(model_path):
            print(f"⚠️  Predictive model not found: {model_path}")
            return

        try:
            self.predictive_model = keras.models.load_model(model_path)
            print(f"✅ Loaded Predictive CNN model")

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.predictive_config = json.load(f)
                print(f"✅ Loaded Predictive CNN config")

            if os.path.exists(scalers_path):
                import joblib
                self.predictive_scalers = joblib.load(scalers_path)
                print(f"✅ Loaded Predictive CNN scalers")

        except Exception as e:
            print(f"❌ Error loading Predictive model: {e}")


    def is_available(self) -> Tuple[bool, bool]:
        """
        Check if models are available

        Returns:
            Tuple of (detection_available, predictive_available)
        """
        return (self.detection_model is not None,
                self.predictive_model is not None)


    def preprocess_data(self, data: pd.DataFrame, scalers: dict) -> pd.DataFrame:
        """
        Preprocess data: create pivot table and normalize

        Args:
            data: Raw sensor data
            scalers: Dictionary of fitted scalers

        Returns:
            Normalized pivot DataFrame
        """
        if data.empty:
            return pd.DataFrame()

        # Create sensor key
        data = data.copy()
        data['sensor_key'] = data['sensor_type'] + '_' + data['sensor_id']

        # Pivot
        pivot = data.pivot_table(
            index='timestamp',
            columns='sensor_key',
            values='value',
            aggfunc='first'
        )

        # Fill missing
        pivot = pivot.ffill().bfill()

        # Normalize
        normalized = pivot.copy()
        for col in pivot.columns:
            if col in scalers:
                normalized[col] = scalers[col].transform(pivot[[col]])

        return normalized


    def create_sequences(self, data: pd.DataFrame, window_size: int, stride: int = 1) -> np.ndarray:
        """
        Create sequences for CNN input

        Args:
            data: Normalized sensor data
            window_size: Window size
            stride: Stride for sliding window

        Returns:
            Array of shape (n_sequences, window_size, n_sensors)
        """
        if len(data) < window_size:
            return np.array([])

        data_array = data.values
        sequences = []

        for i in range(0, len(data_array) - window_size + 1, stride):
            window = data_array[i:i+window_size]
            sequences.append(window)

        return np.array(sequences)


    def detect_anomalies(self, data: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """
        Use Detection CNN to identify current anomalies

        Args:
            data: Raw sensor data (must have timestamp, sensor_id, sensor_type, value)
            threshold: Detection threshold (default 0.5)

        Returns:
            Dict with detection results
        """
        if self.detection_model is None:
            return {
                'available': False,
                'message': 'Detection model not loaded'
            }

        if data.empty or len(data) < self.detection_config['window_size']:
            return {
                'available': True,
                'sufficient_data': False,
                'message': f"Need at least {self.detection_config['window_size']} timesteps"
            }

        try:
            # Preprocess
            normalized = self.preprocess_data(data, self.detection_scalers)

            if normalized.empty:
                return {
                    'available': True,
                    'sufficient_data': False,
                    'message': 'Preprocessing failed'
                }

            # Create sequences
            X = self.create_sequences(
                normalized,
                self.detection_config['window_size'],
                stride=10
            )

            if len(X) == 0:
                return {
                    'available': True,
                    'sufficient_data': False,
                    'message': 'No sequences created'
                }

            # Predict
            predictions = self.detection_model.predict(X, verbose=0).flatten()

            # Get most recent prediction
            latest_score = float(predictions[-1])
            latest_prediction = int(latest_score > threshold)

            # Aggregate over all sequences
            avg_score = float(np.mean(predictions))
            max_score = float(np.max(predictions))
            anomaly_pct = float((predictions > threshold).mean() * 100)

            # Determine alert level
            if max_score >= 0.9:
                alert_level = 'critical'
            elif max_score >= 0.7:
                alert_level = 'high'
            elif max_score >= 0.5:
                alert_level = 'medium'
            else:
                alert_level = 'low'

            return {
                'available': True,
                'sufficient_data': True,
                'model_type': 'detection',
                'latest_score': latest_score,
                'latest_prediction': latest_prediction,
                'avg_score': avg_score,
                'max_score': max_score,
                'anomaly_pct': anomaly_pct,
                'alert_level': alert_level,
                'anomaly_detected': latest_prediction == 1,
                'threshold': threshold,
                'n_sequences': len(X),
                'message': 'Detection completed successfully'
            }

        except Exception as e:
            return {
                'available': True,
                'sufficient_data': True,
                'error': True,
                'message': f'Detection error: {str(e)}'
            }


    def predict_future_anomalies(self, data: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """
        Use Predictive CNN to forecast future anomalies

        Args:
            data: Raw sensor data
            threshold: Risk threshold (default 0.5)

        Returns:
            Dict with prediction results including risk score
        """
        if self.predictive_model is None:
            return {
                'available': False,
                'message': 'Predictive model not loaded'
            }

        if data.empty or len(data) < self.predictive_config['window_size']:
            return {
                'available': True,
                'sufficient_data': False,
                'message': f"Need at least {self.predictive_config['window_size']} timesteps"
            }

        try:
            # Preprocess
            normalized = self.preprocess_data(data, self.predictive_scalers)

            if normalized.empty:
                return {
                    'available': True,
                    'sufficient_data': False,
                    'message': 'Preprocessing failed'
                }

            # Create sequences
            X = self.create_sequences(
                normalized,
                self.predictive_config['window_size'],
                stride=10
            )

            if len(X) == 0:
                return {
                    'available': True,
                    'sufficient_data': False,
                    'message': 'No sequences created'
                }

            # Predict
            risk_scores = self.predictive_model.predict(X, verbose=0).flatten()

            # Get most recent risk score
            latest_risk = float(risk_scores[-1])
            latest_prediction = int(latest_risk > threshold)

            # Aggregate
            avg_risk = float(np.mean(risk_scores))
            max_risk = float(np.max(risk_scores))
            high_risk_pct = float((risk_scores > threshold).mean() * 100)

            # Determine alert level based on risk score
            if latest_risk >= 0.8:
                alert_level = '⚫ CRITICAL'
                alert_color = '#dc3545'
            elif latest_risk >= 0.7:
                alert_level = '🔴 ALERT'
                alert_color = '#fd7e14'
            elif latest_risk >= 0.5:
                alert_level = '🟠 WARNING'
                alert_color = '#ffc107'
            elif latest_risk >= 0.3:
                alert_level = '🟡 CAUTION'
                alert_color = '#ffeb3b'
            else:
                alert_level = '🟢 NORMAL'
                alert_color = '#28a745'

            # Calculate time to anomaly (prediction horizon)
            prediction_horizon_min = self.predictive_config.get('prediction_horizon_minutes', 5.0)

            return {
                'available': True,
                'sufficient_data': True,
                'model_type': 'predictive',
                'risk_score': latest_risk,
                'risk_percentage': latest_risk * 100,
                'prediction': latest_prediction,
                'avg_risk': avg_risk,
                'max_risk': max_risk,
                'high_risk_pct': high_risk_pct,
                'alert_level': alert_level,
                'alert_color': alert_color,
                'anomaly_predicted': latest_prediction == 1,
                'threshold': threshold,
                'prediction_horizon_minutes': prediction_horizon_min,
                'n_sequences': len(X),
                'message': f'Risk of anomaly in next {prediction_horizon_min:.1f} min: {latest_risk*100:.1f}%'
            }

        except Exception as e:
            return {
                'available': True,
                'sufficient_data': True,
                'error': True,
                'message': f'Prediction error: {str(e)}'
            }


    def score_window(self, data: pd.DataFrame) -> Dict:
        """
        Score data window with both models

        Args:
            data: Raw sensor data

        Returns:
            Dict with results from both models
        """
        detection_result = self.detect_anomalies(data)
        predictive_result = self.predict_future_anomalies(data)

        return {
            'detection': detection_result,
            'predictive': predictive_result
        }
