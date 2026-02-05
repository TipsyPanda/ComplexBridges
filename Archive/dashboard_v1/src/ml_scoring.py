"""
src/ml_scoring.py
Machine Learning Model Scoring Module
Loads trained Isolation Forest models and computes risk scores
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class MLScorer:
    """
    Loads and manages trained ML models for anomaly detection.
    Computes risk scores for real-time sensor data.
    """
    
    def __init__(self, artifact_dir: str):
        """
        Initialize the ML scorer with path to trained models
        
        Args:
            artifact_dir: Path to directory containing trained models
        """
        self.artifact_dir = artifact_dir
        self.models = {}  # Cache for loaded models
        self.thresholds = {}  # Cache for thresholds
        self.scalers = {}  # Cache for scalers if available
        
        # Feature definitions from training notebook
        self.feature_dict = {
            "strain_gauge": ["f_mean", "f_std", "f_rms", "f_p2p", "f_slope", 
                           "fft_low", "fft_mid", "fft_high"],
            "accelerometer_rms": ["f_rms", "fft_centroid", "fft_entropy", "fft_dominant"],
            "temperature": ["f_mean", "f_slope", "ctx_tl_mean"],
        }
        
        print(f"MLScorer initialized with artifact_dir: {artifact_dir}")
    
    
    def load_model(self, span_id: str, sensor_type: str) -> Tuple[Optional[object], Optional[float]]:
        """
        Load trained Isolation Forest model and threshold for a span/sensor combination
        
        Args:
            span_id: Span identifier (e.g., "SPAN_1")
            sensor_type: Sensor type (e.g., "strain_gauge")
        
        Returns:
            Tuple of (model, threshold) or (None, None) if not found
        """
        key = f"{span_id}_{sensor_type}"
        
        # Check cache first
        if key in self.models:
            return self.models[key], self.thresholds.get(key)
        
        # Construct paths
        model_dir = os.path.join(self.artifact_dir, key)
        model_path = os.path.join(model_dir, "isoforest.pkl")
        threshold_path = os.path.join(model_dir, "threshold.json")
        
        # Load model
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None, None
        
        try:
            model = joblib.load(model_path)
            self.models[key] = model
            
            # Load threshold
            threshold = None
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                    threshold = threshold_data.get('score_threshold')
            
            self.thresholds[key] = threshold
            
            print(f"✓ Loaded model: {key}, threshold: {threshold}")
            return model, threshold
            
        except Exception as e:
            print(f"Error loading model {key}: {e}")
            return None, None
    
    
    def compute_features(self, data: pd.DataFrame, sensor_type: str) -> pd.DataFrame:
        """
        Compute features from raw sensor data
        
        Args:
            data: DataFrame with 'value' column
            sensor_type: Type of sensor
        
        Returns:
            DataFrame with computed features
        """
        if data.empty or 'value' not in data.columns:
            return pd.DataFrame()
        
        values = data['value'].values
        
        if sensor_type == "strain_gauge":
            features = self._compute_strain_features(values)
        elif sensor_type == "accelerometer_rms":
            features = self._compute_accel_features(values)
        elif sensor_type == "temperature":
            features = self._compute_temp_features(values)
        else:
            return pd.DataFrame()
        
        return pd.DataFrame([features])
    
    
    def _compute_strain_features(self, values: np.ndarray) -> Dict:
        """Compute strain gauge features"""
        if len(values) == 0:
            return {}
        
        # Time-domain features
        f_mean = np.mean(values)
        f_std = np.std(values)
        f_rms = np.sqrt(np.mean(values**2))
        f_p2p = np.max(values) - np.min(values)
        
        # Slope (linear trend)
        if len(values) > 1:
            x = np.arange(len(values))
            f_slope = np.polyfit(x, values, 1)[0]
        else:
            f_slope = 0.0
        
        # FFT features
        fft_low, fft_mid, fft_high = self._compute_fft_bands(values)
        
        return {
            "f_mean": f_mean,
            "f_std": f_std,
            "f_rms": f_rms,
            "f_p2p": f_p2p,
            "f_slope": f_slope,
            "fft_low": fft_low,
            "fft_mid": fft_mid,
            "fft_high": fft_high
        }
    
    
    def _compute_accel_features(self, values: np.ndarray) -> Dict:
        """Compute accelerometer features"""
        if len(values) == 0:
            return {}
        
        f_rms = np.sqrt(np.mean(values**2))
        
        # FFT-based features
        if len(values) > 1:
            fft_vals = np.fft.rfft(values)
            power = np.abs(fft_vals)**2
            freqs = np.fft.rfftfreq(len(values))
            
            # Spectral centroid
            if power.sum() > 0:
                fft_centroid = np.sum(freqs * power) / power.sum()
            else:
                fft_centroid = 0.0
            
            # Spectral entropy
            power_norm = power / (power.sum() + 1e-10)
            fft_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
            
            # Dominant frequency
            fft_dominant = freqs[np.argmax(power)] if len(power) > 0 else 0.0
        else:
            fft_centroid = 0.0
            fft_entropy = 0.0
            fft_dominant = 0.0
        
        return {
            "f_rms": f_rms,
            "fft_centroid": fft_centroid,
            "fft_entropy": fft_entropy,
            "fft_dominant": fft_dominant
        }
    
    
    def _compute_temp_features(self, values: np.ndarray) -> Dict:
        """Compute temperature features"""
        if len(values) == 0:
            return {}
        
        f_mean = np.mean(values)
        
        # Slope
        if len(values) > 1:
            x = np.arange(len(values))
            f_slope = np.polyfit(x, values, 1)[0]
        else:
            f_slope = 0.0
        
        # Context: traffic load mean (if available, else use window mean)
        ctx_tl_mean = f_mean
        
        return {
            "f_mean": f_mean,
            "f_slope": f_slope,
            "ctx_tl_mean": ctx_tl_mean
        }
    
    
    def _compute_fft_bands(self, values: np.ndarray) -> Tuple[float, float, float]:
        """Compute FFT energy in low, mid, and high frequency bands"""
        if len(values) < 2:
            return 0.0, 0.0, 0.0
        
        fft_vals = np.fft.rfft(values)
        power = np.abs(fft_vals)**2
        
        # Divide into 3 bands
        n = len(power)
        band_size = n // 3
        
        if band_size > 0:
            fft_low = np.mean(power[:band_size])
            fft_mid = np.mean(power[band_size:2*band_size])
            fft_high = np.mean(power[2*band_size:])
        else:
            fft_low = fft_mid = fft_high = np.mean(power)
        
        return float(fft_low), float(fft_mid), float(fft_high)
    
    
    def compute_risk_score(self, 
                          data: pd.DataFrame, 
                          span_id: str, 
                          sensor_type: str) -> Dict:
        """
        Compute ML-based risk score for sensor data
        
        Args:
            data: DataFrame with sensor readings
            span_id: Span identifier
            sensor_type: Sensor type
        
        Returns:
            Dict with risk score and metadata
        """
        # Load model
        model, threshold = self.load_model(span_id, sensor_type)
        
        if model is None:
            return {
                'has_model': False,
                'risk_score': 0.0,
                'risk_level': 'unknown',
                'anomaly_detected': False,
                'message': 'No trained model available'
            }
        
        # Compute features
        features_df = self.compute_features(data, sensor_type)
        
        if features_df.empty:
            return {
                'has_model': True,
                'risk_score': 0.0,
                'risk_level': 'unknown',
                'anomaly_detected': False,
                'message': 'Insufficient data for feature computation'
            }
        
        # Get feature columns in correct order
        feature_cols = self.feature_dict.get(sensor_type, [])
        
        # Ensure all features are present
        missing = [f for f in feature_cols if f not in features_df.columns]
        if missing:
            return {
                'has_model': True,
                'risk_score': 0.0,
                'risk_level': 'error',
                'anomaly_detected': False,
                'message': f'Missing features: {missing}'
            }
        
        # Extract feature array
        X = features_df[feature_cols].values
        
        # Score with Isolation Forest
        try:
            score = float(model.decision_function(X)[0])
            
            # Determine anomaly status
            anomaly_detected = False
            risk_level = 'normal'
            
            if threshold is not None:
                anomaly_detected = (score < threshold)
                
                # Risk levels based on distance from threshold
                if score < threshold:
                    # Below threshold = anomaly
                    distance = abs(score - threshold)
                    if distance > 0.1:
                        risk_level = 'critical'
                    else:
                        risk_level = 'high'
                elif score < threshold + 0.05:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
            
            return {
                'has_model': True,
                'risk_score': score,
                'threshold': threshold,
                'anomaly_detected': anomaly_detected,
                'risk_level': risk_level,
                'features': features_df[feature_cols].to_dict('records')[0],
                'message': 'Score computed successfully'
            }
            
        except Exception as e:
            return {
                'has_model': True,
                'risk_score': 0.0,
                'risk_level': 'error',
                'anomaly_detected': False,
                'message': f'Scoring error: {str(e)}'
            }
    
    
    def batch_score(self, 
                    window_data: pd.DataFrame, 
                    window_size: int = 50) -> pd.DataFrame:
        """
        Score multiple sensors across a time window
        
        Args:
            window_data: DataFrame with columns ['sensor_id', 'span_id', 'sensor_type', 'value', ...]
            window_size: Size of rolling window for feature computation
        
        Returns:
            DataFrame with risk scores per sensor
        """
        results = []
        
        # Group by sensor
        for (sensor_id, span_id, sensor_type), group in window_data.groupby(
            ['sensor_id', 'span_id', 'sensor_type']
        ):
            if len(group) < window_size:
                continue
            
            # Take most recent window
            recent = group.tail(window_size)
            
            # Compute risk score
            risk_data = self.compute_risk_score(recent, span_id, sensor_type)
            
            results.append({
                'sensor_id': sensor_id,
                'span_id': span_id,
                'sensor_type': sensor_type,
                'risk_score': risk_data['risk_score'],
                'risk_level': risk_data['risk_level'],
                'anomaly_detected': risk_data['anomaly_detected'],
                'has_model': risk_data['has_model']
            })
        
        return pd.DataFrame(results)


def get_risk_color(risk_level: str) -> str:
    """
    Get color code for risk level
    
    Args:
        risk_level: Risk level string
    
    Returns:
        Color hex code
    """
    colors = {
        'critical': '#dc3545',  # Red
        'high': '#fd7e14',      # Orange
        'medium': '#ffc107',    # Yellow
        'low': '#28a745',       # Green
        'normal': '#28a745',    # Green
        'unknown': '#6c757d',   # Gray
        'error': '#6c757d'      # Gray
    }
    return colors.get(risk_level, '#6c757d')


def get_risk_emoji(risk_level: str) -> str:
    """
    Get emoji for risk level
    
    Args:
        risk_level: Risk level string
    
    Returns:
        Emoji string
    """
    emojis = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢',
        'normal': '🟢',
        'unknown': '⚪',
        'error': '⚫'
    }
    return emojis.get(risk_level, '⚪')