"""
Configuration file for the Bridge Monitoring System
Contains constants, settings, and configuration parameters
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "Bridge Sensor Monitor",
    "page_icon": "🌉",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Data source URLs
DATA_URLS = {
    "primary": "https://github.com/TipsyPanda/ComplexBridges/raw/main/data/ipmb_5sensors_30min_1_to_10hz.csv",
    "backup": "https://raw.githubusercontent.com/TipsyPanda/ComplexBridges/main/ipmb_5sensors_30min_1_to_10hz.csv"
}

# Local data paths to try
LOCAL_DATA_PATHS = [
    "ipmb_5sensors_30min_1_to_10hz.csv",
    "../ipmb_5sensors_30min_1_to_10hz.csv",
    "data/ipmb_5sensors_30min_1_to_10hz.csv",
]

# Default thresholds
DEFAULT_THRESHOLDS = {
    'strain_gauge': 200.0,
    'accelerometer_rms': 0.05,
    'temperature': 35.0
}

# Playback settings
PLAYBACK_CONFIG = {
    'speed_min': 10,
    'speed_max': 1000,
    'speed_default': 100,
    'speed_step': 10,
    'window_min': 100,
    'window_max': 2000,
    'window_default': 500,
    'window_step': 100,
    'update_interval_min': 50,
    'update_interval_max': 1000,
    'update_interval_default': 100,
    'update_interval_step': 50
}

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'success': 'green',
    'warning': 'orange',
    'danger': 'red',
    'normal': '#1f77b4',
    'anomaly': '#ef4444'
}

# Status thresholds
STATUS_THRESHOLDS = {
    'warning_multiplier': 0.9,  # 90% of threshold = warning
    'alert_multiplier': 1.0     # 100% of threshold = alert
}

# ML Model paths
ML_CONFIG = {
    'artifact_dir': 'artifacts/',  
    'enable_ml_scoring': True,         # Toggle ML-based anomaly detection
    'feature_window_size': 50,         # Window size for feature computation
    'score_update_frequency': 10,      # Update scores every N records
}

# Risk level thresholds and colors
RISK_LEVELS = {
    'critical': {'color': '#dc3545', 'emoji': '🔴', 'priority': 4},
    'high':     {'color': '#fd7e14', 'emoji': '🟠', 'priority': 3},
    'medium':   {'color': '#ffc107', 'emoji': '🟡', 'priority': 2},
    'low':      {'color': '#28a745', 'emoji': '🟢', 'priority': 1},
    'normal':   {'color': '#28a745', 'emoji': '🟢', 'priority': 0},
    'unknown':  {'color': '#6c757d', 'emoji': '⚪', 'priority': -1},
    'error':    {'color': '#6c757d', 'emoji': '⚫', 'priority': -2}
}

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .stPlotlyChart {
        transition: opacity 0.1s ease-in-out;
    }
    .element-container {
        transition: none !important;
    }
</style>
"""