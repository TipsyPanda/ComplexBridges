"""
Metrics calculation module
Functions for computing statistics and KPIs
"""

import pandas as pd


def calculate_window_stats(current_window):
    """
    Calculate statistics for the current data window
    
    Args:
        current_window: DataFrame of current window
    
    Returns:
        dict: Dictionary of statistics
    """
    total_records = len(current_window)
    anomaly_count = current_window['anomaly'].sum()
    anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0
    
    avg_traffic = current_window['traffic_load_proxy'].mean() if total_records > 0 else 0
    
    return {
        'total_records': total_records,
        'anomaly_count': int(anomaly_count),
        'anomaly_rate': anomaly_rate,
        'avg_traffic': avg_traffic
    }


def get_sensor_statistics(sensor_data):
    """
    Calculate statistics for a specific sensor
    
    Args:
        sensor_data: DataFrame containing data for one sensor
    
    Returns:
        dict: Sensor statistics
    """
    return {
        'current_value': sensor_data['value'].iloc[-1] if len(sensor_data) > 0 else 0,
        'avg_value': sensor_data['value'].mean(),
        'min_value': sensor_data['value'].min(),
        'max_value': sensor_data['value'].max(),
        'anomaly_count': int(sensor_data['anomaly'].sum()),
        'sensor_type': sensor_data['sensor_type'].iloc[0],
        'unit': sensor_data['unit'].iloc[0],
        'threshold': sensor_data['rule_threshold'].iloc[0],
        'span': sensor_data['span_id'].iloc[0]
    }


def determine_sensor_status(current_value, threshold, warning_multiplier=0.9):
    """
    Determine sensor status based on current value and threshold
    
    Args:
        current_value: Current sensor reading
        threshold: Alert threshold
        warning_multiplier: Multiplier for warning level
    
    Returns:
        tuple: (status_text, status_color)
    """
    if current_value > threshold:
        return "🔴 ALERT", "red"
    elif current_value > threshold * warning_multiplier:
        return "🟡 WARNING", "orange"
    else:
        return "🟢 NORMAL", "green"