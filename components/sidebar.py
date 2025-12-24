"""
Sidebar component
Renders all sidebar controls and returns configuration
"""

import streamlit as st
from config import PLAYBACK_CONFIG, DEFAULT_THRESHOLDS, ML_CONFIG


def render_sidebar(data):
    """
    Render sidebar controls
    
    Args:
        data: Full dataset
    
    Returns:
        dict: Dictionary of control values
    """
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Playback controls
        st.subheader("Playback Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Play" if not st.session_state.is_playing else "⏸️ Pause", key="play_pause"):
                st.session_state.is_playing = not st.session_state.is_playing
        
        with col2:
            if st.button("⏮️ Reset", key="reset"):
                st.session_state.current_index = 0
                st.session_state.is_playing = False

        with col3:
            if st.button("⏭️ Skip", key="skip"):
                st.session_state.current_index = min(st.session_state.current_index + 500, len(data) - 1)
        
        # Speed control
        speed = st.slider(
            "Playback Speed (records/update)",
            min_value=PLAYBACK_CONFIG['speed_min'],
            max_value=PLAYBACK_CONFIG['speed_max'],
            value=st.session_state.speed,
            step=PLAYBACK_CONFIG['speed_step'],
            help="How many records to advance per update"
        )
        if speed != st.session_state.speed:
            st.session_state.speed = speed
        
        # Window size
        window_size = st.slider(
            "Display Window (records)",
            min_value=PLAYBACK_CONFIG['window_min'],
            max_value=PLAYBACK_CONFIG['window_max'],
            value=PLAYBACK_CONFIG['window_default'],
            step=PLAYBACK_CONFIG['window_step'],
            help="Number of recent records to display"
        )
        
        # Update interval
        update_interval = st.slider(
            "Update Interval (ms)",
            min_value=PLAYBACK_CONFIG['update_interval_min'],
            max_value=PLAYBACK_CONFIG['update_interval_max'],
            value=PLAYBACK_CONFIG['update_interval_default'],
            step=PLAYBACK_CONFIG['update_interval_step'],
            help="Time between updates (lower = smoother but more CPU)"
        )
        
        st.markdown("---")
        
        # Sensor selection
        st.subheader("Sensor Selection")
        available_sensors = data['sensor_id'].unique()
        selected_sensors = st.multiselect(
            "Select Sensors to Display",
            options=available_sensors,
            default=available_sensors,
            help="Choose which sensors to monitor"
        )
        
        # Span selection
        available_spans = data['span_id'].unique()
        selected_spans = st.multiselect(
            "Select Spans",
            options=available_spans,
            default=available_spans,
            help="Choose which spans to monitor"
        )
        
        st.markdown("---")
        
        # Threshold overrides
        st.subheader("Threshold Settings")
        use_custom_thresholds = st.checkbox("Use Custom Thresholds", value=False)
        
        custom_thresholds = {}
        if use_custom_thresholds:
            for sensor_type, default_value in DEFAULT_THRESHOLDS.items():
                custom_thresholds[sensor_type] = st.number_input(
                    f"{sensor_type.replace('_', ' ').title()} Threshold",
                    value=default_value,
                    step=0.01 if sensor_type == 'accelerometer_rms' else 1.0
                )
                

        if ML_CONFIG['enable_ml_scoring']:
            st.markdown("---")
            st.subheader("🤖 ML Model Settings")
            
            ml_window = st.slider(
                "ML Feature Window",
                min_value=20,
                max_value=200,
                value=ML_CONFIG['feature_window_size'],
                step=10,
                help="Number of recent records for ML feature computation"
            )
            
            show_ml_features = st.checkbox(
                "Show Feature Details",
                value=False,
                help="Display computed features in risk cards"
            )
                    
    
    return {
        'speed': speed,
        'window_size': window_size,
        'update_interval': update_interval,
        'selected_sensors': selected_sensors,
        'selected_spans': selected_spans,
        'use_custom_thresholds': use_custom_thresholds,
        'custom_thresholds': custom_thresholds,
        'ml_window': ml_window,
        'show_ml_features': show_ml_features
    }