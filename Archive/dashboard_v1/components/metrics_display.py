"""
Metrics display component
Renders the top metrics bar with KPIs
"""

import streamlit as st
from src.metrics import calculate_window_stats
from src.utils import format_timestamp, get_progress_percentage


def render_metrics(current_window, filtered_data, window_size):
    """
    Render top metrics bar
    
    Args:
        current_window: Current data window
        filtered_data: Filtered dataset
        window_size: Requested window size
    """
    stats = calculate_window_stats(current_window)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        progress = get_progress_percentage(
            st.session_state.current_index, 
            len(filtered_data)
        )
        st.metric(
            "Current Record",
            f"{st.session_state.current_index:,} / {len(filtered_data):,}",
            delta=f"{progress:.1f}%"
        )
    
    with col2:
        st.metric(
            "Window Size",
            f"{stats['total_records']}",
            delta=f"{window_size} requested"
        )
    
    with col3:
        st.metric(
            "Anomalies Detected",
            f"{stats['anomaly_count']}",
            delta=f"{stats['anomaly_rate']:.2f}%",
            delta_color="inverse"
        )
    
    with col4:
        if stats['total_records'] > 0:
            st.metric(
                "Avg Traffic Load",
                f"{stats['avg_traffic']:.2f}",
                delta="High" if stats['avg_traffic'] > 0.7 else "Normal"
            )
        else:
            st.metric("Avg Traffic Load", "N/A")
    
    with col5:
        current_time = current_window['timestamp'].iloc[-1] if len(current_window) > 0 else "N/A"
        st.metric("Current Time", format_timestamp(current_time))