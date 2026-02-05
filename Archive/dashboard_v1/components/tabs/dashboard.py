"""
Sensor dashboard tab component
Displays individual sensor cards with gauges
"""

import streamlit as st
from src.metrics import get_sensor_statistics, determine_sensor_status
from src.visualization import create_gauge_chart
from config import STATUS_THRESHOLDS


def render_dashboard(current_window, controls):
    """
    Render sensor dashboard with cards and gauges
    
    Args:
        current_window: Current data window
        controls: Control settings from sidebar
    """
    st.subheader("Sensor Dashboard")
    
    sensor_ids = current_window['sensor_id'].unique()
    
    if len(sensor_ids) == 0:
        st.info("No sensors selected")
        return
    
    # Create grid layout
    num_cols = min(3, len(sensor_ids))
    cols = st.columns(num_cols)
    
    for idx, sensor_id in enumerate(sensor_ids):
        with cols[idx % num_cols]:
            sensor_data = current_window[current_window['sensor_id'] == sensor_id]
            
            if len(sensor_data) == 0:
                continue
            
            # Get statistics
            stats = get_sensor_statistics(sensor_data)
            
            # Apply custom threshold if set
            threshold = stats['threshold']
            if controls['use_custom_thresholds'] and stats['sensor_type'] in controls['custom_thresholds']:
                threshold = controls['custom_thresholds'][stats['sensor_type']]
            
            # Determine status
            status, status_color = determine_sensor_status(
                stats['current_value'], 
                threshold,
                STATUS_THRESHOLDS['warning_multiplier']
            )
            
            # Render sensor card
            render_sensor_card(sensor_id, stats, status, status_color, threshold)
            
            # Render gauge chart
            fig = create_gauge_chart(
                stats['current_value'],
                threshold,
                stats['unit'],
                status_color
            )
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                key=f"gauge_{sensor_id}_{st.session_state.current_index}"
            )


def render_sensor_card(sensor_id, stats, status, status_color, threshold):
    """
    Render individual sensor information card
    
    Args:
        sensor_id: Sensor identifier
        stats: Sensor statistics dictionary
        status: Status text
        status_color: Status color
        threshold: Alert threshold
    """
    st.markdown(f"""
    <div style="border: 2px solid {status_color}; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h4>{sensor_id}</h4>
        <p><strong>Type:</strong> {stats['sensor_type'].replace('_', ' ').title()}</p>
        <p><strong>Span:</strong> {stats['span']}</p>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Current:</strong> {stats['current_value']:.2f} {stats['unit']}</p>
        <p><strong>Average:</strong> {stats['avg_value']:.2f} {stats['unit']}</p>
        <p><strong>Range:</strong> {stats['min_value']:.2f} - {stats['max_value']:.2f}</p>
        <p><strong>Threshold:</strong> {threshold:.2f} {stats['unit']}</p>
        <p><strong>Anomalies:</strong> {stats['anomaly_count']}</p>
    </div>
    """, unsafe_allow_html=True)