"""
Real-time plots tab component
"""

import streamlit as st
from src.visualization import create_sensor_plot


def render_realtime_plots(current_window, controls):
    """Render real-time sensor plots"""
    st.subheader("Real-Time Sensor Readings")
    
    sensor_types = current_window['sensor_type'].unique()
    
    for sensor_type in sensor_types:
        st.markdown(f"### {sensor_type.replace('_', ' ').title()}")
        
        sensor_data = current_window[current_window['sensor_type'] == sensor_type]
        
        if len(sensor_data) == 0:
            st.info(f"No data for {sensor_type}")
            continue
        
        # Get threshold
        threshold = sensor_data['rule_threshold'].iloc[0]
        if controls['use_custom_thresholds'] and sensor_type in controls['custom_thresholds']:
            threshold = controls['custom_thresholds'][sensor_type]
        
        unit = sensor_data['unit'].iloc[0]
        
        # Create plot for each sensor
        for sensor_id in sensor_data['sensor_id'].unique():
            sensor_specific = sensor_data[sensor_data['sensor_id'] == sensor_id]
            
            fig = create_sensor_plot(sensor_specific, sensor_id, threshold, unit)
            
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={'displayModeBar': False, 'staticPlot': False},
                key=f"plot_{sensor_id}_{st.session_state.current_index}"
            )