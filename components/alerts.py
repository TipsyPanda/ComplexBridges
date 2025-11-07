"""
Alert component
Displays anomaly alerts when detected
"""

import streamlit as st


def render_alerts(current_window):
    """
    Render anomaly alerts
    
    Args:
        current_window: Current data window
    """
    current_anomalies = current_window[current_window['anomaly'] == 1]
    
    if len(current_anomalies) > 0:
        st.error(f"⚠️ **ALERT**: {len(current_anomalies)} anomalies detected in current window!")
        
        # Show details of latest anomaly
        latest_anomaly = current_anomalies.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write(f"**Sensor:** {latest_anomaly['sensor_id']}")
        with col2:
            st.write(f"**Type:** {latest_anomaly['sensor_type']}")
        with col3:
            st.write(f"**Value:** {latest_anomaly['value']:.2f} {latest_anomaly['unit']}")
        with col4:
            st.write(f"**Threshold:** {latest_anomaly['rule_threshold']:.2f}")
        
        st.markdown("---")