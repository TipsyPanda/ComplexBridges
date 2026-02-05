"""
Anomaly analysis tab component
Displays charts and tables for anomaly investigation
"""

import streamlit as st
from src.visualization import (
    create_anomaly_bar_chart, 
    create_anomaly_pie_chart, 
    create_anomaly_timeline
)


def render_anomaly_analysis(current_window):
    """
    Render anomaly analysis visualizations
    
    Args:
        current_window: Current data window
    """
    st.subheader("Anomaly Analysis")
    
    # Get anomaly data
    current_anomalies = current_window[current_window['anomaly'] == 1]
    
    # Top row: Bar and Pie charts
    col1, col2 = st.columns(2)
    
    with col1:
        render_anomaly_by_sensor(current_window)
    
    with col2:
        render_anomaly_by_type(current_window)
    
    # Anomaly timeline
    render_anomaly_timeline_section(current_window)
    
    # Recent anomalies table
    render_recent_anomalies_table(current_anomalies)


def render_anomaly_by_sensor(current_window):
    """Render anomalies by sensor chart"""
    anomaly_by_sensor = current_window[current_window['anomaly'] == 1].groupby('sensor_id').size().reset_index(name='count')
    
    if len(anomaly_by_sensor) > 0:
        fig = create_anomaly_bar_chart(anomaly_by_sensor)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("No anomalies in current window")


def render_anomaly_by_type(current_window):
    """Render anomalies by type chart"""
    anomaly_by_type = current_window[current_window['anomaly'] == 1].groupby('sensor_type').size().reset_index(name='count')
    
    if len(anomaly_by_type) > 0:
        fig = create_anomaly_pie_chart(anomaly_by_type)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("No anomalies in current window")


def render_anomaly_timeline_section(current_window):
    """Render anomaly timeline chart"""
    st.markdown("### Anomaly Timeline")
    
    anomalies_over_time = current_window.groupby(current_window.index // 10)['anomaly'].sum().reset_index()
    anomalies_over_time.columns = ['time_block', 'anomaly_count']
    
    fig = create_anomaly_timeline(anomalies_over_time)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_recent_anomalies_table(current_anomalies):
    """Render table of recent anomalies"""
    if len(current_anomalies) > 0:
        st.markdown("### Recent Anomalies")
        anomaly_display = current_anomalies[[
            'timestamp', 'sensor_id', 'sensor_type', 
            'value', 'unit', 'rule_threshold', 'span_id'
        ]].tail(10)
        st.dataframe(anomaly_display, use_container_width=True, height=300)
    else:
        st.info("No anomalies detected in current window")