"""
Data table tab component
Displays raw data in tabular format with download option
"""

import streamlit as st


def render_data_table(current_window):
    """
    Render data table with filtering and download options
    
    Args:
        current_window: Current data window
    """
    st.subheader("Current Data Window")
    
    # Display options
    col1, col2 = st.columns([1, 3])
    with col1:
        show_anomalies_only = st.checkbox("Show Anomalies Only", value=False)
    
    # Filter data
    display_data = current_window if not show_anomalies_only else current_window[current_window['anomaly'] == 1]
    
    if len(display_data) == 0:
        st.info("No data to display with current filters")
        return
    
    # Display dataframe
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )
    
    # Statistics
    render_table_statistics(display_data, show_anomalies_only)
    
    # Download button
    render_download_button(display_data)


def render_table_statistics(display_data, show_anomalies_only):
    """Display statistics about the table"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rows Displayed", len(display_data))
    
    with col2:
        if not show_anomalies_only:
            anomaly_count = display_data['anomaly'].sum()
            st.metric("Anomalies", int(anomaly_count))
        else:
            st.metric("All Anomalies", len(display_data))
    
    with col3:
        unique_sensors = display_data['sensor_id'].nunique()
        st.metric("Sensors", unique_sensors)


def render_download_button(display_data):
    """Render CSV download button"""
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Current Window as CSV",
        data=csv,
        file_name=f"bridge_data_window_{st.session_state.current_index}.csv",
        mime="text/csv"
    )