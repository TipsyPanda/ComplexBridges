"""
Bridge Sensor Real-Time Monitoring System
Main application entry point
"""

import streamlit as st
from config import PAGE_CONFIG, CUSTOM_CSS, ML_CONFIG
from src.data_loader import load_data
from src.utils import initialize_session_state
from components.header import render_header
from components.sidebar import render_sidebar
from components.metrics_display import render_metrics
from components.alerts import render_alerts
from components.tabs.realtime_plots import render_realtime_plots
from components.tabs.dashboard import render_dashboard
from components.tabs.anomaly_analysis import render_anomaly_analysis
from components.tabs.data_table import render_data_table
from src.ml_scoring import MLScorer
from components.ml_display import render_ml_risk_panel
import time


# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Load data
data = load_data()
if data is None:
    st.stop()

# Initialize ML scorer 
if ML_CONFIG['enable_ml_scoring']:
    try:
        ml_scorer = MLScorer(ML_CONFIG['artifact_dir'])
    except Exception as e:
        st.warning(f"ML models not available: {e}")
        ml_scorer = None
else:
    ml_scorer = None

# Render header
render_header()

# Render sidebar and get controls
controls = render_sidebar(data)

# Filter data based on sidebar selections
filtered_data = data[
    (data['sensor_id'].isin(controls['selected_sensors'])) & 
    (data['span_id'].isin(controls['selected_spans']))
]

# Get current window
start_idx = max(0, st.session_state.current_index - controls['window_size'])
end_idx = st.session_state.current_index + 1
current_window = filtered_data.iloc[start_idx:end_idx]

# Render metrics
render_metrics(current_window, filtered_data, controls['window_size'])

st.markdown("---")

# Render alerts
render_alerts(current_window)

# Add ML Risk Panel 
if ml_scorer:
    render_ml_risk_panel(ml_scorer, current_window, controls['window_size'])
    st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Real-Time Plots", 
    "📈 Sensor Dashboard", 
    "🔔 Anomaly Analysis", 
    "📋 Data Table"
])

with tab1:
    render_realtime_plots(current_window, controls)

with tab2:
    render_dashboard(current_window, controls)

with tab3:
    render_anomaly_analysis(current_window)

with tab4:
    render_data_table(current_window)

# Auto-advance logic
if st.session_state.is_playing:
    if st.session_state.current_index < len(filtered_data) - 1:
        st.session_state.current_index = min(
            st.session_state.current_index + st.session_state.speed,
            len(filtered_data) - 1
        )
        time.sleep(controls['update_interval'] / 1000.0)
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.success("✅ Reached end of data!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🌉 Bridge Sensor Real-Time Monitoring System | Optimized for smooth playback
</div>
""", unsafe_allow_html=True)