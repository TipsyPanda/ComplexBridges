import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Bridge Sensor Monitor",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌉 Bridge Sensor Real-Time Monitor</div>', unsafe_allow_html=True)
st.markdown("---")

# Load data with caching
@st.cache_data
def load_data():
    """Load the bridge sensor data from CSV"""
    try:
        print("Loading data...")
        url = "https://github.com/TipsyPanda/ComplexBridges/raw/main/data/ipmb_5sensors_30min_1_to_10hz.csv"
        
        try:
            data = pd.read_csv(url)
            print("Data loaded successfully.")
            return data
            
        except pd.errors.EmptyDataError:
            st.error("❌ The data file is empty. Please check the data source.")
            return None
            
        except pd.errors.ParserError:
            st.error("❌ Unable to parse the CSV file. The file may be corrupted or in an incorrect format.")
            return None
            
    except Exception as e:
        st.error(f"""
        ❌ Failed to load the bridge sensor data:
        
        **Error**: {str(e)}
        
        Please ensure:
        - You have a stable internet connection
        - The data file exists at the specified URL
        - You have permission to access the file
        
        Try refreshing the page. If the problem persists, contact the system administrator.
        """)
    
   
    return None

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'speed' not in st.session_state:
    st.session_state.speed = 1

# Load data
with st.spinner('Loading bridge sensor data...'):
    data = load_data()

# Check if data loaded successfully
if data is None:
    st.stop()

# Sidebar controls
st.sidebar.header("⚙️ Control Panel")

# Playback controls
st.sidebar.subheader("Playback Controls")
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button("▶️ Play" if not st.session_state.is_playing else "⏸️ Pause"):
        st.session_state.is_playing = not st.session_state.is_playing

with col2:
    if st.button("⏮️ Reset"):
        st.session_state.current_index = 0
        st.session_state.is_playing = False

with col3:
    if st.button("⏭️ Skip"):
        st.session_state.current_index = min(st.session_state.current_index + 100, len(data) - 1)

# Speed control
st.session_state.speed = st.sidebar.slider(
    "Playback Speed",
    min_value=1,
    max_value=1000,
    value=st.session_state.speed,
    help="Control how fast the data streams"
)

# Window size control
window_size = st.sidebar.slider(
    "Display Window (records)",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
    help="Number of recent records to display"
)

# Sensor selection
st.sidebar.subheader("Sensor Selection")
available_sensors = data['sensor_id'].unique()
selected_sensors = st.sidebar.multiselect(
    "Select Sensors to Display",
    options=available_sensors,
    default=available_sensors,
    help="Choose which sensors to monitor"
)

# Span selection
available_spans = data['span_id'].unique()
selected_spans = st.sidebar.multiselect(
    "Select Spans",
    options=available_spans,
    default=available_spans,
    help="Choose which spans to monitor"
)

st.sidebar.markdown("---")

# Threshold overrides
st.sidebar.subheader("Threshold Settings")
use_custom_thresholds = st.sidebar.checkbox("Use Custom Thresholds", value=False)

custom_thresholds = {}
if use_custom_thresholds:
    custom_thresholds['strain_gauge'] = st.sidebar.number_input(
        "Strain Gauge Threshold (microstrain)",
        value=200.0,
        step=10.0
    )
    custom_thresholds['accelerometer_rms'] = st.sidebar.number_input(
        "Accelerometer Threshold (g)",
        value=0.05,
        step=0.01
    )
    custom_thresholds['temperature'] = st.sidebar.number_input(
        "Temperature Threshold (°C)",
        value=35.0,
        step=1.0
    )

# Filter data based on selections
filtered_data = data[
    (data['sensor_id'].isin(selected_sensors)) & 
    (data['span_id'].isin(selected_spans))
]

# Get current window of data
start_idx = max(0, st.session_state.current_index - window_size)
end_idx = st.session_state.current_index + 1
current_window = filtered_data.iloc[start_idx:end_idx]

# Calculate statistics for current window
total_records = len(current_window)
anomaly_count = current_window['anomaly'].sum()
anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0

# Top metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Record",
        f"{st.session_state.current_index:,} / {len(filtered_data):,}",
        delta=f"{(st.session_state.current_index/len(filtered_data)*100):.1f}%"
    )

with col2:
    st.metric(
        "Window Size",
        f"{total_records}",
        delta=f"{window_size} requested"
    )

with col3:
    st.metric(
        "Anomalies Detected",
        f"{int(anomaly_count)}",
        delta=f"{anomaly_rate:.2f}%",
        delta_color="inverse"
    )

with col4:
    if total_records > 0:
        avg_traffic = current_window['traffic_load_proxy'].mean()
        st.metric(
            "Avg Traffic Load",
            f"{avg_traffic:.2f}",
            delta="High" if avg_traffic > 0.7 else "Normal"
        )
    else:
        st.metric("Avg Traffic Load", "N/A")

with col5:
    current_time = current_window['timestamp'].iloc[-1] if len(current_window) > 0 else "N/A"
    st.metric("Current Time", str(current_time).split('.')[0] if current_time != "N/A" else "N/A")

st.markdown("---")

# Check for active anomalies in current view
current_anomalies = current_window[current_window['anomaly'] == 1]
if len(current_anomalies) > 0:
    st.markdown('<div class="anomaly-alert">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time Plots", "📈 Sensor Dashboard", "🔔 Anomaly Analysis", "📋 Data Table"])

with tab1:
    st.subheader("Real-Time Sensor Readings")
    
    # Group by sensor type for plotting
    sensor_types = current_window['sensor_type'].unique()
    
    for sensor_type in sensor_types:
        st.markdown(f"### {sensor_type.replace('_', ' ').title()}")
        
        sensor_data = current_window[current_window['sensor_type'] == sensor_type]
        
        if len(sensor_data) == 0:
            st.info(f"No data for {sensor_type}")
            continue
        
        # Get threshold
        threshold = sensor_data['rule_threshold'].iloc[0]
        if use_custom_thresholds and sensor_type in custom_thresholds:
            threshold = custom_thresholds[sensor_type]
        
        unit = sensor_data['unit'].iloc[0]
        
        # Create figure for each sensor in this type
        for sensor_id in sensor_data['sensor_id'].unique():
            sensor_specific = sensor_data[sensor_data['sensor_id'] == sensor_id]
            
            fig = go.Figure()
            
            # Add normal readings
            normal_data = sensor_specific[sensor_specific['anomaly'] == 0]
            fig.add_trace(go.Scatter(
                x=list(range(len(normal_data))),
                y=normal_data['value'],
                mode='lines',
                name='Normal',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Value</b>: %{y:.2f} ' + unit + '<br><b>Index</b>: %{x}<extra></extra>'
            ))
            
            # Add anomaly readings
            anomaly_data = sensor_specific[sensor_specific['anomaly'] == 1]
            if len(anomaly_data) > 0:
                # Get the x positions for anomalies
                anomaly_indices = []
                for idx, row in anomaly_data.iterrows():
                    pos = normal_data.index.get_loc(idx) if idx in normal_data.index else len(normal_data)
                    anomaly_indices.append(pos)
                
                fig.add_trace(go.Scatter(
                    x=anomaly_indices,
                    y=anomaly_data['value'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=10, symbol='x'),
                    hovertemplate='<b>ANOMALY!</b><br><b>Value</b>: %{y:.2f} ' + unit + '<br><b>Threshold</b>: ' + str(threshold) + '<extra></extra>'
                ))
            
            # Add threshold line
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold}",
                annotation_position="right"
            )
            
            # Add safe zone
            fig.add_hrect(
                y0=0, y1=threshold,
                fillcolor="green", opacity=0.1,
                layer="below", line_width=0,
                annotation_text="Safe Zone", annotation_position="top left"
            )
            
            fig.update_layout(
                title=f"{sensor_id} - {sensor_specific['span_id'].iloc[0]}",
                xaxis_title="Reading Number",
                yaxis_title=f"Value ({unit})",
                hovermode='closest',
                height=300,
                showlegend=True,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Sensor Dashboard")
    
    # Create a grid of sensor metrics
    sensor_ids = current_window['sensor_id'].unique()
    
    cols = st.columns(min(3, len(sensor_ids)))
    
    for idx, sensor_id in enumerate(sensor_ids):
        with cols[idx % 3]:
            sensor_data = current_window[current_window['sensor_id'] == sensor_id]
            
            if len(sensor_data) > 0:
                sensor_type = sensor_data['sensor_type'].iloc[0]
                unit = sensor_data['unit'].iloc[0]
                current_value = sensor_data['value'].iloc[-1]
                threshold = sensor_data['rule_threshold'].iloc[0]
                span = sensor_data['span_id'].iloc[0]
                
                # Calculate statistics
                avg_value = sensor_data['value'].mean()
                min_value = sensor_data['value'].min()
                max_value = sensor_data['value'].max()
                anomalies = sensor_data['anomaly'].sum()
                
                # Determine status
                if current_value > threshold:
                    status = "🔴 ALERT"
                    status_color = "red"
                elif current_value > threshold * 0.9:
                    status = "🟡 WARNING"
                    status_color = "orange"
                else:
                    status = "🟢 NORMAL"
                    status_color = "green"
                
                st.markdown(f"""
                <div style="border: 2px solid {status_color}; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <h4>{sensor_id}</h4>
                    <p><strong>Type:</strong> {sensor_type.replace('_', ' ').title()}</p>
                    <p><strong>Span:</strong> {span}</p>
                    <p><strong>Status:</strong> {status}</p>
                    <p><strong>Current:</strong> {current_value:.2f} {unit}</p>
                    <p><strong>Average:</strong> {avg_value:.2f} {unit}</p>
                    <p><strong>Range:</strong> {min_value:.2f} - {max_value:.2f}</p>
                    <p><strong>Threshold:</strong> {threshold:.2f} {unit}</p>
                    <p><strong>Anomalies:</strong> {int(anomalies)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Mini chart
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=current_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{unit}"},
                    delta={'reference': threshold, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, threshold * 1.5]},
                        'bar': {'color': status_color},
                        'steps': [
                            {'range': [0, threshold], 'color': "lightgray"},
                            {'range': [threshold, threshold * 1.5], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Anomaly Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly distribution by sensor
        anomaly_by_sensor = current_window[current_window['anomaly'] == 1].groupby('sensor_id').size().reset_index(name='count')
        
        if len(anomaly_by_sensor) > 0:
            fig = px.bar(
                anomaly_by_sensor,
                x='sensor_id',
                y='count',
                title='Anomalies by Sensor',
                labels={'sensor_id': 'Sensor ID', 'count': 'Anomaly Count'},
                color='count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies in current window")
    
    with col2:
        # Anomaly distribution by sensor type
        anomaly_by_type = current_window[current_window['anomaly'] == 1].groupby('sensor_type').size().reset_index(name='count')
        
        if len(anomaly_by_type) > 0:
            fig = px.pie(
                anomaly_by_type,
                values='count',
                names='sensor_type',
                title='Anomalies by Sensor Type',
                color_discrete_sequence=px.colors.sequential.Reds
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies in current window")
    
    # Anomaly timeline
    st.markdown("### Anomaly Timeline")
    anomalies_over_time = current_window.groupby(current_window.index // 10)['anomaly'].sum().reset_index()
    anomalies_over_time.columns = ['time_block', 'anomaly_count']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomalies_over_time['time_block'],
        y=anomalies_over_time['anomaly_count'],
        mode='lines+markers',
        fill='tozeroy',
        name='Anomalies',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Anomaly Frequency Over Time',
        xaxis_title='Time Block',
        yaxis_title='Anomaly Count',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent anomalies table
    if len(current_anomalies) > 0:
        st.markdown("### Recent Anomalies")
        anomaly_display = current_anomalies[['timestamp', 'sensor_id', 'sensor_type', 'value', 'unit', 'rule_threshold', 'span_id']].tail(10)
        st.dataframe(anomaly_display, use_container_width=True)

with tab4:
    st.subheader("Current Data Window")
    
    # Display options
    col1, col2 = st.columns([1, 3])
    with col1:
        show_anomalies_only = st.checkbox("Show Anomalies Only", value=False)
    
    # Filter and display data
    display_data = current_window if not show_anomalies_only else current_window[current_window['anomaly'] == 1]
    
    st.dataframe(
        display_data.style.apply(
            lambda x: ['background-color: #ffcccc' if v == 1 else '' for v in x],
            subset=['anomaly']
        ),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Current Window as CSV",
        data=csv,
        file_name=f"bridge_data_window_{st.session_state.current_index}.csv",
        mime="text/csv"
    )

# Auto-advance if playing
if st.session_state.is_playing:
    if st.session_state.current_index < len(filtered_data) - 1:
        st.session_state.current_index += st.session_state.speed
        #time.sleep(0.1)
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.success("✅ Reached end of data!")
        st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🌉 Bridge Sensor Real-Time Monitoring System | Data refreshes automatically during playback
</div>
""", unsafe_allow_html=True)