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

# Custom CSS for better styling and smooth transitions
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
    /* Smooth transitions */
    .stPlotlyChart {
        transition: opacity 0.1s ease-in-out;
    }
    /* Reduce layout shift */
    .element-container {
        transition: none !important;
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

# Initialize session state with more efficient tracking
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'speed' not in st.session_state:
    st.session_state.speed = 100
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'cached_plots' not in st.session_state:
    st.session_state.cached_plots = {}

# Load data
data = load_data()

# Check if data loaded successfully
if data is None:
    st.stop()

# Sidebar controls (only render once)
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Playback controls
    st.subheader("Playback Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        play_button = st.button("▶️ Play" if not st.session_state.is_playing else "⏸️ Pause", key="play_pause")
        if play_button:
            st.session_state.is_playing = not st.session_state.is_playing
    
    with col2:
        if st.button("⏮️ Reset", key="reset"):
            st.session_state.current_index = 0
            st.session_state.is_playing = False
            st.rerun()
    
    with col3:
        if st.button("⏭️ Skip", key="skip"):
            st.session_state.current_index = min(st.session_state.current_index + 500, len(data) - 1)
            st.rerun()
    
    # Speed control
    speed = st.slider(
        "Playback Speed (records/update)",
        min_value=10,
        max_value=1000,
        value=st.session_state.speed,
        step=10,
        help="How many records to advance per update"
    )
    if speed != st.session_state.speed:
        st.session_state.speed = speed
    
    # Window size control
    window_size = st.slider(
        "Display Window (records)",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Number of recent records to display"
    )
    
    # Update interval
    update_interval = st.slider(
        "Update Interval (ms)",
        min_value=50,
        max_value=1000,
        value=100,
        step=50,
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
        custom_thresholds['strain_gauge'] = st.number_input(
            "Strain Gauge Threshold (microstrain)",
            value=200.0,
            step=10.0
        )
        custom_thresholds['accelerometer_rms'] = st.number_input(
            "Accelerometer Threshold (g)",
            value=0.05,
            step=0.01
        )
        custom_thresholds['temperature'] = st.number_input(
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

# Create placeholders for dynamic content
metrics_placeholder = st.container()
alert_placeholder = st.container()

# Top metrics (use container to reduce flicker)
with metrics_placeholder:
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
with alert_placeholder:
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

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time Plots", "📈 Sensor Dashboard", "🔔 Anomaly Analysis", "📋 Data Table"])

# Function to create optimized plot
def create_sensor_plot(sensor_specific, sensor_id, threshold, unit, use_webgl=True):
    """Create an optimized sensor plot with WebGL rendering for smoothness"""
    
    fig = go.Figure()
    
    # Prepare data once
    normal_mask = sensor_specific['anomaly'] == 0
    anomaly_mask = sensor_specific['anomaly'] == 1
    
    normal_data = sensor_specific[normal_mask]
    anomaly_data = sensor_specific[anomaly_mask]
    
    # Use scattergl for better performance with large datasets
    scatter_type = go.Scattergl if use_webgl and len(sensor_specific) > 100 else go.Scatter
    
    # Add normal readings with WebGL
    if len(normal_data) > 0:
        fig.add_trace(scatter_type(
            x=list(range(len(sensor_specific))),
            y=sensor_specific['value'].values,
            mode='lines',
            name='Reading',
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='<b>Value</b>: %{y:.2f} ' + unit + '<br><b>Index</b>: %{x}<extra></extra>',
            showlegend=True
        ))
    
    # Add anomaly markers
    if len(anomaly_data) > 0:
        anomaly_positions = sensor_specific.index[anomaly_mask].tolist()
        anomaly_x = [sensor_specific.index.get_loc(idx) for idx in anomaly_positions]
        
        fig.add_trace(go.Scatter(
            x=anomaly_x,
            y=anomaly_data['value'].values,
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=8, symbol='x', line=dict(width=1)),
            hovertemplate='<b>ANOMALY!</b><br><b>Value</b>: %{y:.2f} ' + unit + '<extra></extra>',
            showlegend=True
        ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        line_width=1,
        annotation_text=f"Threshold: {threshold}",
        annotation_position="right"
    )
    
    # Add safe zone (simplified)
    fig.add_hrect(
        y0=0, y1=threshold,
        fillcolor="green", opacity=0.05,
        layer="below", line_width=0
    )
    
    # Optimize layout for performance
    fig.update_layout(
        title=f"{sensor_id} - {sensor_specific['span_id'].iloc[0]}",
        xaxis_title="Reading Number",
        yaxis_title=f"Value ({unit})",
        hovermode='closest',
        height=300,
        showlegend=True,
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        # Performance optimizations
        uirevision='constant',  # Prevents zoom reset
        dragmode='pan'
    )
    
    # Optimize axes
    fig.update_xaxes(fixedrange=False, rangeslider_visible=False)
    fig.update_yaxes(fixedrange=False)
    
    return fig

with tab1:
    st.subheader("Real-Time Sensor Readings")
    
    # Group by sensor type for plotting
    sensor_types = current_window['sensor_type'].unique()
    
    # Create plots more efficiently
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
            
            # Create plot with optimizations
            fig = create_sensor_plot(sensor_specific, sensor_id, threshold, unit)
            
            # Use plotly with config to reduce flicker
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'displayModeBar': False,  # Hide toolbar to reduce redraws
                    'staticPlot': False
                },
                key=f"plot_{sensor_id}_{st.session_state.current_index}"  # Unique key
            )

with tab2:
    st.subheader("Sensor Dashboard")
    
    # Create a grid of sensor metrics
    sensor_ids = current_window['sensor_id'].unique()
    
    # Use columns more efficiently
    num_cols = min(3, len(sensor_ids))
    cols = st.columns(num_cols)
    
    for idx, sensor_id in enumerate(sensor_ids):
        with cols[idx % num_cols]:
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
                
                # Mini gauge chart with optimization
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=current_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{unit}"},
                    gauge={
                        'axis': {'range': [None, threshold * 1.5]},
                        'bar': {'color': status_color},
                        'steps': [
                            {'range': [0, threshold], 'color': "lightgray"},
                            {'range': [threshold, threshold * 1.5], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': threshold
                        }
                    }
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_{sensor_id}_{st.session_state.current_index}")

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
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
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
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No anomalies in current window")
    
    # Anomaly timeline with WebGL
    st.markdown("### Anomaly Timeline")
    anomalies_over_time = current_window.groupby(current_window.index // 10)['anomaly'].sum().reset_index()
    anomalies_over_time.columns = ['time_block', 'anomaly_count']
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(  # Use WebGL for performance
        x=anomalies_over_time['time_block'],
        y=anomalies_over_time['anomaly_count'],
        mode='lines+markers',
        fill='tozeroy',
        name='Anomalies',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title='Anomaly Frequency Over Time',
        xaxis_title='Time Block',
        yaxis_title='Anomaly Count',
        template='plotly_white',
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Recent anomalies table
    if len(current_anomalies) > 0:
        st.markdown("### Recent Anomalies")
        anomaly_display = current_anomalies[['timestamp', 'sensor_id', 'sensor_type', 'value', 'unit', 'rule_threshold', 'span_id']].tail(10)
        st.dataframe(anomaly_display, use_container_width=True, height=300)

with tab4:
    st.subheader("Current Data Window")
    
    # Display options
    col1, col2 = st.columns([1, 3])
    with col1:
        show_anomalies_only = st.checkbox("Show Anomalies Only", value=False)
    
    # Filter and display data
    display_data = current_window if not show_anomalies_only else current_window[current_window['anomaly'] == 1]
    
    # Show dataframe without styling for better performance
    st.dataframe(
        display_data,
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

# Auto-advance if playing (optimized)
if st.session_state.is_playing:
    if st.session_state.current_index < len(filtered_data) - 1:
        st.session_state.current_index = min(
            st.session_state.current_index + st.session_state.speed,
            len(filtered_data) - 1
        )
        # Use sleep to control update rate
        time.sleep(update_interval / 1000.0)
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