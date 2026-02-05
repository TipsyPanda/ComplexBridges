"""
Visualization module
Functions for creating Plotly charts and graphs
"""

import plotly.graph_objects as go
import plotly.express as px


def create_sensor_plot(sensor_specific, sensor_id, threshold, unit, use_webgl=True):
    """
    Create an optimized sensor plot with WebGL rendering
    
    Args:
        sensor_specific: DataFrame for specific sensor
        sensor_id: Sensor identifier
        threshold: Alert threshold
        unit: Measurement unit
        use_webgl: Whether to use WebGL for performance
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Prepare data
    normal_mask = sensor_specific['anomaly'] == 0
    anomaly_mask = sensor_specific['anomaly'] == 1
    
    anomaly_data = sensor_specific[anomaly_mask]
    
    # Use WebGL for large datasets
    scatter_type = go.Scattergl if use_webgl and len(sensor_specific) > 100 else go.Scatter
    
    # Add main line
    fig.add_trace(scatter_type(
        x=list(range(len(sensor_specific))),
        y=sensor_specific['value'].values,
        mode='lines',
        name='Reading',
        line=dict(color='#1f77b4', width=1.5),
        hovertemplate=f'<b>Value</b>: %{{y:.2f}} {unit}<br><b>Index</b>: %{{x}}<extra></extra>',
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
            hovertemplate=f'<b>ANOMALY!</b><br><b>Value</b>: %{{y:.2f}} {unit}<extra></extra>',
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
    
    # Add safe zone
    fig.add_hrect(
        y0=0, y1=threshold,
        fillcolor="green", opacity=0.05,
        layer="below", line_width=0
    )
    
    # Layout
    fig.update_layout(
        title=f"{sensor_id} - {sensor_specific['span_id'].iloc[0]}",
        xaxis_title="Reading Number",
        yaxis_title=f"Value ({unit})",
        hovermode='closest',
        height=300,
        showlegend=True,
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        uirevision='constant',
        dragmode='pan'
    )
    
    fig.update_xaxes(fixedrange=False, rangeslider_visible=False)
    fig.update_yaxes(fixedrange=False)
    
    return fig


def create_gauge_chart(current_value, threshold, unit, status_color):
    """
    Create a gauge chart for sensor status
    
    Args:
        current_value: Current sensor value
        threshold: Alert threshold
        unit: Measurement unit
        status_color: Color for the gauge
    
    Returns:
        go.Figure: Plotly figure object
    """
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
    return fig


def create_anomaly_bar_chart(anomaly_by_sensor):
    """Create bar chart of anomalies by sensor"""
    return px.bar(
        anomaly_by_sensor,
        x='sensor_id',
        y='count',
        title='Anomalies by Sensor',
        labels={'sensor_id': 'Sensor ID', 'count': 'Anomaly Count'},
        color='count',
        color_continuous_scale='Reds'
    )


def create_anomaly_pie_chart(anomaly_by_type):
    """Create pie chart of anomalies by type"""
    return px.pie(
        anomaly_by_type,
        values='count',
        names='sensor_type',
        title='Anomalies by Sensor Type',
        color_discrete_sequence=px.colors.sequential.Reds
    )


def create_anomaly_timeline(anomalies_over_time):
    """Create timeline chart of anomaly frequency"""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
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
    return fig