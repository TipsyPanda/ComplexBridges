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

            # Add this function to realtime_plots.py

def add_ml_score_annotation(fig, ml_scorer, sensor_data, span_id, sensor_type):
    """Add ML risk score annotation to plot"""
    
    if ml_scorer is None:
        return fig
    
    try:
        risk_data = ml_scorer.compute_risk_score(sensor_data, span_id, sensor_type)
        
        if risk_data['has_model']:
            score = risk_data['risk_score']
            level = risk_data['risk_level']
            emoji = ml_scorer.get_risk_emoji(level)
            color = ml_scorer.get_risk_color(level)
            
            # Add annotation box
            fig.add_annotation(
                text=f"{emoji} ML Risk: {level.upper()}<br>Score: {score:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                showarrow=False,
                bgcolor=color,
                opacity=0.8,
                font=dict(color="white", size=10),
                borderpad=4,
                bordercolor=color,
                borderwidth=2
            )
    except Exception as e:
        pass  # Silently skip if scoring fails
    
    return fig

    # Modify create_sensor_plot call in render_realtime_plots
    fig = create_sensor_plot(sensor_specific, sensor_id, threshold, unit)

    # Add ML annotation if available
    if 'ml_scorer' in st.session_state and st.session_state.ml_scorer:
        fig = add_ml_score_annotation(
            fig, 
            st.session_state.ml_scorer, 
            sensor_specific, 
            sensor_specific['span_id'].iloc[0],
            sensor_type
        )