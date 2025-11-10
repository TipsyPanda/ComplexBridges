"""
ML risk score display component
Shows ML-based anomaly detection results
"""

import streamlit as st
from src.ml_scoring import MLScorer, get_risk_color, get_risk_emoji
from config import ML_CONFIG, RISK_LEVELS


def render_ml_risk_panel(ml_scorer: MLScorer, current_window, window_size: int):
    """
    Render ML-based risk assessment panel
    
    Args:
        ml_scorer: MLScorer instance
        current_window: Current data window
        window_size: Window size for feature computation
    """
    if not ML_CONFIG['enable_ml_scoring']:
        return
    
    st.subheader("🤖 ML Risk Assessment")
    
    if current_window.empty:
        st.info("No data available for ML scoring")
        return
    
    # Batch score all sensors
    try:
        scores_df = ml_scorer.batch_score(current_window, window_size)
        
        if scores_df.empty:
            st.info("Insufficient data for ML scoring (need at least 50 records per sensor)")
            return
        
        # Display risk summary
        render_risk_summary(scores_df)
        
        # Display detailed scores per sensor
        render_sensor_risk_cards(scores_df, ml_scorer, current_window, window_size)
        
    except Exception as e:
        st.error(f"ML Scoring Error: {str(e)}")


def render_risk_summary(scores_df):
    """Display aggregate risk summary"""
    
    # Count by risk level
    risk_counts = scores_df['risk_level'].value_counts()
    
    # Overall risk (highest priority)
    max_priority = -999
    overall_risk = 'unknown'
    
    for level in risk_counts.index:
        priority = RISK_LEVELS.get(level, {}).get('priority', -999)
        if priority > max_priority:
            max_priority = priority
            overall_risk = level
    
    # Display cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        emoji = get_risk_emoji(overall_risk)
        color = get_risk_color(overall_risk)
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {color};">
            <h3>{emoji} Overall Risk</h3>
            <h2>{overall_risk.upper()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical_count = risk_counts.get('critical', 0) + risk_counts.get('high', 0)
        st.metric(
            "🚨 High Priority Alerts",
            int(critical_count),
            delta="Requires attention" if critical_count > 0 else "All clear"
        )
    
    with col3:
        total_sensors = len(scores_df)
        sensors_with_models = scores_df['has_model'].sum()
        st.metric(
            "🤖 ML Coverage",
            f"{sensors_with_models}/{total_sensors}",
            delta=f"{(sensors_with_models/total_sensors*100):.0f}%" if total_sensors > 0 else "0%"
        )
    
    with col4:
        anomaly_count = scores_df['anomaly_detected'].sum()
        st.metric(
            "⚠️ ML Anomalies",
            int(anomaly_count),
            delta=f"{(anomaly_count/len(scores_df)*100):.1f}%" if len(scores_df) > 0 else "0%"
        )


def render_sensor_risk_cards(scores_df, ml_scorer, current_window, window_size):
    """Display individual sensor risk cards"""
    
    st.markdown("### Sensor-Level Risk Scores")
    
    # Sort by risk priority
    def get_priority(row):
        return RISK_LEVELS.get(row['risk_level'], {}).get('priority', -999)
    
    scores_df['_priority'] = scores_df.apply(get_priority, axis=1)
    scores_df = scores_df.sort_values('_priority', ascending=False)
    
    # Display in grid
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx, row in scores_df.iterrows():
        with cols[idx % num_cols]:
            render_single_sensor_card(row, ml_scorer, current_window, window_size)


def render_single_sensor_card(row, ml_scorer, current_window, window_size):
    """Render a single sensor risk card"""
    
    sensor_id = row['sensor_id']
    risk_level = row['risk_level']
    risk_score = row['risk_score']
    
    emoji = get_risk_emoji(risk_level)
    color = get_risk_color(risk_level)
    
    # Get detailed scoring info
    sensor_data = current_window[
        (current_window['sensor_id'] == sensor_id) &
        (current_window['span_id'] == row['span_id'])
    ].tail(window_size)
    
    risk_data = ml_scorer.compute_risk_score(
        sensor_data, 
        row['span_id'], 
        row['sensor_type']
    )
    
    # Build card HTML
    anomaly_badge = "🚨 ANOMALY" if row['anomaly_detected'] else ""
    
    st.markdown(f"""
    <div style="border: 2px solid {color}; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h4>{emoji} {sensor_id} {anomaly_badge}</h4>
        <p><strong>Span:</strong> {row['span_id']}</p>
        <p><strong>Type:</strong> {row['sensor_type'].replace('_', ' ').title()}</p>
        <p><strong>Risk Level:</strong> <span style="color: {color}; font-weight: bold;">{risk_level.upper()}</span></p>
        <p><strong>ML Score:</strong> {risk_score:.4f}</p>
        {f"<p><strong>Threshold:</strong> {risk_data.get('threshold', 'N/A'):.4f}</p>" if risk_data.get('threshold') else ""}
    </div>
    """, unsafe_allow_html=True)
    
    # Show feature importance if available
    if 'features' in risk_data and risk_data['features']:
        with st.expander(f"📊 Feature Values - {sensor_id}"):
            features = risk_data['features']
            for feat, val in features.items():
                st.text(f"{feat}: {val:.4f}")