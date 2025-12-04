"""
CNN model display component
Shows CNN-based anomaly detection and prediction results
"""

import streamlit as st
from src.cnn_scoring import CNNScorer
import pandas as pd


def render_cnn_panel(cnn_scorer: CNNScorer, current_window: pd.DataFrame):
    """
    Render CNN model panel with both Detection and Predictive models

    Args:
        cnn_scorer: CNNScorer instance
        current_window: Current data window
    """
    st.subheader("🧠 CNN Deep Learning Models")

    if current_window.empty:
        st.info("No data available for CNN scoring")
        return

    # Check model availability
    detection_available, predictive_available = cnn_scorer.is_available()

    if not detection_available and not predictive_available:
        st.warning("No CNN models loaded. Please ensure model artifacts are in the artifacts/ directory.")
        return

    # Score the window with both models
    try:
        results = cnn_scorer.score_window(current_window)

        # Create two columns for the two models
        col1, col2 = st.columns(2)

        with col1:
            render_detection_model(results['detection'])

        with col2:
            render_predictive_model(results['predictive'])

        # Add explanation
        with st.expander("ℹ️ About These Models"):
            st.markdown("""
            ### 🔍 Detection Model
            - **Purpose**: Real-time anomaly detection
            - **Input**: Last 50 timesteps of sensor data
            - **Output**: Binary classification (Normal / Anomaly)
            - **Use**: Identifies anomalies happening NOW

            ### 🔮 Predictive Model
            - **Purpose**: Future anomaly forecasting
            - **Input**: Last 50 timesteps of sensor data
            - **Output**: Risk score (0-100%) of future anomaly
            - **Prediction Horizon**: Next 5 minutes
            - **Use**: Early warning system for upcoming anomalies

            ### Combined Workflow
            1. **Predictive Model** provides advance warning (1-5 min)
            2. **Detection Model** confirms when anomaly occurs
            3. Together they enable proactive intervention
            """)

    except Exception as e:
        st.error(f"CNN Scoring Error: {str(e)}")


def render_detection_model(result: dict):
    """Render Detection Model results"""

    st.markdown("### 🔍 Detection Model")
    st.markdown("*Real-time anomaly identification*")

    if not result.get('available', False):
        st.warning("Detection model not available")
        return

    if not result.get('sufficient_data', False):
        st.info(result.get('message', 'Insufficient data'))
        return

    if result.get('error', False):
        st.error(result.get('message', 'Error occurred'))
        return

    # Get results
    latest_score = result['latest_score']
    anomaly_detected = result['anomaly_detected']
    alert_level = result['alert_level']
    max_score = result['max_score']
    anomaly_pct = result['anomaly_pct']

    # Determine color
    if alert_level == 'critical':
        color = '#dc3545'
        emoji = '🔴'
    elif alert_level == 'high':
        color = '#fd7e14'
        emoji = '🟠'
    elif alert_level == 'medium':
        color = '#ffc107'
        emoji = '🟡'
    else:
        color = '#28a745'
        emoji = '🟢'

    # Status card
    status = "⚠️ ANOMALY DETECTED" if anomaly_detected else "✅ NORMAL"
    st.markdown(f"""
    <div style="background-color: {color}20; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid {color}; margin-bottom: 1rem;">
        <h3 style="margin: 0;">{emoji} {status}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Current State: <strong>{alert_level.upper()}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Latest Score",
            f"{latest_score:.2%}",
            delta="Above threshold" if anomaly_detected else "Normal"
        )

    with col2:
        st.metric(
            "Max Score",
            f"{max_score:.2%}",
            delta=f"{anomaly_pct:.1f}% anomalous"
        )

    # Score bar
    st.markdown("**Detection Score**")
    st.progress(min(latest_score, 1.0))

    # Details
    with st.expander("📊 Detection Details"):
        st.write(f"**Average Score:** {result['avg_score']:.4f}")
        st.write(f"**Max Score:** {result['max_score']:.4f}")
        st.write(f"**Threshold:** {result['threshold']:.2f}")
        st.write(f"**Sequences Analyzed:** {result['n_sequences']}")
        st.write(f"**Anomaly %:** {result['anomaly_pct']:.1f}%")


def render_predictive_model(result: dict):
    """Render Predictive Model results"""

    st.markdown("### 🔮 Predictive Model")
    st.markdown("*Future anomaly forecasting*")

    if not result.get('available', False):
        st.warning("Predictive model not available")
        return

    if not result.get('sufficient_data', False):
        st.info(result.get('message', 'Insufficient data'))
        return

    if result.get('error', False):
        st.error(result.get('message', 'Error occurred'))
        return

    # Get results
    risk_score = result['risk_score']
    risk_pct = result['risk_percentage']
    alert_level = result['alert_level']
    alert_color = result['alert_color']
    anomaly_predicted = result['anomaly_predicted']
    horizon = result['prediction_horizon_minutes']

    # Status card
    status = f"⚠️ ANOMALY PREDICTED in {horizon:.0f} min" if anomaly_predicted else f"✅ LOW RISK (next {horizon:.0f} min)"
    st.markdown(f"""
    <div style="background-color: {alert_color}20; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid {alert_color}; margin-bottom: 1rem;">
        <h3 style="margin: 0;">{alert_level}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{status}</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Risk Score",
            f"{risk_pct:.1f}%",
            delta="High risk" if anomaly_predicted else "Low risk"
        )

    with col2:
        st.metric(
            "Prediction Horizon",
            f"{horizon:.1f} min",
            delta=f"Max risk: {result['max_risk']*100:.1f}%"
        )

    # Risk score bar with colored zones
    st.markdown("**Risk Level**")

    # Create risk gauge
    if risk_score < 0.3:
        gauge_color = '#28a745'  # Green
    elif risk_score < 0.5:
        gauge_color = '#ffeb3b'  # Yellow
    elif risk_score < 0.7:
        gauge_color = '#ffc107'  # Orange
    elif risk_score < 0.8:
        gauge_color = '#fd7e14'  # Orange-Red
    else:
        gauge_color = '#dc3545'  # Red

    st.markdown(f"""
    <div style="background: linear-gradient(to right,
        #28a745 0%, #28a745 30%,
        #ffeb3b 30%, #ffeb3b 50%,
        #ffc107 50%, #ffc107 70%,
        #fd7e14 70%, #fd7e14 80%,
        #dc3545 80%, #dc3545 100%);
        height: 20px; border-radius: 10px; position: relative;">
        <div style="position: absolute; left: {risk_score*100}%; top: -5px;
            width: 4px; height: 30px; background: black;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 0.3rem;">
        <span>0% Low</span>
        <span>50% Medium</span>
        <span>100% Critical</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<p style='text-align: center; font-weight: bold; color: {gauge_color}; font-size: 1.2rem;'>{risk_pct:.1f}% Risk</p>", unsafe_allow_html=True)

    # Recommended action
    st.markdown("**Recommended Action:**")
    if risk_score < 0.3:
        st.success("✅ Normal operations - routine monitoring")
    elif risk_score < 0.5:
        st.info("⚠️ Monitor closely - increase logging frequency")
    elif risk_score < 0.7:
        st.warning("🚨 Alert operators - prepare inspection team")
    elif risk_score < 0.8:
        st.error("🔴 Dispatch team - anomaly likely within 5 minutes")
    else:
        st.error("⚫ CRITICAL - Begin preventive action immediately!")

    # Details
    with st.expander("📊 Prediction Details"):
        st.write(f"**Risk Score:** {risk_score:.4f}")
        st.write(f"**Average Risk:** {result['avg_risk']:.4f}")
        st.write(f"**Max Risk:** {result['max_risk']:.4f}")
        st.write(f"**Threshold:** {result['threshold']:.2f}")
        st.write(f"**Sequences Analyzed:** {result['n_sequences']}")
        st.write(f"**High Risk %:** {result['high_risk_pct']:.1f}%")
        st.write(f"**Prediction Horizon:** {horizon:.1f} minutes")
        st.write(f"**Message:** {result['message']}")


def render_model_comparison(detection_result: dict, predictive_result: dict):
    """
    Render a comparison between Detection and Predictive model results
    """
    st.markdown("### 📊 Model Comparison")

    if not detection_result.get('sufficient_data') or not predictive_result.get('sufficient_data'):
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Detection (Now)**")
        det_score = detection_result.get('latest_score', 0)
        st.progress(min(det_score, 1.0))
        st.caption(f"{det_score:.2%}")

    with col2:
        st.markdown("**Prediction (Future)**")
        pred_score = predictive_result.get('risk_score', 0)
        st.progress(min(pred_score, 1.0))
        st.caption(f"{pred_score:.2%}")

    # Combined alert
    both_high = (detection_result.get('anomaly_detected', False) and
                 predictive_result.get('anomaly_predicted', False))

    if both_high:
        st.error("🚨 **URGENT**: Anomaly detected NOW and more predicted ahead!")
    elif detection_result.get('anomaly_detected', False):
        st.warning("⚠️ Anomaly detected now, but future looks clear")
    elif predictive_result.get('anomaly_predicted', False):
        st.warning("⚠️ No current anomaly, but one predicted soon - prepare now!")
    else:
        st.success("✅ All clear - current and future states normal")
