"""
Header component
Renders the application header and title
"""

import streamlit as st


def render_header():
    """Render the main application header"""
    st.markdown(
        '<div class="main-header">🌉 Bridge Sensor Real-Time Monitor</div>', 
        unsafe_allow_html=True
    )
    st.markdown("---")