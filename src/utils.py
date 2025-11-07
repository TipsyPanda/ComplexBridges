"""
Utility functions
Helper functions for session state and common operations
"""

import streamlit as st


def initialize_session_state():
    """Initialize Streamlit session state variables"""
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


def format_timestamp(timestamp):
    """Format timestamp for display"""
    if timestamp == "N/A":
        return "N/A"
    return str(timestamp).split('.')[0]


def get_progress_percentage(current_index, total_length):
    """Calculate progress percentage"""
    if total_length == 0:
        return 0.0
    return (current_index / total_length) * 100