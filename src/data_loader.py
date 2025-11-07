"""
Data loading and caching module
Handles CSV loading from multiple sources
"""

import streamlit as st
import pandas as pd
from config import DATA_URLS, LOCAL_DATA_PATHS


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


def get_filtered_data(data, selected_sensors, selected_spans):
    """
    Filter data based on sensor and span selections
    
    Args:
        data: Full dataset
        selected_sensors: List of sensor IDs to include
        selected_spans: List of span IDs to include
    
    Returns:
        pd.DataFrame: Filtered data
    """
    return data[
        (data['sensor_id'].isin(selected_sensors)) & 
        (data['span_id'].isin(selected_spans))
    ]


def get_data_window(data, current_index, window_size):
    """
    Extract a window of data around the current index
    
    Args:
        data: Full dataset
        current_index: Current position in the data
        window_size: Number of records to include
    
    Returns:
        pd.DataFrame: Data window
    """
    start_idx = max(0, current_index - window_size)
    end_idx = current_index + 1
    return data.iloc[start_idx:end_idx]