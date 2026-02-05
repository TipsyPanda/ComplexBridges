"""Dashboard configuration — centralizes paths, defaults, and styling."""

DATA_PATH = "data/ipmb_5sensors_30min_1_to_10hz.csv"

DEFAULT_THRESHOLDS = {
    "ACC": 0.07,
    "SG": 200.0,
    "TMP": 35.0,
}

PLAYBACK_REFRESH_SEC = 0.1
DEFAULT_PLAYBACK_SPEED = 50
DEFAULT_DISPLAY_WINDOW = 500
DEFAULT_TIME_RANGE_MIN = 30
SKIP_AMOUNT = 100
RECENT_ANOMALIES_LIMIT = 50
HISTOGRAM_BINS = 30

# Palette
CLR_DARK_BLUE = "#0A1853"
CLR_BLUE = "#557FDE"
CLR_LIGHT_BLUE = "#CAE0F5"
CLR_TEXT = "rgba(202, 224, 245, 0.5)"  # #CAE0F5 at 50% opacity
CLR_ORANGE = "#E96547"

CHART_COLORS = {
    "primary": CLR_BLUE,
    "anomaly": CLR_ORANGE,
    "threshold": CLR_LIGHT_BLUE,
    "traffic": CLR_ORANGE,
}

HEALTH_COLORS = {
    "good": CLR_BLUE,
    "fair": CLR_ORANGE,
    "poor": CLR_ORANGE,
    "critical": CLR_ORANGE,
    "unknown": CLR_LIGHT_BLUE,
}

CUSTOM_CSS = f"""
<style>
    /* ---------- Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

    * {{
        font-family: 'Montserrat', sans-serif !important;
    }}

    /* ---------- Global ---------- */
    .stApp {{
        background-color: {CLR_DARK_BLUE};
        color: {CLR_TEXT};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #081040;
    }}
    section[data-testid="stSidebar"] * {{
        color: {CLR_TEXT} !important;
    }}
    section[data-testid="stSidebar"] .stSubheader,
    section[data-testid="stSidebar"] h2 {{
        color: {CLR_LIGHT_BLUE} !important;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: #0D1E6B;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #1A3080;
    }}
    [data-testid="stMetricValue"] {{
        color: {CLR_LIGHT_BLUE} !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {CLR_TEXT} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {CLR_TEXT};
        background-color: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {CLR_LIGHT_BLUE} !important;
        border-bottom-color: {CLR_ORANGE} !important;
    }}

    /* Headers */
    h1 {{
        color: {CLR_LIGHT_BLUE} !important;
        font-weight: 700 !important;
    }}
    h2, h3 {{
        color: {CLR_LIGHT_BLUE} !important;
        font-weight: 600 !important;
    }}

    /* Body text / paragraphs */
    p, span, label, .stMarkdown {{
        color: {CLR_TEXT};
    }}

    /* Dataframes */
    .stDataFrame {{
        border: 1px solid #1A3080;
        border-radius: 8px;
    }}

    /* Progress bar */
    .stProgress > div > div {{
        background-color: {CLR_BLUE} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: #0D1E6B;
        color: {CLR_LIGHT_BLUE};
        border: 1px solid {CLR_BLUE};
    }}
    .stButton > button:hover {{
        background-color: {CLR_BLUE};
        color: white;
        border-color: {CLR_BLUE};
    }}

    /* Multiselect tags */
    span[data-baseweb="tag"] {{
        background-color: {CLR_BLUE} !important;
    }}

    /* Slider */
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {CLR_ORANGE} !important;
    }}

    /* Download button */
    .stDownloadButton > button {{
        background-color: {CLR_BLUE};
        color: white;
        border: none;
    }}
    .stDownloadButton > button:hover {{
        background-color: {CLR_ORANGE};
    }}

    /* Dividers */
    hr {{
        border-color: #1A3080 !important;
    }}

    /* ---- Anti-flicker for Plotly charts during fragment reruns ---- */
    .stPlotlyChart {{
        contain: layout style;
        min-height: 350px;
    }}
</style>
"""
