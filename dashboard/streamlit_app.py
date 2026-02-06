"""
Combined Bridge Monitoring Dashboard
Base: Dashboard 2 (Arsla12) with features from Dashboard 1 (ComplexBridges)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta

from config import (
    CHART_COLORS,
    CLR_BLUE,
    CLR_DARK_BLUE,
    CLR_LIGHT_BLUE,
    CLR_ORANGE,
    CLR_TEXT,
    CUSTOM_CSS,
    DATA_PATH,
    DEFAULT_DISPLAY_WINDOW,
    DEFAULT_PLAYBACK_SPEED,
    DEFAULT_THRESHOLDS,
    DEFAULT_TIME_RANGE_MIN,
    HEALTH_COLORS,
    HISTOGRAM_BINS,
    PLAYBACK_REFRESH_SEC,
    RECENT_ANOMALIES_LIMIT,
    SKIP_AMOUNT,
)
from utils.visualization import get_threshold, get_unit, sensor_timeseries

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Bridge Monitoring Dashboard",
    layout="wide",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- Initialize session state ----------------
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# ---------------- Load data ----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Data file not found: {DATA_PATH}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

df = load_data()
if df is None:
    st.stop()

# ---------------- Title ----------------
st.title("Bridge Monitoring Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")

# Bridge selection
selected_bridge = st.sidebar.selectbox(
    "Select Bridge",
    sorted(df["bridge_id"].unique()),
)

# Time range slider
start_time_all = df["timestamp"].min()
end_time_all = df["timestamp"].max()
max_minutes = int((end_time_all - start_time_all).total_seconds() // 60) + 1

time_range = st.sidebar.slider(
    "Time Range (minutes from start)",
    min_value=1,
    max_value=max_minutes,
    value=(1, min(DEFAULT_TIME_RANGE_MIN, max_minutes)),
    step=1,
)

start_time = start_time_all + pd.Timedelta(minutes=time_range[0])
end_time = start_time_all + pd.Timedelta(minutes=time_range[1])

# Span selection
st.sidebar.markdown("---")
st.sidebar.subheader("Span Selection")

available_spans = sorted(df["span_id"].unique()) if "span_id" in df.columns else ["SPAN_1"]
selected_spans = st.sidebar.multiselect(
    "Select Spans",
    options=available_spans,
    default=available_spans,
    help="Filter data by bridge span location",
)

# ---------------- Playback Controls ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Playback Controls")

col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button("▶ Play" if not st.session_state.is_playing else "⏸ Pause"):
        st.session_state.is_playing = not st.session_state.is_playing

with col2:
    if st.button("↺ Reset"):
        st.session_state.current_index = 0
        st.session_state.is_playing = False

with col3:
    if st.button("⏭ Skip"):
        st.session_state.current_index = min(
            st.session_state.current_index + SKIP_AMOUNT,
            len(df) - 1,
        )

playback_speed = st.sidebar.slider(
    "Playback Speed (records/update)",
    min_value=1,
    max_value=500,
    value=DEFAULT_PLAYBACK_SPEED,
    help="Number of records to advance per update",
)

display_window = st.sidebar.slider(
    "Display Window (records)",
    min_value=100,
    max_value=2000,
    value=DEFAULT_DISPLAY_WINDOW,
    help="Number of records to show in plots",
)

# ---------------- Threshold Settings ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Threshold Settings")

use_custom_thresholds = st.sidebar.checkbox(
    "Use Custom Thresholds",
    value=False,
    help="Override default thresholds from data",
)

custom_thresholds = {}
if use_custom_thresholds:
    st.sidebar.markdown("**Set thresholds by sensor type:**")

    custom_thresholds["ACC"] = st.sidebar.number_input(
        "Accelerometer (g)",
        min_value=0.01,
        max_value=1.0,
        value=DEFAULT_THRESHOLDS["ACC"],
        step=0.01,
        format="%.2f",
    )
    custom_thresholds["SG"] = st.sidebar.number_input(
        "Strain Gauge (microstrain)",
        min_value=50.0,
        max_value=500.0,
        value=DEFAULT_THRESHOLDS["SG"],
        step=10.0,
        format="%.0f",
    )
    custom_thresholds["TMP"] = st.sidebar.number_input(
        "Temperature (°C)",
        min_value=20.0,
        max_value=60.0,
        value=DEFAULT_THRESHOLDS["TMP"],
        step=1.0,
        format="%.0f",
    )

# ---------------- Filter data ----------------
filtered_df = df[
    (df["bridge_id"] == selected_bridge)
    & (df["timestamp"] >= start_time)
    & (df["timestamp"] <= end_time)
]

# Apply span filter
if "span_id" in filtered_df.columns and selected_spans:
    filtered_df = filtered_df[filtered_df["span_id"].isin(selected_spans)]

# Apply custom thresholds — vectorized (replaces slow iterrows)
if use_custom_thresholds and "sensor_id" in filtered_df.columns:
    filtered_df = filtered_df.copy()
    for prefix, key in [("ACC", "ACC"), ("SG", "SG"), ("TMP", "TMP")]:
        mask = filtered_df["sensor_id"].str.startswith(prefix)
        filtered_df.loc[mask, "rule_threshold"] = custom_thresholds.get(
            key, DEFAULT_THRESHOLDS[key]
        )

# Reset playback when filters change to avoid stale index
filter_key = f"{selected_bridge}|{'|'.join(sorted(selected_spans))}|{time_range}"
if st.session_state.get("_last_filter_key") != filter_key:
    st.session_state.current_index = 0
    st.session_state.is_playing = False
    st.session_state._last_filter_key = filter_key

# ---------------- Helper ----------------
def compute_health_score(data: pd.DataFrame) -> tuple:
    """Aggregate bridge health score (0–100) from CSV columns."""
    if data.empty:
        return 100.0, "Unknown", HEALTH_COLORS["unknown"]

    anomaly_rate = (data["anomaly"] == 1).mean() if "anomaly" in data.columns else 0.0
    anomaly_score = 50 * (1 - min(anomaly_rate * 5, 1.0))

    if "rule_threshold" in data.columns and data["rule_threshold"].notna().any():
        valid = data["rule_threshold"] > 0
        if valid.any():
            ratio = (data.loc[valid, "value"] / data.loc[valid, "rule_threshold"]).mean()
            headroom_score = 30 * max(0.0, 1 - ratio)
        else:
            headroom_score = 30.0
    else:
        headroom_score = 30.0

    grouped = data.groupby("sensor_id")["value"]
    cv = (grouped.std() / grouped.mean().replace(0, np.nan)).mean()
    stability_score = max(0.0, 20 * (1 - min((cv if np.isfinite(cv) else 0), 1.0)))

    health = anomaly_score + headroom_score + stability_score

    if health >= 80:
        return health, "Good", HEALTH_COLORS["good"]
    elif health >= 60:
        return health, "Fair", HEALTH_COLORS["fair"]
    elif health >= 40:
        return health, "Poor", HEALTH_COLORS["poor"]
    else:
        return health, "Critical", HEALTH_COLORS["critical"]


# ================================================================
# Playback helper — compute the current data window from session
# state so each fragment can independently read the same slice.
# ================================================================
_refresh = timedelta(milliseconds=int(PLAYBACK_REFRESH_SEC * 1000))


def _playback_window():
    """Return (playback_df, in_playback) based on current session state."""
    in_playback = st.session_state.is_playing or st.session_state.current_index > 0
    if in_playback:
        window_start = max(0, st.session_state.current_index - display_window)
        window_end = st.session_state.current_index + 1
        return filtered_df.iloc[window_start:window_end], True
    return filtered_df, False


# ================================================================
# Fragment 1 — Header metrics + progress bar.
# Only these few elements re-render every tick during playback.
# ================================================================
@st.fragment(run_every=_refresh if st.session_state.is_playing else None)
def _live_header():
    # Advance playback index
    if st.session_state.is_playing:
        st.session_state.current_index = min(
            st.session_state.current_index + playback_speed,
            len(filtered_df) - 1,
        )
        if st.session_state.current_index >= len(filtered_df) - 1:
            st.session_state.is_playing = False
            st.toast("Reached end of data!")

    playback_df, in_playback = _playback_window()
    health_source = playback_df if in_playback else filtered_df
    health, status, color = compute_health_score(health_source)
    anomaly_pct = (
        (health_source["anomaly"] == 1).mean() * 100
        if "anomaly" in health_source.columns
        else 0.0
    )

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Bridge Health", f"{health:.0f} / 100")
    h2.markdown(
        f"### <span style='color:{color}'>{status}</span>",
        unsafe_allow_html=True,
    )
    h3.metric("Total Records", f"{len(health_source):,}")
    h4.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")

    st.markdown("---")

    if in_playback:
        progress = st.session_state.current_index / max(len(filtered_df) - 1, 1)
        st.progress(
            progress,
            text=f"Record {st.session_state.current_index:,} / {len(filtered_df):,}",
        )


_live_header()

# ---- Tabs (outside any auto-rerunning fragment) ----
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Monitoring",
    "Alerts",
    "Historical Analysis",
    "Correlation",
])

# ==================== TAB 1: LIVE MONITORING ====================
# Wrapped in its own fragment so only the chart grid re-renders
# during playback — the tab bar and other tabs stay untouched.
# ================================================================
with tab1:
    @st.fragment(run_every=_refresh if st.session_state.is_playing else None)
    def _live_charts():
        st.subheader("Live Sensor Monitoring")

        sensor_list = sorted(filtered_df["sensor_id"].unique())
        selected_sensors = st.multiselect(
            "Select sensors",
            options=sensor_list,
            default=sensor_list,
            key="live_sensor_select",
        )

        playback_df, in_playback = _playback_window()
        plot_source = playback_df if in_playback else filtered_df
        plot_df = plot_source[plot_source["sensor_id"].isin(selected_sensors)]

        for i in range(0, len(selected_sensors), 2):
            cols = st.columns(2)
            for col_idx, sid in enumerate(selected_sensors[i : i + 2]):
                with cols[col_idx]:
                    s_df = plot_df[plot_df["sensor_id"] == sid].sort_values("timestamp")
                    fig = sensor_timeseries(
                        s_df,
                        sid,
                        unit=get_unit(s_df),
                        threshold=get_threshold(s_df),
                    )
                    st.plotly_chart(fig, key=f"live_{sid}", width='stretch')

    _live_charts()

# ==================== TAB 2: ALERTS ====================
# Static — only re-renders on full page rerun (sidebar changes).
with tab2:
    st.subheader("Alerts")
    alerts_df = filtered_df.copy()
    total_points = len(alerts_df)

    if "anomaly" in alerts_df.columns and total_points > 0:
        anomaly_points = int((alerts_df["anomaly"] == 1).sum())
        normal_points = total_points - anomaly_points

        met1, met2, met3 = st.columns(3)
        met1.metric("Total Data Points", f"{total_points:,}")
        met2.metric("Normal Points", f"{normal_points:,}")
        met3.metric("Anomaly Points", f"{anomaly_points:,}")

        st.markdown("---")

        critical_count = 0
        warning_count = 0

        if "rule_threshold" in alerts_df.columns and alerts_df["rule_threshold"].notna().any():
            valid = alerts_df["rule_threshold"] > 0
            sev_df = alerts_df[valid].copy()
            if not sev_df.empty:
                exceed = (
                    (sev_df["value"] - sev_df["rule_threshold"]) / sev_df["rule_threshold"]
                )
                above = sev_df["value"] > sev_df["rule_threshold"]
                critical_count = int((above & (exceed > 0.5)).sum())
                warning_count = int((above & (exceed <= 0.5)).sum())

        st.subheader("Alert Severity")
        sev1, sev2 = st.columns(2)
        sev1.metric("Critical Alerts", critical_count)
        sev2.metric("Warning Alerts", warning_count)

        st.markdown("---")

        if anomaly_points > 0:
            st.subheader("Recent Anomalies")

            fcol1, fcol2 = st.columns(2)
            with fcol1:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=["critical", "warning"],
                    default=["critical", "warning"],
                    key="alert_severity_filter",
                )
            with fcol2:
                alert_sensor_filter = st.multiselect(
                    "Filter by Sensor",
                    options=sorted(filtered_df["sensor_id"].unique()),
                    default=sorted(filtered_df["sensor_id"].unique()),
                    key="alert_sensor_filter",
                )

            recent = filtered_df[filtered_df["anomaly"] == 1].copy()

            if "rule_threshold" in recent.columns:
                ratio = (recent["value"] - recent["rule_threshold"]) / recent[
                    "rule_threshold"
                ].replace(0, np.nan)
                recent["severity"] = "warning"
                recent.loc[ratio > 0.5, "severity"] = "critical"
            else:
                recent["severity"] = "warning"

            recent = recent[
                recent["severity"].isin(severity_filter)
                & recent["sensor_id"].isin(alert_sensor_filter)
            ]
            recent = recent.sort_values("timestamp").tail(RECENT_ANOMALIES_LIMIT)

            show_cols = [
                c
                for c in [
                    "timestamp", "sensor_id", "span_id", "value", "unit",
                    "rule_threshold", "anomaly_type", "severity",
                ]
                if c in recent.columns
            ]

            if not recent.empty:
                st.dataframe(recent[show_cols], key="alerts_table", width='stretch')
                csv_anomalies = recent[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Filtered Anomalies CSV",
                    data=csv_anomalies,
                    file_name=f"anomalies_{selected_bridge}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_anomalies",
                )
            else:
                st.info("No anomalies match the current filters.")
        else:
            st.success("No anomalies detected in the selected time range.")
    else:
        st.warning("No anomaly information available in the dataset.")

# ==================== TAB 3: HISTORICAL ANALYSIS ====================
# Static — only re-renders on full page rerun (sidebar changes).
with tab3:
    st.subheader("Historical Analysis")

    sensor_ids = sorted(filtered_df["sensor_id"].unique())
    selected_sensor = st.selectbox("Select sensor", sensor_ids, key="hist_sensor_select")

    show_anomalies_only = st.checkbox(
        "Show anomalies only", value=False, key="show_anomalies_only"
    )

    s_df = filtered_df[filtered_df["sensor_id"] == selected_sensor].sort_values("timestamp")

    if show_anomalies_only and "anomaly" in s_df.columns:
        s_df = s_df[s_df["anomaly"] == 1]

    if s_df.empty:
        st.warning("No data available for this selection.")
    else:
        unit = get_unit(s_df)
        threshold = get_threshold(s_df)

        fig = sensor_timeseries(
            s_df,
            selected_sensor,
            unit=unit,
            threshold=threshold,
            height=450,
            title=f"Sensor {selected_sensor} — Historical Trend",
        )
        st.plotly_chart(fig, key=f"hist_trend_{selected_sensor}", width='stretch')

        st.subheader("Value Distribution")
        hist = go.Figure()
        hist.add_trace(
            go.Histogram(
                x=s_df["value"],
                nbinsx=HISTOGRAM_BINS,
                marker_color=CHART_COLORS["primary"],
            )
        )
        hist.update_layout(
            xaxis_title=f"Value ({unit})" if unit else "Value",
            yaxis_title="Count",
            height=350,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor=CLR_DARK_BLUE,
            plot_bgcolor="#081040",
            font=dict(family="Montserrat", color=CLR_TEXT),
            xaxis=dict(gridcolor="#1A3080"),
            yaxis=dict(gridcolor="#1A3080"),
            uirevision=f"hist_{selected_sensor}",
        )
        st.plotly_chart(hist, key=f"hist_dist_{selected_sensor}", width='stretch')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{s_df['value'].mean():.2f}")
        c2.metric("Std", f"{s_df['value'].std():.2f}")
        c3.metric("Min", f"{s_df['value'].min():.2f}")
        c4.metric("Max", f"{s_df['value'].max():.2f}")

        # ---- Traffic Load Analysis ----
        if "traffic_load_proxy" in s_df.columns and s_df["traffic_load_proxy"].notna().any():
            st.markdown("---")
            st.subheader("Traffic Load Analysis")

            correlation = s_df[["value", "traffic_load_proxy"]].corr().iloc[0, 1]

            tcol1, tcol2 = st.columns(2)

            with tcol1:
                fig_traffic = go.Figure()
                fig_traffic.add_trace(
                    go.Scatter(
                        x=s_df["timestamp"],
                        y=s_df["traffic_load_proxy"],
                        mode="lines",
                        name="Traffic Load",
                        line=dict(color=CHART_COLORS["traffic"], width=2),
                    )
                )
                fig_traffic.update_layout(
                    title="Traffic Load Proxy Over Time",
                    xaxis_title="Time",
                    yaxis_title="Traffic Load (0–1)",
                    height=350,
                    template="plotly_dark",
                    paper_bgcolor=CLR_DARK_BLUE,
                    plot_bgcolor="#081040",
                    font=dict(family="Montserrat", color=CLR_TEXT),
                    xaxis=dict(gridcolor="#1A3080"),
                    yaxis=dict(gridcolor="#1A3080"),
                    uirevision=f"traffic_{selected_sensor}",
                )
                st.plotly_chart(
                    fig_traffic, key=f"traffic_load_{selected_sensor}", width='stretch'
                )

            with tcol2:
                color_col = "anomaly" if "anomaly" in s_df.columns else None
                fig_scatter = px.scatter(
                    s_df,
                    x="traffic_load_proxy",
                    y="value",
                    color=color_col,
                    title=f"Sensor vs Traffic Load (r = {correlation:.3f})",
                    labels={
                        "traffic_load_proxy": "Traffic Load",
                        "value": f"Value ({unit})",
                    },
                    color_discrete_map={
                        0: CHART_COLORS["primary"],
                        1: CHART_COLORS["anomaly"],
                    },
                    template="plotly_dark",
                )
                fig_scatter.update_layout(
                    height=350,
                    paper_bgcolor=CLR_DARK_BLUE,
                    plot_bgcolor="#081040",
                    font=dict(family="Montserrat", color=CLR_TEXT),
                    xaxis=dict(gridcolor="#1A3080"),
                    yaxis=dict(gridcolor="#1A3080"),
                    uirevision=f"scatter_{selected_sensor}",
                )
                st.plotly_chart(
                    fig_scatter, key=f"traffic_scatter_{selected_sensor}", width='stretch'
                )

            st.metric("Pearson Correlation with Traffic Load", f"{correlation:.3f}")

# ==================== TAB 4: CORRELATION ====================
# Static — only re-renders on full page rerun (sidebar changes).
with tab4:
    st.subheader("Cross-Sensor Correlation Analysis")

    pivot = filtered_df.pivot_table(
        index="timestamp",
        columns="sensor_id",
        values="value",
        aggfunc="first",
    ).ffill().bfill()

    if not pivot.empty and len(pivot.columns) > 1:
        corr = pivot.corr()

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=[
                    [0, CLR_ORANGE],
                    [0.5, CLR_DARK_BLUE],
                    [1, CLR_BLUE],
                ],
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10, "family": "Montserrat", "color": CLR_LIGHT_BLUE},
                colorbar=dict(
                    title="r", tickfont=dict(family="Montserrat", color=CLR_TEXT)
                ),
            )
        )
        fig_heat.update_layout(
            title="Sensor Correlation Matrix",
            height=500,
            template="plotly_dark",
            paper_bgcolor=CLR_DARK_BLUE,
            plot_bgcolor="#081040",
            font=dict(family="Montserrat", color=CLR_TEXT),
            uirevision="corr_heatmap",
        )
        st.plotly_chart(fig_heat, key="corr_heatmap", width='stretch')

        st.subheader("Strongest Correlations")
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        pairs = corr.where(mask).stack().sort_values(key=abs, ascending=False)
        for (s1, s2), r in pairs.head(10).items():
            st.markdown(f"- **{s1}** ↔ **{s2}**: {r:.3f}")
    else:
        st.warning("Not enough sensors for correlation analysis.")
