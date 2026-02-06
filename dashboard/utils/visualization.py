"""Shared Plotly chart helpers to avoid duplication in app.py."""

from __future__ import annotations
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

# Plotly layout matching the dashboard palette
_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0A1853",
    plot_bgcolor="#081040",
    font=dict(family="Montserrat", color="#FFFFFF"),
)


def get_unit(df: pd.DataFrame) -> str:
    if "unit" in df.columns and df["unit"].notna().any():
        return str(df["unit"].dropna().iloc[0])
    return ""


def get_threshold(df: pd.DataFrame) -> Optional[float]:
    if "rule_threshold" in df.columns and df["rule_threshold"].notna().any():
        return float(df["rule_threshold"].dropna().iloc[0])
    return None


def sensor_timeseries(
    df: pd.DataFrame,
    sensor_id: str,
    *,
    unit: str = "",
    threshold: Optional[float] = None,
    show_anomalies: bool = True,
    height: int = 350,
    title: Optional[str] = None,
) -> go.Figure:
    """Build a standard sensor time-series figure."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name=sensor_id,
            line=dict(color="#557FDE"),
        )
    )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#CAE0F5",
            annotation_text="Threshold",
            annotation_position="right",
            annotation_font=dict(family="Montserrat", color="#FFFFFF"),
        )

        # Highlight line segments above the threshold in orange
        above_vals = df["value"].where(df["value"] > threshold)
        if above_vals.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=above_vals,
                    mode="lines",
                    name="Above threshold",
                    line=dict(color="#E96547", width=2),
                    connectgaps=False,
                )
            )

    if show_anomalies and "anomaly" in df.columns:
        a_df = df[df["anomaly"] == 1]
        if not a_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=a_df["timestamp"],
                    y=a_df["value"],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(symbol="x", size=10, color="#E96547"),
                )
            )

    fig.update_layout(
        title=dict(text=title or f"Sensor {sensor_id}", font=dict(color="#CAE0F5")),
        xaxis=dict(title="Time", gridcolor="#1A3080", zerolinecolor="#1A3080"),
        yaxis=dict(
            title=f"Value ({unit})" if unit else "Value",
            gridcolor="#1A3080",
            zerolinecolor="#1A3080",
        ),
        height=height,
        showlegend=False,
        uirevision=sensor_id,
        transition={"duration": 80},
        **_DARK_LAYOUT,
    )

    return fig
