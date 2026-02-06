# Bridge Monitor (Streamlit)

Local, Python-first anomaly dashboard for bridge sensor time series.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/streamlit_app.py
```

## Project structure

```
dashboard/
├── app.py              # Main Streamlit application
├── config.py           # Paths, thresholds, palette, CSS
└── utils/
    └── visualization.py  # Shared Plotly chart helpers
data/
└── ipmb_5sensors_30min_1_to_10hz.csv  # Sensor readings
requirements.txt
```

## Dashboard overview

The dashboard visualises accelerometer (ACC), strain gauge (SG), and temperature (TMP) sensor data from instrumented bridges. It is built with **Streamlit** and **Plotly** and provides four tabs:

| Tab | Purpose |
|-----|---------|
| **Live Monitoring** | Real-time playback of sensor time series with per-sensor plots in a 2-column grid |
| **Alerts** | Anomaly counts, severity breakdown (critical / warning), and a filterable table of recent anomalies |
| **Historical Analysis** | Single-sensor trend, value distribution histogram, traffic-load correlation scatter & time series |
| **Correlation** | Cross-sensor Pearson correlation heatmap with ranked pair list |

### Sidebar controls

- **Bridge & span selection** — filter by bridge ID and span(s)
- **Time range slider** — restrict the analysis window (minutes from start)
- **Playback** — Play / Pause / Reset / Skip with configurable speed and display window
- **Custom thresholds** — override default rule-based thresholds per sensor type
- **CSV export** — download the currently filtered dataset

### Key defaults (config.py)

| Parameter | Value |
|-----------|-------|
| Playback refresh | 100 ms |
| Playback speed | 50 records / update |
| Display window | 500 records |
| ACC threshold | 0.07 g |
| SG threshold | 200 microstrain |
| TMP threshold | 35 °C |

### Health score

A composite 0–100 score computed from three components:

- **Anomaly rate** (50 pts) — penalises high anomaly frequency
- **Threshold headroom** (30 pts) — how far readings stay below thresholds
- **Sensor stability** (20 pts) — coefficient of variation across sensors

Colour-coded as Good (≥ 80), Fair (≥ 60), Poor (≥ 40), or Critical (< 40).
