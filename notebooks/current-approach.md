# Multi-Sensor CNN Pipeline for Bridge Anomaly Detection & Forecasting

## Overview

This document summarizes the approach implemented in our two main notebooks for Structural Health Monitoring (SHM) of bridges using deep learning.

---

## Pipeline Architecture

### Two-Model Approach

We implement **two complementary 1D CNN models** with identical architectures but different labeling strategies:

| Model | Notebook | Purpose | Label Strategy |
|-------|----------|---------|----------------|
| **Unified CNN** | `Unified_CNN_Anomaly_Detection.ipynb` | Current-state anomaly **detection** | Label = anomaly within current window |
| **Predictive CNN** | `Predictive_CNN_Anomaly_Forecasting_CORRECTED.ipynb` | Future-state anomaly **forecasting** | Label = anomaly in future horizon |

---

## Data Pipeline

### Input Data
- **Source**: `ipmb_5sensors_30min_1_to_10hz.csv`
- **Sampling Rate**: 10 Hz (10 samples/second)
- **Duration**: 30 minutes (18,000 timesteps per sensor)
- **Total Rows**: 90,000 (5 sensors × 18,000 timesteps)

### Sensor Configuration
| Sensor Type | Sensors | Unit | Scale |
|-------------|---------|------|-------|
| Strain Gauge | SG_001, SG_002 | microstrain | ~100-200 |
| Accelerometer RMS | ACC_101, ACC_102 | g | ~0.01-0.03 |
| Temperature | TMP_201 | °C | ~20-30 |

### Data Preprocessing Steps

1. **Timestamp Parsing**: Convert ISO8601 timestamps to datetime
2. **Sensor Pivot**: Create wide-format table (rows=timestamps, columns=sensors)
3. **Missing Value Handling**: Forward-fill then backward-fill
4. **Per-Sensor Normalization**: StandardScaler per sensor channel
5. **Sliding Window Sequences**: Create fixed-length windows for CNN input

---

## Model Architecture

Both notebooks use **identical 1D CNN architecture**:

```
Input: (window_size, n_sensors) = (50, 5)

Conv Block 1: Conv1D(64, k=5) → BatchNorm → MaxPool(2) → Dropout(0.3)
Conv Block 2: Conv1D(128, k=3) → BatchNorm → MaxPool(2) → Dropout(0.3)
Conv Block 3: Conv1D(256, k=3) → BatchNorm → GlobalMaxPool → Dropout(0.3)

Dense: 128 → Dropout(0.3) → 64 → Dropout(0.3) → 1 (sigmoid)

Total Parameters: ~167,937
```

---

## Notebook 1: Unified CNN Anomaly Detection

### Purpose
Detect if the **current window** contains an anomaly (any sensor type).

### Labeling Strategy
```
Timeline: [----WINDOW----]
          Label = 1 if ANY timestep in window is anomalous
```

### Key Parameters
- **Window Size**: 50 timesteps (5 seconds)
- **Stride**: 10 timesteps (1 second)
- **Sequences Created**: 1,795

### Reported Performance
| Metric | Value |
|--------|-------|
| Accuracy | 98.89% |
| ROC-AUC | 0.9996 |
| Anomaly Recall | 100% |
| False Positives | 3/270 test samples |

### Outputs
- `artifacts/unified_cnn/unified_cnn_model.keras`
- `artifacts/unified_cnn/model_config.json`
- `artifacts/unified_cnn/scalers.pkl`
- `artifacts/unified_cnn/evaluation_metrics.json`

---

## Notebook 2: Predictive CNN Anomaly Forecasting

### Purpose
Predict if an anomaly will occur **in the future** (configurable horizon).

### Labeling Strategy
```
Timeline: [----OBSERVE----][--------HORIZON--------]
          |    5 sec      |     3-5 minutes       |

Label = 1 if ANY anomaly occurs in the HORIZON window
```

### Prediction Modes
| Mode | Horizon | Timesteps | Use Case |
|------|---------|-----------|----------|
| SHORT | 5 seconds | 50 | Immediate automated response |
| MEDIUM | 3 minutes | 1,800 | Alert operators |
| LONG | 5 minutes | 3,000 | Schedule maintenance |

### Early Warning Analysis
The notebook distinguishes between:
- **Forecast Horizon**: How far ahead the label window extends
- **Effective Lead Time**: Actual advance warning before anomaly onset

### Minimum Lead Time Constraint
To avoid counting near-zero lead-time "predictions" that are effectively detections, a configurable `MIN_LEAD_SECONDS` parameter (default: 30s) filters positive labels.

### Key Parameters
- **Window Size**: 50 timesteps (5 seconds of history)
- **Stride**: 10 timesteps
- **Default Mode**: MEDIUM (3 minutes)

### Outputs
- `artifacts/predictive_cnn/predictive_cnn_model_{MODE}.keras`
- `artifacts/predictive_cnn/config_{MODE}.json`
- `artifacts/predictive_cnn/scalers_{MODE}.pkl`

---

## Alignment with Research Questions (Chapter 1)

| RQ | Question | How Addressed |
|----|----------|---------------|
| **RQ1** | Multi-sensor CNN for heterogeneous sensors? | Unified CNN fuses strain, accel, temp → single prediction |
| **RQ2** | CNN vs Isolation Forest baseline? | *Not yet implemented* - needs baseline notebook |
| **RQ3** | Same architecture for detection vs forecasting? | Yes - identical CNN, different labeling strategies |
| **RQ4** | Early warning lead time and trade-offs? | Predictive CNN with configurable horizons & lead time analysis |

---

## Issues & Updates Needed

### Critical Issues

1. **Missing Baseline Comparison (RQ2)**
   - Need to implement Isolation Forest baseline for comparison
   - Add notebook: `Baseline_Isolation_Forest.ipynb`

2. **Data Leakage Risk in Predictive Model**
   - Current temporal split may not have anomalies in test set
   - Falls back to stratified random split (violates time series independence)
   - **Fix**: Ensure anomalies are distributed across train/val/test periods in data generation

3. **Synthetic Data Limitations**
   - Current data appears to be synthetic (generated patterns)
   - Real-world validation needed before deployment claims

### Recommended Improvements

1. **Add Isolation Forest Baseline**
   ```python
   # Create new notebook with:
   from sklearn.ensemble import IsolationForest
   # Compare detection accuracy, training time, inference time
   ```

2. **Cross-Validation for Predictive Model**
   - Implement time-series cross-validation (e.g., `TimeSeriesSplit`)
   - Avoid random stratification that leaks future information

3. **Explainability / Interpretability**
   - Add attention mechanism or Grad-CAM for sensor importance
   - Which sensors contribute most to predictions?

4. **Edge Deployment Preparation**
   - Model quantization (TensorFlow Lite)
   - Inference time benchmarking
   - Memory footprint analysis

5. **Documentation Consistency**
   - Unify parameter naming between notebooks
   - Add docstrings to all functions
   - Create shared config file for common parameters

---

## Storyline Alignment

The **Introduction (Chapter 1)** establishes a clear narrative:

1. **Problem**: Bridges need continuous monitoring; manual inspection is limited
2. **Challenge**: Multi-variate time series (MVTS) with heterogeneous sensors and EOCs
3. **Evolution**: From reactive detection to proactive prediction (Rytter's 4 levels)
4. **Approach**: Supervised multi-sensor learning with explicit lead-time evaluation
5. **Gap**: Most studies use single-modality; few address true prediction vs detection

**Our notebooks directly address this storyline**:
- Multi-sensor fusion ✓
- Detection AND forecasting ✓
- Lead-time analysis ✓
- Distinguishing forecast horizon from effective lead time ✓

**Missing from notebooks** (but mentioned in intro):
- Isolation Forest baseline comparison
- Discussion of EOC effects (temperature compensation, traffic load normalization)
- Edge computing considerations

---

## Quick Start

### Run Detection Model
```bash
cd Github/notebooks
jupyter notebook Unified_CNN_Anomaly_Detection.ipynb
# Run all cells
```

### Run Predictive Model
```bash
cd Github/notebooks
jupyter notebook Predictive_CNN_Anomaly_Forecasting_CORRECTED.ipynb
# Set PREDICTION_MODE = 'SHORT' / 'MEDIUM' / 'LONG'
# Run all cells
```

---

## File Structure

```
Github/
├── data/
│   └── ipmb_5sensors_30min_1_to_10hz.csv
├── notebooks/
│   ├── Unified_CNN_Anomaly_Detection.ipynb        # Detection
│   ├── Predictive_CNN_Anomaly_Forecasting_CORRECTED.ipynb  # Forecasting
│   └── artifacts/
│       ├── unified_cnn/          # Detection model outputs
│       └── predictive_cnn/       # Forecasting model outputs
```

---

## Summary

| Aspect | Detection (Notebook 1) | Forecasting (Notebook 2) |
|--------|------------------------|--------------------------|
| **Goal** | Is there an anomaly NOW? | Will there be an anomaly SOON? |
| **Architecture** | 1D CNN (167K params) | 1D CNN (167K params) |
| **Input** | 5s window (50 timesteps) | 5s window (50 timesteps) |
| **Output** | Binary (anomaly/normal) | Risk score (0-1) |
| **Label Source** | Current window | Future horizon (3-5 min) |
| **Key Metric** | ROC-AUC: 0.9996 | Lead time distribution |

The pipeline successfully demonstrates **RQ1, RQ3, and RQ4** from the thesis. **RQ2 (baseline comparison)** requires an additional Isolation Forest implementation.
