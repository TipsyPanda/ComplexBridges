# CNN-LSTM Anomaly Detection for Bridge Structural Health Monitoring

## Project Overview

This project implements a deep learning-based anomaly detection system for real-time structural health monitoring of bridges using strain sensor data. The system is designed for **live streaming inference**, detecting damage states from continuous sensor measurements.

**Primary Use Case**: Detecting structural damage in bridges from strain gauge measurements in real-time, with temperature compensation for environmental effects.

---

## Dataset: HBTA (Hell Bridge Test Arena)

### Source
- **File**: `data/data_100Hz.h5` (HDF5 format)
- **Collection Period**: September 22-28, 2020
- **Sampling Rate**: 100 Hz

### Structure
| Attribute | Value |
|-----------|-------|
| Total Recordings | 50 |
| Damage States | 9 (UDS + DS1-DS8) |
| Sensors | 15 strain gauges (8 SB + 7 SC) |
| Temperature Range | 9°C - 22°C |

### Damage State Distribution

| State | Description | Recordings | Severity |
|-------|-------------|------------|----------|
| UDS | Undamaged State | 10 | 0 (baseline) |
| DS1 | Damage State 1 | 5 | 1 (lightest) |
| DS2 | Damage State 2 | 5 | 2 |
| DS3 | Damage State 3 | 5 | 3 |
| DS4 | Damage State 4 | 5 | 4 |
| DS5 | Damage State 5 | 5 | 5 |
| DS6 | Damage State 6 | 5 | 6 |
| DS7 | Damage State 7 | 5 | 7 |
| DS8 | Damage State 8 | 5 | 8 (most severe) |

### Sensor Configuration

**SB Sensors (8 channels)**: SB01-SB08, measuring strain in x-direction
**SC Sensors (7 channels)**: SC01-SC07, measuring strain in y-direction

Total: **15 input channels** per timestep

### Recording Naming Convention
```
MVS_P{phase}_{damage_state}_{mode}_{direction}_{number}
```
- **Phase**: P1 or P2 (test phase)
- **Damage State**: UDS, DS1-DS8
- **Mode**: NM (normal mode) or SM (special mode)
- **Direction**: Y or Z
- **Number**: Recording sequence number

### Metadata per Recording
Each recording includes HDF5 attributes:
- `date`: Recording date (YYYY-MM-DD)
- `time`: Recording time
- `temperature`: Ambient temperature in °C

---

## Data Preprocessing Pipeline

### 1. Windowing
```
Raw signal → Fixed-size windows → Training samples
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window Size | 512 samples | ~5.12 seconds at 100Hz, power of 2 for FFT efficiency |
| Overlap | 50% | Increases training data, captures transitions |
| Max Windows/Recording | 100 | Balances dataset size and memory |

### 2. Per-Window Normalization
```python
normalized = (window - mean) / std
```
- Computed independently per window
- Removes DC offset variations (temperature drift, sensor drift)
- Preserves dynamic signal content (vibration patterns)
- **Critical for live inference**: No dependency on historical statistics

### 3. STFT (Short-Time Fourier Transform)

Converts time-domain signals to frequency-domain spectrograms.

| Parameter | Value | Resulting Shape |
|-----------|-------|-----------------|
| n_fft | 64 | 33 frequency bins (0-50 Hz) |
| hop_length | 16 | 75% overlap between frames |
| Window | Hann | Smooth spectral estimation |

**Transformation**:
```
Input:  (512 timesteps, 15 channels)
Output: (29 frames, 33 freq_bins, 15 channels)
```

**Post-processing**:
- Log scale: `log(1 + magnitude)` for better dynamic range
- Per-spectrogram normalization: zero mean, unit variance

### 4. Temperature Normalization
```python
temp_normalized = (temp_celsius - (-10)) / (40 - (-10))
```
- Range: -10°C to 40°C mapped to [0, 1]
- Allows deployment flexibility beyond training data range (9-22°C)

---

## Model Architecture

### Overview
**Type**: CNN-LSTM with Temperature Conditioning
**Input**: Spectrogram (29 × 33 × 15) + Temperature scalar
**Output**: Anomaly score [0, 1]

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    SPECTROGRAM INPUT                        │
│                   (29 frames × 33 freq × 15 ch)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    2D CNN BLOCKS                            │
│  Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(1,2)         │
│  Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(1,2)         │
│  Conv2D(64, 3×3) → BatchNorm → ReLU                        │
│                                                             │
│  Output: (29 frames × 8 freq × 64 filters)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    RESHAPE                                  │
│              (29, 8×64) = (29, 512)                         │
│              Preserves temporal dimension                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LSTM STACK                               │
│  LayerNorm                                                  │
│  LSTM(64, return_sequences=True, unidirectional)           │
│  LayerNorm                                                  │
│  LSTM(32, return_sequences=False, unidirectional)          │
│  Dropout(0.4)                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               TEMPERATURE CONDITIONING                      │
│                                                             │
│  Temperature (1,) → Dense(8, ReLU) → Concatenate           │
│                                                             │
│  Merged: LSTM output (32) + Temp embedding (8) = (40)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 CLASSIFICATION HEAD                         │
│  Dense(64, ReLU) → Dropout(0.4)                            │
│  Dense(32, ReLU) → Dropout(0.3)                            │
│  Dense(1, Sigmoid) → Anomaly Score [0, 1]                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Unidirectional (Causal) LSTM
- **Why**: Enables live streaming inference
- **Constraint**: Only uses past data, no future lookahead
- **Trade-off**: Slightly lower accuracy than bidirectional, but necessary for real-time

#### 2. MaxPooling on Frequency Axis Only
- **Why**: Reduces frequency dimension while preserving temporal resolution
- **Benefit**: LSTM receives full temporal sequence for pattern learning

#### 3. Temperature as Conditional Input
- **Why**: Bridge material properties change with temperature
- **Implementation**: Embedded through small dense layer, concatenated before classification
- **Benefit**: Model learns temperature-damage interaction

#### 4. L2 Regularization + Dropout
- **Values**: L2 weight = 0.005, Dropout = 0.3-0.4
- **Why**: Prevents overfitting on limited dataset (50 recordings)

### Model Parameters
| Component | Parameters |
|-----------|------------|
| Conv2D layers | ~60,000 |
| LSTM layers | ~50,000 |
| Dense layers | ~5,000 |
| **Total** | **~115,000** |

---

## Training Configuration

### Data Splitting Strategy

**Method**: GroupShuffleSplit by Recording ID

```python
GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

**Critical Requirement**: All windows from one recording stay together (train OR validation, never split)

**Why This Matters**:
- Windows overlap 50% → adjacent windows share data
- Splitting windows randomly → data leakage → inflated accuracy
- Splitting by recording → honest evaluation of generalization

**Split Result**:
- Training: ~40 recordings
- Validation: ~10 recordings

### Class Imbalance Handling

```python
class_weights = {
    0: total / (2 * n_undamaged),  # Weight for undamaged
    1: total / (2 * n_damaged)     # Weight for damaged
}
```
- Undamaged: 10 recordings (20%)
- Damaged: 40 recordings (80%)
- Weights compensate for imbalance

### Optimizer Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 (initial) |
| Gradient Clipping | norm = 1.0 |
| Loss | Binary Crossentropy |
| Label Smoothing | 0.1 |

### Callbacks

1. **EarlyStopping**
   - Monitor: `val_loss`
   - Patience: 20 epochs
   - Restore best weights: Yes

2. **ReduceLROnPlateau**
   - Monitor: `val_loss`
   - Factor: 0.5
   - Patience: 7 epochs
   - Min LR: 1e-6

### Data Augmentation (Time-Domain Only)

Applied with random probability to training data:

| Augmentation | Probability | Parameters |
|--------------|-------------|------------|
| Jitter (Gaussian noise) | 70% | σ = 0.03 |
| Scaling | 70% | σ = 0.15 |
| Magnitude Warp | 50% | σ = 0.2, knots = 4 |

**Augmentation Factor**: 3× (original data expanded 4×)

---

## Results

### Final Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 88.5% |
| **F1 Score** | 90.7% |
| **Precision** | 87.9% |
| **Recall** | 93.8% |

### Confusion Matrix

```
                 Predicted
              Undamaged  Damaged
Actual  Undamaged   TN       FP
        Damaged     FN       TP

Precision = TP / (TP + FP) = 87.9%
Recall    = TP / (TP + FN) = 93.8%
```

### Training History

- **Epochs Trained**: 47 (early stopping triggered)
- **Best Epoch**: 27
- **Final Learning Rate**: Reduced from 0.001

### Ablation Study

| Configuration | Accuracy |
|--------------|----------|
| Time-domain CNN-LSTM (baseline) | ~70% |
| + STFT transformation | 81% |
| + Temperature conditioning | **88.5%** |

**Key Finding**: Temperature conditioning provided +7.5% accuracy improvement, indicating significant environmental influence on sensor readings.

---

## Live Inference System

### LiveAnomalyDetector Class

Designed for real-time streaming deployment:

```python
detector = LiveAnomalyDetector(
    model_path='anomaly_detector.keras',
    window_size=512,
    anomaly_threshold=0.5,
    ema_alpha=0.3,
    use_stft=True,
    use_temperature=True
)

# Set current environmental conditions
detector.set_temperature(15.0)  # Celsius

# Processing loop
while True:
    samples = read_sensors()  # (n_samples, 15)
    results = detector.process(samples)
    for result in results:
        if result['is_anomaly']:
            trigger_alert(result['anomaly_score'])
```

### Features

1. **Sliding Window Buffer**: Accumulates samples until complete window
2. **Per-Window Normalization**: No historical statistics needed
3. **STFT Transformation**: Real-time frequency analysis
4. **Temperature Compensation**: Environmental adjustment
5. **EMA Smoothing**: Reduces prediction jitter (α = 0.3)
6. **Configurable Threshold**: Default 0.5, adjustable for sensitivity

### Output per Window

```python
{
    'anomaly_score': 0.73,      # Raw model output
    'smoothed_score': 0.65,    # EMA-filtered score
    'is_anomaly': True,        # Thresholded decision
    'window_id': 142,          # Sequence number
    'temperature': 15.0        # Current temperature
}
```

### Latency Considerations

| Component | Time (approximate) |
|-----------|-------------------|
| Window accumulation | 5.12 seconds |
| STFT computation | ~5 ms |
| Model inference | ~20 ms |
| **Total latency** | ~5.15 seconds |

---

## Limitations

### 1. Binary Classification Only (Current Implementation)

**Issue**: The dataset contains 9 damage severity levels (UDS, DS1-DS8), but the current model reduces this to binary (damaged vs. undamaged).

**Impact**:
- Cannot distinguish between minor (DS1) and severe (DS8) damage
- No damage progression tracking
- No remaining useful life prediction

**Potential Solution**: Retrain as multi-class classification or regression for damage severity estimation.

### 2. Limited Dataset Size

**Issue**: Only 50 recordings total (10 undamaged, 40 damaged)

**Impact**:
- Limited generalization to unseen conditions
- Validation on only ~10 recordings
- May not capture full variability of real-world conditions

**Potential Solution**: Collect more data, especially edge cases and different environmental conditions.

### 3. Narrow Temperature Range

**Issue**: Training data covers only 9°C - 22°C

**Impact**:
- Model extrapolation beyond this range is untested
- Winter (< 0°C) or summer (> 30°C) conditions may behave differently
- Temperature normalization assumes linear relationship

**Potential Solution**: Collect data across full annual temperature range.

### 4. No Time-of-Day Information

**Issue**: All recordings have `00:00:00` as timestamp (time not recorded)

**Impact**:
- Cannot account for diurnal variations (traffic patterns, thermal expansion cycles)
- Day/night behavioral differences not captured

### 5. Single Bridge Structure

**Issue**: All data from one specific bridge

**Impact**:
- Model is bridge-specific, not generalizable
- Different bridge types (suspension, truss, beam) would need separate models
- Transfer learning to new structures not validated

### 6. Detection vs. Prediction

**Issue**: Current model detects **existing** damage, does not **predict** future damage

**Impact**:
- No early warning capability
- Cannot estimate time-to-failure
- No prognostics for maintenance scheduling

**Potential Solution**:
- Reframe as regression on damage severity (0-8 scale)
- Train on temporal sequences showing damage progression
- Implement trend analysis on damage severity predictions

### 7. Fixed Window Size

**Issue**: 512-sample (5.12 second) windows are fixed

**Impact**:
- Very short transient damage events may be missed
- Very slow degradation patterns may not be captured within single window
- No multi-scale analysis

### 8. Sensor Failure Handling

**Issue**: Model assumes all 15 sensors provide valid data

**Impact**:
- Single sensor failure could cause false positives/negatives
- No graceful degradation capability
- No sensor health monitoring

### 9. Class Imbalance

**Issue**: 80% damaged vs 20% undamaged in dataset

**Impact**:
- Model may be biased toward predicting "damaged"
- Class weights help but don't fully solve
- Real-world ratio likely inverse (mostly healthy)

### 10. Environmental Factors

**Issue**: Only temperature is considered

**Impact**:
- Humidity effects not captured
- Wind loading correlation not modeled
- Precipitation effects unknown
- Traffic loading not explicitly modeled

---

## File Structure

```
hell/
├── data/
│   └── data_100Hz.h5           # Raw HDF5 dataset
├── src/
│   └── train_cnn_lstm.py       # Main training script
├── artifacts/
│   ├── anomaly_detector.keras  # Trained model
│   └── learning_curves.png     # Training visualization
└── docs/
    └── PROJECT_DOCUMENTATION.md # This file
```

---

## Usage

### Training
```bash
# Full training
python src/train_cnn_lstm.py

# Quick test (10 epochs, limited data)
python src/train_cnn_lstm.py --quick

# Medium test (100 epochs, moderate data)
python src/train_cnn_lstm.py --medium

# With live simulation
python src/train_cnn_lstm.py --live
```

### Configuration

Edit `CONFIG` dictionary in `train_cnn_lstm.py`:

```python
CONFIG = {
    'window_size': 512,
    'window_overlap': 0.5,
    'val_split': 0.2,
    'epochs': 100,
    'batch_size': 64,
    'use_stft': True,
    'n_fft': 64,
    'hop_length': 16,
    'use_temperature': True,
    'temp_min': -10,
    'temp_max': 40,
}
```

---

## Future Work

1. **Multi-class/Regression**: Predict damage severity (0-8) instead of binary
2. **Damage Localization**: Identify which sensors detect damage (attention mechanisms)
3. **Uncertainty Quantification**: Bayesian approaches for confidence estimation
4. **Transfer Learning**: Pre-train on multiple bridges, fine-tune on target
5. **Online Learning**: Adapt model to changing baseline over time
6. **Multi-scale Analysis**: Variable window sizes for different damage types
7. **Sensor Fusion**: Incorporate accelerometer, displacement, or other sensor types
8. **Explainability**: SHAP/LIME analysis for model interpretability

---

## References

- HBTA Dataset: Hell Bridge Test Arena structural health monitoring data
- Architecture inspired by: CNN-LSTM approaches for time-series classification
- STFT implementation: Standard signal processing techniques for vibration analysis

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial binary classification model |
| 1.1 | 2024 | Added STFT frequency domain transformation |
| 1.2 | 2024 | Added temperature conditioning (+7.5% accuracy) |

---

*Document generated for the Hell Bridge Structural Health Monitoring Project*
