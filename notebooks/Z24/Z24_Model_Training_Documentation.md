# Z24 Bridge Anomaly Prediction - Model Training Documentation

## Project Overview

This project implements a deep learning-based anomaly detection system for the **Z24 Bridge**, a benchmark dataset in structural health monitoring (SHM). The system predicts structural anomalies (damage) in the bridge using accelerometer and temperature sensor data.

### Objective

Given **10 seconds of sensor history** (accelerometer vibrations + temperature), predict the **probability of an anomaly occurring in the next 60 seconds**.

---

## Dataset: Z24 Bridge

The Z24 Bridge was a post-tensioned concrete highway bridge in Switzerland that was progressively damaged in a controlled manner before demolition in 1998. It has become one of the most important benchmark datasets in structural health monitoring research.

### Data Specifications

| Parameter | Value |
|-----------|-------|
| **Sampling Rate** | 50 Hz (downsampled from 100 Hz) |
| **Nyquist Frequency** | 25 Hz |
| **Input Window** | 10 seconds (500 timesteps) |
| **Prediction Window** | 60 seconds (30 output timesteps at 100 Hz) |
| **Channels** | 6 (5 accelerometers + 1 temperature) |

### Channel Configuration

- **CH0-CH4**: Accelerometer channels capturing structural vibrations
- **TEMP**: Deck temperature from EMS environmental sensors

### Data Split

| Split | Samples | Shape |
|-------|---------|-------|
| Training | 10,953 | (10953, 500, 6) |
| Validation | 3,651 | (3651, 500, 6) |
| Test | 3,651 | (3651, 500, 6) |

### Data Source

The dataset was created from EMS+PDT_v2_transitions data, with 100,000 samples per class and 5 segments. The data loader (`z24_loader.py`) handles the preprocessing and loading.

---

## Model Architecture: CNNLSTMv2

The model uses a **dual-pathway architecture** that separately processes accelerometer and temperature data before fusion.

### Architecture Diagram

```
Input (500 timesteps x 6 channels)
        |
        +---> [Accelerometer Path: 5 channels]
        |           |
        |           v
        |     Conv1d(5, 32, k=7) + BN + ReLU + MaxPool
        |           |
        |           v
        |     Conv1d(32, 64, k=5) + BN + ReLU + MaxPool
        |           |
        |           v
        |     Conv1d(64, 128, k=3) + BN + ReLU + MaxPool
        |           |
        |           v
        |     LSTM(128, 64, 2 layers)
        |           |
        |           v
        |     LayerNorm -> Accel Features (64-dim)
        |
        +---> [Temperature Path: 1 channel]
        |           |
        |           v
        |     Mean Pooling (temporal)
        |           |
        |           v
        |     MLP: Linear(1, 16) + ReLU + Linear(16, 16) + ReLU
        |           |
        |           v
        |     LayerNorm -> Temp Features (16-dim)
        |
        +---> Concatenate [Accel(64) + Temp(16)] = 80-dim
                    |
                    v
              Linear(80, 30) + Sigmoid
                    |
                    v
              Output: 30 timestep predictions
```

### Architecture Details

| Component | Configuration |
|-----------|--------------|
| **CNN Layers** | 3 convolutional layers (32, 64, 128 filters) |
| **Kernel Sizes** | 7, 5, 3 (progressively smaller) |
| **Pooling** | MaxPool1d(2) after each conv layer |
| **LSTM** | 2 layers, 64 hidden units, bidirectional=False |
| **Temperature MLP** | 2-layer MLP (1 -> 16 -> 16) |
| **Dropout** | 0.4 throughout |
| **Total Parameters** | 122,446 |

### Design Rationale

1. **Separate Pathways**: Accelerometer data contains high-frequency vibration patterns while temperature is a slowly-varying scalar. Processing them separately allows each pathway to specialize.

2. **CNN for Feature Extraction**: The CNN layers extract local temporal patterns and frequency-domain features from the accelerometer signals.

3. **LSTM for Temporal Dependencies**: The LSTM captures long-range temporal dependencies in the compressed feature representations.

4. **Temperature Mean Pooling**: Since temperature varies slowly over 10 seconds, the mean value is sufficient representation.

---

## Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 128 |
| **Learning Rate** | 1e-3 (max) |
| **Weight Decay** | 1e-2 |
| **Optimizer** | AdamW |
| **Loss Function** | Binary Cross-Entropy (BCE) |
| **Epochs** | 80 (full training) |
| **Early Stopping Patience** | 25 epochs |

### Learning Rate Schedule

**OneCycleLR** scheduler with:
- 30% warmup phase
- Cosine annealing strategy
- Initial LR: max_lr / 25
- Final LR: max_lr / 1000

### Data Augmentation

```python
class TimeSeriesAugmentation:
    - add_noise(x, noise_factor=0.03)  # Gaussian noise
    - scale(x, scale_range=(0.8, 1.2))  # Random scaling
    - channel_dropout(x, p=0.1)  # Random channel dropout
```

### Regularization Techniques

1. **Dropout (0.4)**: Applied after each CNN layer and before final FC
2. **Weight Decay (1e-2)**: L2 regularization via AdamW
3. **Gradient Clipping**: max_norm=1.0
4. **Layer Normalization**: Applied to LSTM and MLP outputs
5. **Batch Normalization**: Applied after each CNN layer

---

## Training Results

### Loss Curves

The training converged smoothly over 80 epochs:

| Metric | Value |
|--------|-------|
| **Best Validation Loss** | 0.0448 |
| **Final Training Loss** | 0.0546 |
| **Train-Val Gap** | ~0.01 (minimal overfitting) |

### Training Progression

```
Epoch   1/80 | Train: 0.6179 | Val: 0.4849
Epoch  10/80 | Train: 0.1502 | Val: 0.1516
Epoch  25/80 | Train: 0.1117 | Val: 0.0974
Epoch  50/80 | Train: 0.0696 | Val: 0.0626
Epoch  80/80 | Train: 0.0546 | Val: 0.0553

Best model restored at val_loss=0.0448
```

---

## Test Results

### Classification Metrics (Threshold = 0.5)

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.74% |
| **Precision** | 98.61% |
| **Recall** | 97.07% |
| **F1 Score** | 97.83% |
| **AUC-ROC** | 99.67% |

### Confusion Matrix

|  | Predicted Healthy | Predicted Anomaly |
|--|-------------------|-------------------|
| **Actual Healthy** | TN (high) | FP (low) |
| **Actual Anomaly** | FN (low) | TP (high) |

The model achieves excellent separation between healthy and damaged states with minimal false positives and false negatives.

### ROC Curve

The AUC-ROC of **0.9967** indicates near-perfect discrimination capability. The ROC curve hugs the top-left corner, demonstrating high true positive rate with low false positive rate across all thresholds.

---

## Frequency Analysis

### 50 Hz Sampling Advantage

With 50 Hz sampling, the model can capture frequencies up to **25 Hz** (Nyquist frequency), which is critical for structural health monitoring.

### Modal Frequencies of Interest

The Z24 Bridge has known modal frequencies at:
- **3.9 Hz**: First bending mode
- **5.1 Hz**: Second mode
- **9.8 Hz**: Third mode
- **10.3 Hz**: Fourth mode

All these frequencies are well below the 25 Hz Nyquist limit, ensuring accurate capture of structural dynamics.

### Spectral Analysis

Power Spectral Density (PSD) analysis reveals:
- Clear separation between healthy and damaged states in frequency content
- Damaged states show shifts in modal frequency peaks
- Temperature distribution differs between healthy and damaged periods

---

## Model Interpretability

### Feature Visualization (t-SNE)

Three-panel visualization shows:

1. **Vibration Features (LSTM output)**: Clear clustering of healthy (green) vs anomaly (red) samples in the 64-dimensional latent space
2. **Temperature Features**: Latent temperature representation shows organization by input temperature value
3. **Final Fused Representation**: Combined 80-dimensional features show excellent class separation

### Saliency Analysis

Gradient-based saliency maps reveal:
- The model focuses on specific temporal regions within the 10-second window
- Higher gradients indicate time periods most influential for the prediction
- Vibration patterns during certain phases are more diagnostic

---

## Baseline Comparison: Autoencoder

An unsupervised **Convolutional Autoencoder** baseline was trained for comparison.

### Autoencoder Architecture

```
Encoder:
  Conv1d(6, 32, k=7, s=2)  # 500 -> 250
  Conv1d(32, 16, k=5, s=2) # 250 -> 125
  Conv1d(16, 8, k=3, s=2)  # 125 -> 63

Decoder:
  ConvTranspose1d(8, 16)   # 63 -> 125
  ConvTranspose1d(16, 32)  # 125 -> 250
  ConvTranspose1d(32, 6)   # 250 -> 500
```

### Autoencoder Training

- **Training Data**: Healthy samples only (semi-supervised approach)
- **Loss**: MSE reconstruction loss
- **Epochs**: 30

### Autoencoder Results

| Metric | CNNLSTMv2 | Autoencoder Baseline |
|--------|-----------|---------------------|
| **AUC-ROC** | 0.9967 | Lower (baseline) |

The autoencoder provides a reconstruction error distribution that separates healthy and damaged samples, but with less discrimination than the supervised CNNLSTMv2 model.

---

## Version Comparison

| Aspect | v01 | v02 (Current) |
|--------|-----|---------------|
| **Sample Rate** | 10 Hz | 50 Hz |
| **Input Timesteps** | 100 (10s) | 500 (10s) |
| **Channels** | 5 (accel only) | 6 (5 accel + temp) |
| **Nyquist Frequency** | 5 Hz | 25 Hz |
| **Modal Frequency Coverage** | Partial | Full |

The v02 model represents significant improvements by:
1. Capturing all relevant modal frequencies (up to 25 Hz)
2. Including temperature as a feature (known to affect bridge dynamics)
3. Using separate processing pathways for different data modalities

---

## Limitations

### Data Limitations

1. **Single Bridge**: The model is trained on data from a single bridge (Z24). Generalization to other structures is not validated.

2. **Controlled Damage**: The Z24 dataset contains progressively introduced damage under controlled conditions. Real-world damage patterns may differ.

3. **Limited Damage Types**: The dataset primarily captures specific damage scenarios introduced during the demolition study.

4. **Temporal Correlation**: The data may contain temporal correlations (adjacent samples are similar), potentially inflating performance metrics.

5. **Class Balance**: The dataset is artificially balanced (100,000 samples per class). Real-world scenarios have highly imbalanced class distributions.

### Model Limitations

1. **Fixed Input Window**: The model requires exactly 10 seconds of data. Shorter or longer windows require retraining or interpolation.

2. **Fixed Sampling Rate**: Input must be at 50 Hz. Different sampling rates require resampling.

3. **No Uncertainty Quantification**: The model outputs point predictions without confidence intervals or uncertainty estimates.

4. **Black Box Nature**: While saliency maps provide some interpretability, the model's decision process is not fully transparent.

5. **Temperature Simplification**: Using mean temperature over the window may lose transient temperature effects.

### Deployment Limitations

1. **Computational Requirements**: Requires GPU for efficient inference (CUDA device used during training).

2. **No Online Learning**: The model cannot adapt to new data without retraining.

3. **Threshold Sensitivity**: Classification performance depends on the chosen threshold (default 0.5).

4. **No Anomaly Type Classification**: The model only predicts binary anomaly/healthy, not the type or severity of damage.

### Research Limitations

1. **No Cross-Validation**: Results are from a single train/val/test split.

2. **Limited Hyperparameter Search**: The current configuration may not be optimal.

3. **No Comparison with Classical SHM Methods**: Performance is not benchmarked against traditional statistical process control or modal analysis methods.

4. **No Transfer Learning Evaluation**: Ability to transfer to different bridge types is unknown.

---

## Model Artifacts

### Saved Model

- **Path**: `/home/yanni/models/z24_cnn_lstm_v2_20260115_091659.pt`
- **Size**: 1,477.9 KB
- **Format**: PyTorch checkpoint

### Checkpoint Contents

```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': {
        'n_features': 100,
        'hidden_size': 64,
        'output_steps': 30,
        'dropout': 0.4,
        'sample_rate_hz': 50,
        'input_steps': 500,
    },
    'metrics': {...},
    'train_losses': [...],
    'val_losses': [...],
    'best_val_loss': 0.0448,
}
```

---

## Dependencies

```python
numpy
matplotlib
seaborn
torch (2.9.0+cu126)
scipy
sklearn
z24_loader (custom module)
```

---

## Usage

### Loading the Model

```python
import torch
from model import CNNLSTMv2

# Load checkpoint
checkpoint = torch.load('/home/yanni/models/z24_cnn_lstm_v2_20260115_091659.pt')

# Initialize model
model = CNNLSTMv2(
    n_accel_features=5,
    hidden_size=64,
    output_steps=30,
    dropout=0.4
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
# Input shape: (batch, 500, 6) - 500 timesteps, 6 channels
predictions = model(input_tensor)  # Output: (batch, 30, 1)
```

### Data Requirements

- **Input Shape**: `(batch_size, 500, 6)`
- **Channel Order**: `[CH0, CH1, CH2, CH3, CH4, TEMP]`
- **Sample Rate**: 50 Hz
- **Normalization**: Data should be preprocessed consistently with training data

---

## Future Work

1. **Uncertainty Quantification**: Add Bayesian layers or ensemble methods for confidence estimation

2. **Multi-Task Learning**: Extend to predict damage type and severity

3. **Transfer Learning**: Evaluate and improve generalization to other bridge types

4. **Online Adaptation**: Implement continual learning for deployment scenarios

5. **Attention Mechanisms**: Replace LSTM with Transformer architecture for improved interpretability

6. **Edge Deployment**: Optimize for embedded devices and real-time monitoring

---

## References

- Z24 Bridge Benchmark Dataset (1998)
- Original Z24 Bridge SHM studies
- Deep Learning for Structural Health Monitoring literature

---

*Document generated: 2026-01-15*
*Model Version: CNNLSTMv2 (v2 - 50Hz + Temperature)*
