# -*- coding: utf-8 -*-
"""
train_cnn_lstm.py
CNN-LSTM for bridge damage/anomaly detection.

Designed for LIVE STREAMING inference:
- Unidirectional (causal) LSTM - only uses past data, no future lookahead
- Outputs anomaly score (0-1) for real-time monitoring
- STFT frequency domain features for better damage detection
- Stateful LSTM option for continuous streaming

Key features:
1. Split by RECORDING (not window) to prevent data leakage
2. STFT transformation to frequency domain (spectrograms)
3. Causal architecture suitable for deployment
4. Anomaly score output with configurable threshold
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense,
                                      Dropout, BatchNormalization,
                                      GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': Path(__file__).parent.parent / 'data' / 'data_100Hz.h5',
    'output_dir': Path(__file__).parent.parent / 'artifacts',
    'window_size': 512,       # ~5 seconds at 100Hz (power of 2 for FFT efficiency)
    'window_overlap': 0.5,    # 50% overlap for more training data
    'val_split': 0.2,         # 20% of RECORDINGS for validation
    'epochs': 100,
    'batch_size': 64,         # Larger batch for stability
    'random_state': 42,
    'max_windows_per_recording': 100,  # More windows
    'augmentation_factor': 3,  # More augmentation

    # STFT parameters
    'use_stft': True,         # Convert to frequency domain
    'n_fft': 64,              # FFT window size (64 samples = 0.64s at 100Hz)
    'hop_length': 16,         # Hop between FFT windows (75% overlap)
    'sample_rate': 100,       # Hz

    # Temperature conditioning
    'use_temperature': True,  # Include temperature as model input
    'temp_min': -10,          # Min temp for normalization (°C)
    'temp_max': 40,           # Max temp for normalization (°C)
}

# Strain sensors only (most discriminative based on analysis)
STRAIN_SENSORS = {
    'SB': ['SB01', 'SB02', 'SB03', 'SB04', 'SB05', 'SB06', 'SB07', 'SB08'],
    'SC': ['SC01', 'SC02', 'SC03', 'SC04', 'SC05', 'SC06', 'SC07']
}

# ============================================================================
# DATA LOADING
# ============================================================================

def get_recording_label(recording_name, binary=True):
    """Extract damage state label from recording name.

    Format: MVS_P2_<DAMAGE_STATE>_<MODE>_<DIR>_<NUM>

    Args:
        recording_name: e.g., 'MVS_P2_UDS_NM_Z_01'
        binary: if True, return 0 for UDS, 1 for any damage state

    Returns:
        label: 0/1 for binary, or 'UDS'/'DS1'/etc for multiclass
    """
    parts = recording_name.split('_')
    damage_state = parts[2]  # UDS, DS1, DS2, ..., DS8

    if binary:
        return 0 if damage_state == 'UDS' else 1
    return damage_state


def load_strain_data(h5_file, recording):
    """Load strain sensor data for a single recording."""
    data_df = pd.DataFrame()

    # SB sensors (x-direction)
    for sensor in STRAIN_SENSORS['SB']:
        col_name = f"{sensor}/x"
        data_df[col_name] = np.array(h5_file[recording]['strain'][sensor]['x']).flatten()

    # SC sensors (y-direction)
    for sensor in STRAIN_SENSORS['SC']:
        col_name = f"{sensor}/y"
        data_df[col_name] = np.array(h5_file[recording]['strain'][sensor]['y']).flatten()

    return data_df


def create_windows(data, window_size, overlap, max_windows=None):
    """Create overlapping windows from time series data."""
    step = int(window_size * (1 - overlap))
    n_windows = (len(data) - window_size) // step + 1

    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    windows = np.zeros((n_windows, window_size, data.shape[1]), dtype=np.float32)
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        windows[i] = data[start:end]

    return windows


def normalize_windows_per_window(X):
    """Normalize each window independently (subtract mean, divide by std).

    This is better for dynamic/vibration signals where the DC offset
    may vary due to temperature but the dynamic content is the signal.
    """
    # X shape: (n_windows, timesteps, features)
    # Normalize along timesteps axis for each window and feature
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std


# ============================================================================
# STFT FREQUENCY DOMAIN TRANSFORMATION
# ============================================================================

def compute_stft(signal, n_fft=64, hop_length=16):
    """Compute Short-Time Fourier Transform for a 1D signal.

    Args:
        signal: 1D array of time-domain samples
        n_fft: FFT window size
        hop_length: Number of samples between successive frames

    Returns:
        magnitude: 2D array of shape (n_frames, n_freq_bins)
                   where n_freq_bins = n_fft // 2 + 1
    """
    # Number of frequency bins (positive frequencies only)
    n_freq = n_fft // 2 + 1

    # Number of frames
    n_frames = 1 + (len(signal) - n_fft) // hop_length

    # Hann window for smooth spectral estimation
    window = np.hanning(n_fft)

    # Initialize output
    stft_matrix = np.zeros((n_frames, n_freq), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft_matrix[i] = np.abs(spectrum)

    return stft_matrix


def compute_stft_multichannel(window, n_fft=64, hop_length=16):
    """Compute STFT for multi-channel window data.

    Args:
        window: 2D array of shape (timesteps, n_channels)
        n_fft: FFT window size
        hop_length: Hop between frames

    Returns:
        spectrogram: 3D array of shape (n_frames, n_freq_bins, n_channels)
    """
    n_timesteps, n_channels = window.shape
    n_freq = n_fft // 2 + 1
    n_frames = 1 + (n_timesteps - n_fft) // hop_length

    spectrogram = np.zeros((n_frames, n_freq, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        spectrogram[:, :, ch] = compute_stft(window[:, ch], n_fft, hop_length)

    return spectrogram


def apply_stft_to_windows(X, n_fft=64, hop_length=16):
    """Apply STFT transformation to all windows.

    Args:
        X: Windows array of shape (n_windows, timesteps, n_features)
        n_fft: FFT window size
        hop_length: Hop between FFT frames

    Returns:
        X_stft: Spectrograms of shape (n_windows, n_frames, n_freq_bins, n_features)
    """
    n_windows, n_timesteps, n_features = X.shape
    n_freq = n_fft // 2 + 1
    n_frames = 1 + (n_timesteps - n_fft) // hop_length

    X_stft = np.zeros((n_windows, n_frames, n_freq, n_features), dtype=np.float32)

    for i in range(n_windows):
        X_stft[i] = compute_stft_multichannel(X[i], n_fft, hop_length)

    # Convert to log scale (dB) for better dynamic range
    X_stft = np.log1p(X_stft)  # log(1 + x) to handle zeros

    return X_stft


def normalize_spectrograms(X_stft):
    """Normalize spectrograms per-sample (zero mean, unit variance).

    Args:
        X_stft: Shape (n_windows, n_frames, n_freq, n_channels)

    Returns:
        Normalized spectrograms
    """
    # Normalize across freq and time dimensions for each window and channel
    mean = X_stft.mean(axis=(1, 2), keepdims=True)
    std = X_stft.std(axis=(1, 2), keepdims=True)
    std[std == 0] = 1
    return (X_stft - mean) / std


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_jitter(X, sigma=0.03):
    """Add random Gaussian noise to signals."""
    noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
    return X + noise


def augment_scaling(X, sigma=0.1):
    """Scale signals by random factor per feature."""
    # Random scaling factor per sample and feature
    factors = np.random.normal(1.0, sigma, (X.shape[0], 1, X.shape[2])).astype(np.float32)
    return X * factors


def augment_time_warp(X, sigma=0.2, knots=4):
    """Simple time warping by randomly stretching/compressing segments."""
    # Simplified version: random temporal scaling of segments
    n_samples, timesteps, features = X.shape
    X_aug = np.zeros_like(X)

    for i in range(n_samples):
        # Random warp factor
        warp = 1.0 + np.random.uniform(-sigma, sigma)
        orig_steps = np.arange(timesteps)
        new_steps = np.linspace(0, timesteps - 1, int(timesteps * warp))

        # Interpolate back to original length
        for f in range(features):
            if len(new_steps) > timesteps:
                # Compressed - take subset
                indices = np.linspace(0, len(new_steps) - 1, timesteps).astype(int)
                X_aug[i, :, f] = np.interp(orig_steps, new_steps[indices], X[i, :, f])
            else:
                # Stretched - interpolate
                X_aug[i, :, f] = np.interp(new_steps, orig_steps, X[i, :, f])
                # Pad or truncate to original length
                if len(new_steps) < timesteps:
                    X_aug[i, len(new_steps):, f] = X[i, -1, f]

    return X_aug


def augment_magnitude_warp(X, sigma=0.2, knots=4):
    """Smoothly vary magnitude across time."""
    n_samples, timesteps, features = X.shape

    # Create smooth random curves
    orig_steps = np.arange(timesteps)
    random_warps = np.random.normal(1.0, sigma, (n_samples, knots + 2, features))
    warp_steps = np.linspace(0, timesteps - 1, knots + 2)

    X_aug = np.zeros_like(X)
    for i in range(n_samples):
        for f in range(features):
            warp_curve = np.interp(orig_steps, warp_steps, random_warps[i, :, f])
            X_aug[i, :, f] = X[i, :, f] * warp_curve

    return X_aug


def apply_augmentation(X, y, augmentation_factor=2):
    """Apply augmentation to increase training data.

    Each sample gets augmented multiple times with different transforms.
    """
    augmented_X = [X]  # Start with original data
    augmented_y = [y]

    for _ in range(augmentation_factor):
        # Apply random combination of augmentations
        X_aug = X.copy()

        # 70% chance of jitter
        if np.random.random() > 0.3:
            X_aug = augment_jitter(X_aug, sigma=0.03)

        # 70% chance of scaling
        if np.random.random() > 0.3:
            X_aug = augment_scaling(X_aug, sigma=0.15)

        # 50% chance of magnitude warp
        if np.random.random() > 0.5:
            X_aug = augment_magnitude_warp(X_aug, sigma=0.2)

        augmented_X.append(X_aug)
        augmented_y.append(y)

    return np.vstack(augmented_X), np.hstack(augmented_y)


def extract_features(X):
    """Extract statistical features from windows.

    These handcrafted features can help when deep learning struggles.
    """
    # X shape: (n_windows, timesteps, features)
    features = []

    # Time domain features per channel
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    max_val = X.max(axis=1)
    min_val = X.min(axis=1)
    peak_to_peak = max_val - min_val
    rms = np.sqrt(np.mean(X**2, axis=1))

    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1) / X.shape[1]

    # Skewness and kurtosis
    from scipy.stats import skew, kurtosis
    skewness = skew(X, axis=1)
    kurt = kurtosis(X, axis=1)

    # Stack all features
    all_features = np.hstack([mean, std, peak_to_peak, rms, zero_crossings, skewness, kurt])

    return all_features


def load_dataset(config, binary=True):
    """Load full dataset with windowing, tracking recording groups and temperature."""
    print("Loading dataset...")

    h5_file = h5py.File(config['data_path'], 'r')
    recordings = list(h5_file.keys())

    all_windows = []
    all_labels = []
    all_groups = []  # Track which recording each window came from
    all_temps = []   # Temperature for each window

    # Temperature normalization parameters (from dataset: 9-22°C)
    # Using wider range for deployment flexibility: -10°C to 40°C
    temp_min = config.get('temp_min', -10)
    temp_max = config.get('temp_max', 40)

    for rec_idx, recording in enumerate(recordings):
        label = get_recording_label(recording, binary=binary)

        # Get temperature from recording attributes
        temperature = h5_file[recording].attrs.get('temperature', 15)  # Default 15°C
        # Normalize to [0, 1]
        temp_normalized = (temperature - temp_min) / (temp_max - temp_min)
        temp_normalized = np.clip(temp_normalized, 0, 1)

        # Load strain data
        strain_df = load_strain_data(h5_file, recording)
        data = strain_df.values

        # Create windows
        windows = create_windows(
            data,
            config['window_size'],
            config['window_overlap'],
            config.get('max_windows_per_recording')
        )

        all_windows.append(windows)
        all_labels.extend([label] * len(windows))
        all_groups.extend([rec_idx] * len(windows))
        all_temps.extend([temp_normalized] * len(windows))  # Same temp for all windows in recording

        label_str = "Undamaged" if label == 0 else "Damaged"
        print(f"  {recording}: {len(windows)} windows, label={label_str}, temp={temperature}°C")

    h5_file.close()

    X = np.vstack(all_windows)
    y = np.array(all_labels)
    groups = np.array(all_groups)
    temps = np.array(all_temps, dtype=np.float32)

    print(f"\nTotal: {X.shape[0]} windows, {X.shape[1]} timesteps, {X.shape[2]} features")
    print(f"Labels: Undamaged={np.sum(y==0)}, Damaged={np.sum(y==1)}")
    print(f"Unique recordings: {len(np.unique(groups))}")
    print(f"Temperature range (normalized): {temps.min():.2f} - {temps.max():.2f}")

    return X, y, groups, temps, recordings


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def build_cnn_lstm_stft(input_shape, use_temperature=True, stateful=False, batch_size=None):
    """Build CNN-LSTM model for STFT spectrogram input with optional temperature conditioning.

    Designed for frequency-domain anomaly detection:
    - Input: Spectrograms of shape (n_frames, n_freq_bins, n_channels)
    - Optional: Temperature as conditional input (scalar, normalized 0-1)
    - 2D convolutions over time-frequency representation
    - Unidirectional LSTM for temporal sequence modeling
    - Causal architecture for live streaming

    Args:
        input_shape: (n_frames, n_freq_bins, n_channels) - spectrogram shape
        use_temperature: If True, add temperature as conditional input
        stateful: If True, LSTM maintains state between batches
        batch_size: Required if stateful=True
    """
    from tensorflow.keras.layers import (LayerNormalization, Add, Reshape,
                                          Conv2D, MaxPooling2D, TimeDistributed,
                                          Permute, Flatten, Concatenate)
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    l2_reg = 0.005
    n_frames, n_freq, n_channels = input_shape

    # Spectrogram input
    if stateful:
        spec_input = Input(batch_shape=(batch_size,) + input_shape, name='spectrogram')
    else:
        spec_input = Input(shape=input_shape, name='spectrogram')

    # Temperature input (scalar, normalized 0-1)
    if use_temperature:
        temp_input = Input(shape=(1,), name='temperature')

    # 2D CNN over frequency axis (learning frequency patterns)
    x = spec_input
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)  # Pool only frequency
    x = Dropout(0.2)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)  # Pool frequency again
    x = Dropout(0.2)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Reshape for LSTM: (batch, frames, freq*filters) -> (batch, frames, features)
    reduced_freq = n_freq // 4
    x = Reshape((n_frames, reduced_freq * 64))(x)

    # Layer normalization before LSTM
    x = LayerNormalization()(x)

    # Stacked unidirectional LSTMs
    x = LSTM(64, return_sequences=True,
             dropout=0.3,
             recurrent_dropout=0.2,
             kernel_regularizer=l2(l2_reg),
             stateful=stateful)(x)
    x = LayerNormalization()(x)

    x = LSTM(32, return_sequences=False,
             dropout=0.3,
             recurrent_dropout=0.2,
             kernel_regularizer=l2(l2_reg),
             stateful=stateful)(x)
    x = Dropout(0.4)(x)

    # Concatenate temperature conditioning before dense layers
    if use_temperature:
        # Expand temperature through a small embedding
        temp_embed = Dense(8, activation='relu', name='temp_embed')(temp_input)
        x = Concatenate()([x, temp_embed])

    # Dense layers for anomaly scoring
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.3)(x)

    # Output: anomaly score [0, 1]
    outputs = Dense(1, activation='sigmoid', name='anomaly_score')(x)

    # Build model with appropriate inputs
    if use_temperature:
        model = Model([spec_input, temp_input], outputs, name='CNN_LSTM_STFT_TempCond')
    else:
        model = Model(spec_input, outputs, name='CNN_LSTM_STFT_AnomalyDetector')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    return model


def build_cnn_lstm(input_shape, n_classes=2, stateful=False, batch_size=None):
    """Build CNN-LSTM model for time-domain anomaly detection.

    CAUSAL architecture for live streaming:
    - Unidirectional LSTM (no future lookahead)
    - Causal padding on convolutions with dilations for larger receptive field
    - Stacked LSTMs for better temporal modeling
    - Optional stateful mode for continuous streaming
    - Outputs anomaly score [0, 1]

    Args:
        input_shape: (timesteps, features)
        n_classes: Not used (kept for compatibility), always outputs score
        stateful: If True, LSTM maintains state between batches (for streaming)
        batch_size: Required if stateful=True
    """
    from tensorflow.keras.layers import LayerNormalization, Add, Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    l2_reg = 0.005

    # For stateful LSTM, need to specify batch_input_shape
    if stateful:
        inputs = Input(batch_shape=(batch_size, input_shape[0], input_shape[1]))
    else:
        inputs = Input(shape=input_shape)

    # Initial projection
    x = Conv1D(32, kernel_size=1, activation='relu')(inputs)

    # Dilated causal convolutions (TCN-style) for larger receptive field
    # Dilation pattern: 1, 2, 4 gives receptive field of 1+2+4=7 per block
    for dilation in [1, 2, 4]:
        residual = x
        x = Conv1D(32, kernel_size=3, dilation_rate=dilation,
                   activation='relu', padding='causal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # Residual connection
        if residual.shape[-1] != x.shape[-1]:
            residual = Conv1D(32, kernel_size=1)(residual)
        x = Add()([x, residual])

    # Downsample
    x = MaxPooling1D(pool_size=4)(x)

    # Second dilated block with more filters
    for dilation in [1, 2, 4]:
        residual = x
        x = Conv1D(64, kernel_size=3, dilation_rate=dilation,
                   activation='relu', padding='causal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = Conv1D(64, kernel_size=1)(residual)
        x = Add()([x, residual])

    # Downsample again
    x = MaxPooling1D(pool_size=4)(x)

    # Layer normalization before LSTM
    x = LayerNormalization()(x)

    # Stacked unidirectional LSTMs for better temporal modeling
    x = LSTM(64, return_sequences=True,
             dropout=0.3,
             recurrent_dropout=0.2,
             kernel_regularizer=l2(l2_reg),
             stateful=stateful)(x)
    x = LayerNormalization()(x)

    x = LSTM(32, return_sequences=False,
             dropout=0.3,
             recurrent_dropout=0.2,
             kernel_regularizer=l2(l2_reg),
             stateful=stateful)(x)
    x = Dropout(0.4)(x)

    # Dense layers for anomaly scoring
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.3)(x)

    # Output: anomaly score [0, 1]
    outputs = Dense(1, activation='sigmoid', name='anomaly_score')(x)

    model = Model(inputs, outputs, name='CNN_LSTM_AnomalyDetector')

    # Adam with gradient clipping and slightly higher LR
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(X, y, groups, temps, config):
    """Train model with proper group-based splitting.

    Key fix: Split by RECORDING (group), not by window.
    This prevents data leakage from overlapping windows.

    Supports both time-domain and STFT frequency-domain inputs.
    Optionally includes temperature as conditional input.

    Args:
        X: Window data (n_windows, timesteps, features)
        y: Labels (n_windows,)
        groups: Recording group IDs (n_windows,)
        temps: Normalized temperature values (n_windows,)
        config: Configuration dictionary
    """
    use_stft = config.get('use_stft', False)
    use_temperature = config.get('use_temperature', True)

    print("\n" + "="*50)
    mode_str = "STFT FREQUENCY DOMAIN" if use_stft else "TIME DOMAIN"
    temp_str = " + TEMPERATURE" if use_temperature else ""
    print(f"TRAINING WITH {mode_str}{temp_str}")
    print("="*50)

    n_samples, n_timesteps, n_features = X.shape

    # CRITICAL FIX: Split by recording group, not by individual windows
    gss = GroupShuffleSplit(n_splits=1, test_size=config['val_split'],
                            random_state=config['random_state'])

    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    temps_train, temps_val = temps[train_idx], temps[val_idx]

    # Verify no group overlap
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    assert len(train_groups & val_groups) == 0, "Data leakage detected!"

    print(f"Training recordings: {len(train_groups)}, Validation recordings: {len(val_groups)}")
    print(f"Training samples: {len(X_train_raw)}, Validation samples: {len(X_val_raw)}")
    print(f"Train labels: Undamaged={np.sum(y_train==0)}, Damaged={np.sum(y_train==1)}")
    print(f"Val labels: Undamaged={np.sum(y_val==0)}, Damaged={np.sum(y_val==1)}")
    if use_temperature:
        print(f"Train temp range: {temps_train.min():.2f} - {temps_train.max():.2f}")

    # STFT TRANSFORMATION
    if use_stft:
        n_fft = config.get('n_fft', 64)
        hop_length = config.get('hop_length', 16)

        print(f"\nApplying STFT (n_fft={n_fft}, hop_length={hop_length})...")

        # First normalize time-domain signals
        X_train_norm = normalize_windows_per_window(X_train_raw)
        X_val_norm = normalize_windows_per_window(X_val_raw)

        # Apply STFT transformation
        X_train = apply_stft_to_windows(X_train_norm, n_fft=n_fft, hop_length=hop_length)
        X_val = apply_stft_to_windows(X_val_norm, n_fft=n_fft, hop_length=hop_length)

        # Normalize spectrograms
        X_train = normalize_spectrograms(X_train)
        X_val = normalize_spectrograms(X_val)

        print(f"  Spectrogram shape: {X_train.shape[1:]} (frames, freq_bins, channels)")

        # Build STFT model
        input_shape = X_train.shape[1:]
        model = build_cnn_lstm_stft(input_shape=input_shape, use_temperature=use_temperature)
    else:
        # TIME DOMAIN processing
        if config.get('per_window_norm', True):
            print("\nUsing per-window normalization...")
            X_train = normalize_windows_per_window(X_train_raw)
            X_val = normalize_windows_per_window(X_val_raw)
        else:
            print("\nUsing global normalization (fit on train only)...")
            X_train_flat = X_train_raw.reshape(-1, n_features)
            scaler = StandardScaler()
            scaler.fit(X_train_flat)

            X_train = scaler.transform(X_train_flat).reshape(-1, n_timesteps, n_features)
            X_val_flat = X_val_raw.reshape(-1, n_features)
            X_val = scaler.transform(X_val_flat).reshape(-1, n_timesteps, n_features)

        # DATA AUGMENTATION - apply only to training data (time-domain only)
        aug_factor = config.get('augmentation_factor', 2)
        if aug_factor > 0:
            print(f"\nApplying data augmentation (factor={aug_factor})...")
            original_size = len(X_train)
            X_train, y_train = apply_augmentation(X_train, y_train, augmentation_factor=aug_factor)
            # Repeat temperatures for augmented samples
            temps_train = np.tile(temps_train, aug_factor + 1)
            print(f"  Training data expanded: {original_size} -> {len(X_train)} samples")

        # Build time-domain model (no temperature support yet for time-domain)
        model = build_cnn_lstm(input_shape=(n_timesteps, n_features))

    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )

    # Compute class weights for imbalance
    n_damaged = np.sum(y_train == 1)
    n_undamaged = np.sum(y_train == 0)
    total = len(y_train)
    class_weights = {
        0: total / (2 * n_undamaged),
        1: total / (2 * n_damaged)
    }
    print(f"\nClass weights: {class_weights}")

    print(f"\nTraining for up to {config['epochs']} epochs...")

    # Prepare training data based on model inputs
    if use_stft and use_temperature:
        train_inputs = [X_train, temps_train.reshape(-1, 1)]
        val_inputs = [X_val, temps_val.reshape(-1, 1)]
    else:
        train_inputs = X_train
        val_inputs = X_val

    # Train
    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        shuffle=True,
        verbose=1
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_inputs, y_val, verbose=0)

    # Additional metrics
    y_pred = (model.predict(val_inputs, verbose=0) > 0.5).astype(int).flatten()
    tp = np.sum((y_pred == 1) & (y_val == 1))
    tn = np.sum((y_pred == 0) & (y_val == 0))
    fp = np.sum((y_pred == 1) & (y_val == 0))
    fn = np.sum((y_pred == 0) & (y_val == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'accuracy': val_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

    return model, history.history, val_acc, val_loss, metrics


def print_results(val_acc, val_loss, epochs_trained, metrics):
    """Print training results."""
    print("\n" + "="*50)
    print("TRAINING RESULTS (Binary: Damaged vs Undamaged)")
    print("="*50)
    print(f"Epochs trained: {epochs_trained}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("-"*50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("-"*50)
    print("Confusion Matrix:")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}")
    print(f"  FN={metrics['fn']}, TN={metrics['tn']}")
    print("="*50)


def plot_learning_curves(history, output_path):
    """Plot learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history['loss']) + 1)

    # Plot 1: Loss
    ax1 = axes[0]
    ax1.plot(epochs, history['loss'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_title('Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history['accuracy'], 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    ax2.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('CNN-LSTM Binary Classification (Group-Split)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nLearning curves saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main(quick_test=False, medium_test=False, simulate_live=False):
    print("CNN-LSTM Anomaly Detector for Bridge Monitoring")
    print("(Causal architecture for live streaming)")
    print("="*60)

    config = CONFIG.copy()

    if quick_test:
        print("\n*** QUICK TEST MODE ***\n")
        config['max_windows_per_recording'] = 10
        config['epochs'] = 10
    elif medium_test:
        print("\n*** MEDIUM TEST MODE ***\n")
        config['max_windows_per_recording'] = 40
        config['epochs'] = 100

    # Ensure output directory exists
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])

    # Load data (binary: normal=0, anomaly=1)
    X, y, groups, temps, recordings = load_dataset(config, binary=True)

    # Train model with proper group-based splitting
    model, history, val_acc, val_loss, metrics = train_model(X, y, groups, temps, config)

    # Print results
    print_results(val_acc, val_loss, len(history['loss']), metrics)

    # Plot learning curves
    plot_path = config['output_dir'] / 'learning_curves.png'
    plot_learning_curves(history, plot_path)

    # Save model for deployment
    model_path = config['output_dir'] / 'anomaly_detector.keras'
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Optionally simulate live detection
    if simulate_live:
        # Get validation data for simulation
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=config['val_split'],
                                random_state=config['random_state'])
        _, val_idx = next(gss.split(X, y, groups))
        X_val = normalize_windows_per_window(X[val_idx])
        y_val = y[val_idx]
        temps_val = temps[val_idx]

        simulate_live_detection(model, X_val, y_val, temps_val, config)

    return model, history, val_acc, metrics


# ============================================================================
# LIVE ANOMALY DETECTOR
# ============================================================================

class LiveAnomalyDetector:
    """Real-time anomaly detector for streaming sensor data.

    Designed for live bridge monitoring:
    - Sliding window approach for continuous monitoring
    - Supports both time-domain and STFT frequency-domain processing
    - Optional temperature conditioning for environmental compensation
    - Per-window normalization (no need to store historical statistics)
    - Configurable anomaly threshold
    - Exponential moving average for smoothing predictions

    Usage:
        # STFT model with temperature
        detector = LiveAnomalyDetector(
            model_path='stft_model.keras',
            use_stft=True,
            use_temperature=True,
            n_fft=64,
            hop_length=16
        )

        # Set current temperature (call whenever temperature changes)
        detector.set_temperature(15.0)  # 15°C

        # In your data acquisition loop:
        while True:
            new_samples = read_sensors()  # shape: (n_samples, n_features)
            results = detector.process(new_samples)
            if results:
                for result in results:
                    if result['is_anomaly']:
                        trigger_alert(result['anomaly_score'])
    """

    def __init__(self, model_path=None, model=None, window_size=512,
                 anomaly_threshold=0.5, ema_alpha=0.3,
                 use_stft=False, n_fft=64, hop_length=16,
                 use_temperature=False, temp_min=-10, temp_max=40):
        """Initialize the live detector.

        Args:
            model_path: Path to saved .keras model file
            model: Pre-loaded model (alternative to model_path)
            window_size: Number of samples per window (default 512 = ~5s at 100Hz)
            anomaly_threshold: Score above this triggers anomaly (default 0.5)
            ema_alpha: Exponential moving average smoothing (0=no smoothing, 1=no memory)
            use_stft: If True, convert to frequency domain using STFT
            n_fft: FFT window size for STFT
            hop_length: Hop between STFT frames
            use_temperature: If True, model expects temperature input
            temp_min: Minimum temperature for normalization (default -10°C)
            temp_max: Maximum temperature for normalization (default 40°C)
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.ema_alpha = ema_alpha

        # STFT parameters
        self.use_stft = use_stft
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Temperature parameters
        self.use_temperature = use_temperature
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.current_temp_normalized = 0.5  # Default: middle of range (15°C)

        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Must provide either model_path or model")

        # Get expected number of input features from model
        # For multi-input models (with temperature), input_shape is a list
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            # Multi-input model: [spectrogram_shape, temperature_shape]
            spec_shape = input_shape[0]
            self.n_features = spec_shape[-1]
        elif use_stft:
            self.n_features = input_shape[-1]
        else:
            self.n_features = input_shape[-1]

        # Buffer for accumulating samples
        self.buffer = None
        self.ema_score = None  # Smoothed anomaly score

        # Statistics
        self.windows_processed = 0
        self.anomalies_detected = 0

    def set_temperature(self, temperature_celsius):
        """Set current temperature for conditioning.

        Args:
            temperature_celsius: Current temperature in Celsius
        """
        normalized = (temperature_celsius - self.temp_min) / (self.temp_max - self.temp_min)
        self.current_temp_normalized = np.clip(normalized, 0, 1)

    def reset(self):
        """Reset the detector state (buffer and EMA)."""
        self.buffer = None
        self.ema_score = None
        self.windows_processed = 0
        self.anomalies_detected = 0

    def _normalize_window(self, window):
        """Per-window normalization (zero mean, unit variance per feature)."""
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True)
        std[std == 0] = 1
        return (window - mean) / std

    def _apply_stft(self, window):
        """Apply STFT transformation to a single window.

        Args:
            window: Shape (timesteps, n_features)

        Returns:
            spectrogram: Shape (n_frames, n_freq, n_features)
        """
        spectrogram = compute_stft_multichannel(window, self.n_fft, self.hop_length)
        # Log scale
        spectrogram = np.log1p(spectrogram)
        # Normalize
        mean = spectrogram.mean(axis=(0, 1), keepdims=True)
        std = spectrogram.std(axis=(0, 1), keepdims=True)
        std[std == 0] = 1
        return (spectrogram - mean) / std

    def process(self, new_samples):
        """Process new sensor samples and return anomaly results.

        Args:
            new_samples: numpy array of shape (n_samples, n_features)
                        or (n_samples,) if single feature

        Returns:
            List of result dicts, one per complete window processed:
            [{'anomaly_score': float, 'is_anomaly': bool, 'smoothed_score': float}, ...]
            Returns empty list if no complete windows ready yet.
        """
        # Ensure 2D
        if new_samples.ndim == 1:
            new_samples = new_samples.reshape(-1, 1)

        # Validate features
        if new_samples.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {new_samples.shape[1]}")

        # Add to buffer
        if self.buffer is None:
            self.buffer = new_samples
        else:
            self.buffer = np.vstack([self.buffer, new_samples])

        results = []

        # Process complete windows (non-overlapping for live streaming)
        while len(self.buffer) >= self.window_size:
            # Extract window
            window = self.buffer[:self.window_size]
            self.buffer = self.buffer[self.window_size:]  # Remove processed samples

            # Normalize time-domain signal
            window_norm = self._normalize_window(window)

            # Transform to appropriate domain
            if self.use_stft:
                processed = self._apply_stft(window_norm)
            else:
                processed = window_norm

            # Prepare model input
            spec_batch = processed[np.newaxis, ...]

            if self.use_temperature:
                temp_batch = np.array([[self.current_temp_normalized]], dtype=np.float32)
                model_input = [spec_batch, temp_batch]
            else:
                model_input = spec_batch

            # Predict
            raw_score = float(self.model.predict(model_input, verbose=0)[0, 0])

            # Apply EMA smoothing
            if self.ema_score is None:
                self.ema_score = raw_score
            else:
                self.ema_score = self.ema_alpha * raw_score + (1 - self.ema_alpha) * self.ema_score

            # Check threshold
            is_anomaly = self.ema_score >= self.anomaly_threshold

            self.windows_processed += 1
            if is_anomaly:
                self.anomalies_detected += 1

            results.append({
                'anomaly_score': raw_score,
                'smoothed_score': self.ema_score,
                'is_anomaly': is_anomaly,
                'window_id': self.windows_processed,
                'temperature': self.current_temp_normalized * (self.temp_max - self.temp_min) + self.temp_min
            })

        return results

    def get_stats(self):
        """Get detector statistics."""
        return {
            'windows_processed': self.windows_processed,
            'anomalies_detected': self.anomalies_detected,
            'anomaly_rate': self.anomalies_detected / max(1, self.windows_processed),
            'buffer_size': len(self.buffer) if self.buffer is not None else 0,
            'current_smoothed_score': self.ema_score,
            'use_stft': self.use_stft,
            'use_temperature': self.use_temperature,
            'current_temperature': self.current_temp_normalized * (self.temp_max - self.temp_min) + self.temp_min
        }


def simulate_live_detection(model, X_test, y_test, temps_test, config):
    """Simulate live detection on test data.

    This demonstrates how the detector would work on streaming data.
    Supports both time-domain and STFT models with temperature conditioning.
    """
    print("\n" + "="*50)
    print("SIMULATING LIVE ANOMALY DETECTION")
    use_stft = config.get('use_stft', False)
    use_temperature = config.get('use_temperature', False)
    if use_stft:
        print("(Using STFT frequency domain)")
    if use_temperature:
        print("(With temperature conditioning)")
    print("="*50)

    detector = LiveAnomalyDetector(
        model=model,
        window_size=config['window_size'],
        anomaly_threshold=0.5,
        ema_alpha=0.3,
        use_stft=use_stft,
        n_fft=config.get('n_fft', 64),
        hop_length=config.get('hop_length', 16),
        use_temperature=use_temperature,
        temp_min=config.get('temp_min', -10),
        temp_max=config.get('temp_max', 40)
    )

    # Flatten test windows back to continuous stream (simulating live data)
    all_predictions = []
    all_labels = []

    for i, (window, label, temp_norm) in enumerate(zip(X_test, y_test, temps_test)):
        # Convert normalized temp back to Celsius for the detector
        temp_celsius = temp_norm * (config.get('temp_max', 40) - config.get('temp_min', -10)) + config.get('temp_min', -10)
        detector.set_temperature(temp_celsius)

        # Simulate receiving samples in chunks (e.g., 50 samples at a time)
        chunk_size = 50
        for start in range(0, len(window), chunk_size):
            chunk = window[start:start + chunk_size]
            results = detector.process(chunk)

            for result in results:
                all_predictions.append(result['is_anomaly'])
                all_labels.append(label)

                if result['is_anomaly']:
                    print(f"  [!] ANOMALY at window {result['window_id']}: "
                          f"score={result['anomaly_score']:.3f}, "
                          f"smoothed={result['smoothed_score']:.3f}, "
                          f"temp={result['temperature']:.1f}°C")

    # Calculate metrics
    if all_predictions:
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        accuracy = np.mean(predictions == labels)
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nLive Detection Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")

    stats = detector.get_stats()
    print(f"\nDetector Stats:")
    print(f"  Windows processed: {stats['windows_processed']}")
    print(f"  Anomalies detected: {stats['anomalies_detected']}")
    print(f"  Anomaly rate: {stats['anomaly_rate']:.2%}")

    return detector


if __name__ == '__main__':
    import sys
    quick_test = '--quick' in sys.argv or '-q' in sys.argv
    medium_test = '--medium' in sys.argv or '-m' in sys.argv
    simulate_live = '--live' in sys.argv or '-l' in sys.argv
    main(quick_test=quick_test, medium_test=medium_test, simulate_live=simulate_live)
