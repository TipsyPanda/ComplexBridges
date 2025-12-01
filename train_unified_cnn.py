"""
Unified 1D CNN for Multi-Sensor Anomaly Detection
================================================
This script trains a 1D CNN that predicts anomalies across all sensor types
(strain gauges, accelerometers, temperature sensors) in the bridge monitoring system.

Architecture:
- Takes time-series windows from all sensor types
- Uses sensor-type-aware normalization
- 1D CNN layers for temporal feature extraction
- Binary classification output (anomaly vs normal)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_PATH = "data/ipmb_5sensors_30min_1_to_10hz.csv"
MODEL_DIR = "artifacts/unified_cnn"
WINDOW_SIZE = 50  # Number of timesteps in each sequence
STRIDE = 10       # Stride for sliding window (overlap)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# Create output directory
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*70)
print("UNIFIED 1D CNN FOR MULTI-SENSOR ANOMALY DETECTION")
print("="*70)


def load_and_preprocess_data(data_path):
    """Load CSV data and perform initial preprocessing"""
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Total rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    # Convert timestamp to datetime (handle ISO format with microseconds)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Print sensor type distribution
    print("\nSensor type distribution:")
    print(df['sensor_type'].value_counts())

    # Print anomaly distribution
    print("\nAnomaly distribution:")
    print(f"Normal (0): {(df['anomaly'] == 0).sum():,} ({(df['anomaly'] == 0).sum() / len(df) * 100:.2f}%)")
    print(f"Anomaly (1): {(df['anomaly'] == 1).sum():,} ({(df['anomaly'] == 1).sum() / len(df) * 100:.2f}%)")

    return df


def create_sensor_pivot(df):
    """
    Create a pivot table where each row is a timestamp and columns are sensor readings
    This creates a unified representation across all sensors
    """
    print("\nCreating sensor pivot table...")

    # Create a unique identifier for each sensor reading
    df['sensor_key'] = df['sensor_type'] + '_' + df['sensor_id']

    # Pivot: rows=timestamp, columns=sensor_key, values=value
    pivot = df.pivot_table(
        index='timestamp',
        columns='sensor_key',
        values='value',
        aggfunc='first'  # In case of duplicates, take first
    )

    # Forward fill then backward fill missing values
    pivot = pivot.ffill().bfill()

    # Get corresponding anomaly labels for each timestamp
    # We'll consider a timestamp anomalous if ANY sensor at that time is anomalous
    anomaly_pivot = df.pivot_table(
        index='timestamp',
        columns='sensor_key',
        values='anomaly',
        aggfunc='max'  # Max will be 1 if any sensor is anomalous
    )

    # Create overall anomaly label: 1 if any sensor shows anomaly
    labels = (anomaly_pivot.max(axis=1) > 0).astype(int)

    print(f"Pivot table shape: {pivot.shape}")
    print(f"Sensors (columns): {list(pivot.columns)}")
    print(f"Labels shape: {labels.shape}")

    return pivot, labels


def normalize_by_sensor_type(pivot_df):
    """
    Normalize each sensor type separately since they have different scales
    - strain_gauge: ~100-200 microstrain
    - accelerometer_rms: ~0.01-0.03 g
    - temperature: ~20-30 C
    """
    print("\nNormalizing sensor data by type...")

    normalized = pivot_df.copy()
    scalers = {}

    for col in pivot_df.columns:
        sensor_type = col.split('_')[0] + '_' + col.split('_')[1]  # e.g., "strain_gauge"

        if sensor_type not in scalers:
            scalers[sensor_type] = StandardScaler()

        # Fit and transform
        if col not in [k for k, v in scalers.items()]:
            scaler = StandardScaler()
            normalized[col] = scaler.fit_transform(pivot_df[[col]])
            scalers[col] = scaler

    print(f"Normalized {len(pivot_df.columns)} sensor channels")

    return normalized, scalers


def create_sequences(data, labels, window_size, stride):
    """
    Create sliding window sequences for CNN input

    Args:
        data: DataFrame with sensor readings (rows=time, cols=sensors)
        labels: Series with anomaly labels
        window_size: Number of timesteps per sequence
        stride: Step size for sliding window

    Returns:
        X: numpy array of shape (n_sequences, window_size, n_sensors)
        y: numpy array of shape (n_sequences,)
    """
    print(f"\nCreating sequences with window_size={window_size}, stride={stride}...")

    data_array = data.values
    labels_array = labels.values

    X_sequences = []
    y_sequences = []

    for i in range(0, len(data_array) - window_size, stride):
        # Extract window
        window = data_array[i:i+window_size]

        # Label: 1 if ANY timestep in window is anomalous
        window_labels = labels_array[i:i+window_size]
        label = 1 if np.any(window_labels == 1) else 0

        X_sequences.append(window)
        y_sequences.append(label)

    X = np.array(X_sequences)
    y = np.array(y_sequences)

    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape} (n_sequences, window_size, n_sensors)")
    print(f"y shape: {y.shape}")
    print(f"Anomaly ratio in sequences: {y.sum() / len(y) * 100:.2f}%")

    return X, y


def build_cnn_model(input_shape, n_filters=[64, 128, 256], kernel_sizes=[5, 3, 3], dropout_rate=0.3):
    """
    Build 1D CNN architecture for multi-sensor anomaly detection

    Args:
        input_shape: (window_size, n_sensors)
        n_filters: List of filter numbers for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        dropout_rate: Dropout rate for regularization

    Returns:
        Keras model
    """
    print("\nBuilding 1D CNN model...")
    print(f"Input shape: {input_shape}")

    model = models.Sequential(name='MultiSensor_1D_CNN')

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Conv Block 1
    model.add(layers.Conv1D(n_filters[0], kernel_sizes[0], padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout_rate))

    # Conv Block 2
    model.add(layers.Conv1D(n_filters[1], kernel_sizes[1], padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout_rate))

    # Conv Block 3
    model.add(layers.Conv1D(n_filters[2], kernel_sizes[2], padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalMaxPooling1D())  # Global pooling instead of flattening
    model.add(layers.Dropout(dropout_rate))

    # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout_rate))

    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    return model


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test, save_dir):
    """Evaluate model and save results"""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    except:
        roc_auc = None
        print("\nROC-AUC Score: N/A (not enough classes)")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, os.path.join(save_dir, 'confusion_matrix.png'))

    # Save metrics
    metrics = {
        'classification_report': classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'], output_dict=True),
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc) if roc_auc is not None else None
    }

    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    """Main training pipeline"""

    # Load data
    df = load_and_preprocess_data(DATA_PATH)

    # Create pivot table
    pivot, labels = create_sensor_pivot(df)

    # Normalize
    normalized, scalers = normalize_by_sensor_type(pivot)

    # Create sequences
    X, y = create_sequences(normalized, labels, WINDOW_SIZE, STRIDE)

    # Check class distribution
    print(f"\nClass distribution in sequences:")
    print(f"Normal (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
    print(f"Anomaly (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")

    # Split data
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set:   {X_val.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Build model
    input_shape = (WINDOW_SIZE, X.shape[2])  # (window_size, n_sensors)
    model = build_cnn_model(input_shape)

    # Compile model
    # Use class weights to handle imbalance
    n_normal = (y_train == 0).sum()
    n_anomaly = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: n_normal / n_anomaly if n_anomaly > 0 else 1.0}

    print(f"\nClass weights: {class_weight}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plot_training_history(history, os.path.join(MODEL_DIR, 'training_history.png'))

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, MODEL_DIR)

    # Save model and artifacts
    print("\n" + "="*70)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*70)

    model.save(os.path.join(MODEL_DIR, 'unified_cnn_model.keras'))
    print(f"Model saved to {os.path.join(MODEL_DIR, 'unified_cnn_model.keras')}")

    # Save configuration
    config = {
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'n_sensors': X.shape[2],
        'sensor_columns': list(pivot.columns),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_input_shape': input_shape,
        'total_parameters': model.count_params(),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test)
    }

    with open(os.path.join(MODEL_DIR, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {os.path.join(MODEL_DIR, 'model_config.json')}")

    # Save scalers
    import joblib
    joblib.dump(scalers, os.path.join(MODEL_DIR, 'scalers.pkl'))
    print(f"Scalers saved to {os.path.join(MODEL_DIR, 'scalers.pkl')}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nAll artifacts saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
