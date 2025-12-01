"""
Predictive 1D CNN for Future Anomaly Forecasting
=================================================
This script trains a 1D CNN that PREDICTS future anomalies with a risk score,
providing early warning before anomalies occur.

Key Differences from Detection Model:
- Labels based on FUTURE anomalies (look-ahead window)
- Outputs risk score: probability of anomaly in next N timesteps
- Focuses on early warning capability
- Evaluates prediction lead time and precision at different thresholds

Prediction Horizons:
- Short-term: 5 minutes ahead
- Medium-term: 10 minutes ahead
- Long-term: 15 minutes ahead
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_PATH = "data/ipmb_5sensors_30min_1_to_10hz.csv"
MODEL_DIR = "artifacts/predictive_cnn"
WINDOW_SIZE = 50  # Historical window to observe
PREDICTION_HORIZON = 50  # Look ahead N timesteps (~5 minutes at 10Hz sampling)
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# Create output directory
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*70)
print("PREDICTIVE 1D CNN FOR FUTURE ANOMALY FORECASTING")
print("="*70)
print(f"Historical window: {WINDOW_SIZE} timesteps")
print(f"Prediction horizon: {PREDICTION_HORIZON} timesteps (~{PREDICTION_HORIZON/10:.1f} minutes)")
print(f"Goal: Predict anomalies {PREDICTION_HORIZON/10:.1f} minutes in advance")


def load_and_preprocess_data(data_path):
    """Load CSV data and perform initial preprocessing"""
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Total rows: {len(df):,}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print("\nSensor type distribution:")
    print(df['sensor_type'].value_counts())

    print("\nAnomaly distribution:")
    print(f"Normal (0): {(df['anomaly'] == 0).sum():,} ({(df['anomaly'] == 0).sum() / len(df) * 100:.2f}%)")
    print(f"Anomaly (1): {(df['anomaly'] == 1).sum():,} ({(df['anomaly'] == 1).sum() / len(df) * 100:.2f}%)")

    return df


def create_sensor_pivot(df):
    """Create pivot table with sensor readings"""
    print("\nCreating sensor pivot table...")

    df['sensor_key'] = df['sensor_type'] + '_' + df['sensor_id']

    pivot = df.pivot_table(
        index='timestamp',
        columns='sensor_key',
        values='value',
        aggfunc='first'
    )

    pivot = pivot.ffill().bfill()

    # Get anomaly labels
    anomaly_pivot = df.pivot_table(
        index='timestamp',
        columns='sensor_key',
        values='anomaly',
        aggfunc='max'
    )

    labels = (anomaly_pivot.max(axis=1) > 0).astype(int)

    print(f"Pivot table shape: {pivot.shape}")
    print(f"Sensors: {list(pivot.columns)}")

    return pivot, labels


def normalize_by_sensor_type(pivot_df):
    """Normalize each sensor channel separately"""
    print("\nNormalizing sensor data...")

    normalized = pivot_df.copy()
    scalers = {}

    for col in pivot_df.columns:
        scaler = StandardScaler()
        normalized[col] = scaler.fit_transform(pivot_df[[col]])
        scalers[col] = scaler

    print(f"Normalized {len(pivot_df.columns)} sensor channels")

    return normalized, scalers


def create_predictive_sequences(data, labels, window_size, prediction_horizon, stride):
    """
    Create sequences for PREDICTIVE modeling

    Key difference: Label is based on FUTURE anomalies, not current window

    Args:
        data: Sensor data
        labels: Anomaly labels
        window_size: Historical window to observe
        prediction_horizon: How far ahead to predict (in timesteps)
        stride: Sliding window stride

    Returns:
        X: Historical sequences (input)
        y: Future anomaly labels (target)
        lead_times: Actual lead time for each prediction
    """
    print(f"\n{'='*70}")
    print("CREATING PREDICTIVE SEQUENCES")
    print(f"{'='*70}")
    print(f"Historical window: {window_size} timesteps")
    print(f"Prediction horizon: {prediction_horizon} timesteps")
    print(f"Stride: {stride}")
    print(f"\nLabel strategy: Look at FUTURE {prediction_horizon} timesteps after each window")

    data_array = data.values
    labels_array = labels.values

    X_sequences = []
    y_sequences = []
    lead_times = []  # Track how far in advance we're predicting

    # Need enough data for: current window + prediction horizon
    max_idx = len(data_array) - window_size - prediction_horizon

    for i in range(0, max_idx, stride):
        # Historical window (what we observe)
        window = data_array[i:i+window_size]

        # Future window (what we want to predict)
        future_start = i + window_size
        future_end = future_start + prediction_horizon
        future_labels = labels_array[future_start:future_end]

        # Label: Will there be an anomaly in the FUTURE window?
        future_anomaly = 1 if np.any(future_labels == 1) else 0

        # Calculate actual lead time (time until first anomaly)
        if future_anomaly == 1:
            first_anomaly_idx = np.where(future_labels == 1)[0][0]
            lead_time = first_anomaly_idx  # timesteps until anomaly
        else:
            lead_time = prediction_horizon  # No anomaly in horizon

        X_sequences.append(window)
        y_sequences.append(future_anomaly)
        lead_times.append(lead_time)

    X = np.array(X_sequences)
    y = np.array(y_sequences)
    lead_times = np.array(lead_times)

    print(f"\nCreated {len(X)} predictive sequences")
    print(f"X shape: {X.shape} (n_sequences, window_size, n_sensors)")
    print(f"y shape: {y.shape}")
    print(f"\n{'='*70}")
    print("PREDICTIVE LABEL DISTRIBUTION")
    print(f"{'='*70}")
    print(f"No future anomaly (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
    print(f"Future anomaly (1):   {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")

    # Analyze lead times for predictions
    if (y == 1).sum() > 0:
        print(f"\n{'='*70}")
        print("EARLY WARNING STATISTICS")
        print(f"{'='*70}")
        anomaly_lead_times = lead_times[y == 1]
        print(f"Average lead time: {anomaly_lead_times.mean():.1f} timesteps ({anomaly_lead_times.mean()/10:.2f} min)")
        print(f"Median lead time:  {np.median(anomaly_lead_times):.1f} timesteps ({np.median(anomaly_lead_times)/10:.2f} min)")
        print(f"Min lead time:     {anomaly_lead_times.min()} timesteps ({anomaly_lead_times.min()/10:.2f} min)")
        print(f"Max lead time:     {anomaly_lead_times.max()} timesteps ({anomaly_lead_times.max()/10:.2f} min)")

    return X, y, lead_times


def build_predictive_cnn(input_shape, n_filters=[64, 128, 256], kernel_sizes=[5, 3, 3], dropout_rate=0.3):
    """
    Build 1D CNN for PREDICTIVE anomaly detection

    Architecture is similar to detection model, but trained with future labels
    """
    print("\nBuilding Predictive 1D CNN model...")
    print(f"Input shape: {input_shape}")

    model = models.Sequential(name='Predictive_1D_CNN')

    model.add(layers.Input(shape=input_shape))

    # Conv Block 1 - Extract low-level temporal patterns
    model.add(layers.Conv1D(n_filters[0], kernel_sizes[0], padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout_rate))

    # Conv Block 2 - Extract mid-level patterns
    model.add(layers.Conv1D(n_filters[1], kernel_sizes[1], padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout_rate))

    # Conv Block 3 - Extract high-level patterns
    model.add(layers.Conv1D(n_filters[2], kernel_sizes[2], padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(dropout_rate))

    # Dense layers for prediction
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout_rate))

    # Output: Risk score (probability of future anomaly)
    model.add(layers.Dense(1, activation='sigmoid', name='risk_score'))

    model.summary()

    return model


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def evaluate_predictive_model(model, X_test, y_test, lead_times_test, save_dir):
    """
    Evaluate predictive model with focus on early warning capability
    """
    print("\n" + "="*70)
    print("PREDICTIVE MODEL EVALUATION")
    print("="*70)

    # Get risk scores
    risk_scores = model.predict(X_test, verbose=0).flatten()

    # Evaluate at different risk thresholds
    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]

    print("\n" + "="*70)
    print("PERFORMANCE AT DIFFERENT RISK THRESHOLDS")
    print("="*70)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Alerts':<10}")
    print("-"*70)

    results = []
    for thresh in thresholds:
        y_pred = (risk_scores > thresh).astype(int)

        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        n_alerts = y_pred.sum()

        print(f"{thresh:<12.1f} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {n_alerts:<10}")

        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_alerts': int(n_alerts)
        })

    # Default threshold 0.5
    y_pred = (risk_scores > 0.5).astype(int)

    print("\n" + "="*70)
    print("DETAILED METRICS (Threshold = 0.5)")
    print("="*70)
    print(classification_report(y_test, y_pred, target_names=['No Future Anomaly', 'Future Anomaly']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, risk_scores)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    except:
        roc_auc = None

    # Early warning analysis
    print("\n" + "="*70)
    print("EARLY WARNING CAPABILITY")
    print("="*70)

    # For correctly predicted anomalies, analyze lead time
    correct_predictions = (y_pred == 1) & (y_test == 1)
    if correct_predictions.sum() > 0:
        warning_lead_times = lead_times_test[correct_predictions]
        print(f"Successfully predicted anomalies: {correct_predictions.sum()}")
        print(f"Average warning time: {warning_lead_times.mean():.1f} timesteps ({warning_lead_times.mean()/10:.2f} min)")
        print(f"Median warning time:  {np.median(warning_lead_times):.1f} timesteps ({np.median(warning_lead_times)/10:.2f} min)")
        print(f"Min warning time:     {warning_lead_times.min()} timesteps ({warning_lead_times.min()/10:.2f} min)")
        print(f"Max warning time:     {warning_lead_times.max()} timesteps ({warning_lead_times.max()/10:.2f} min)")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Future Anomaly', 'Future Anomaly'],
                yticklabels=['No Future Anomaly', 'Future Anomaly'],
                ax=axes[0, 0])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title('Confusion Matrix (Threshold = 0.5)')

    # Risk score distribution
    axes[0, 1].hist(risk_scores[y_test == 0], bins=50, alpha=0.5, label='No Future Anomaly', color='green')
    axes[0, 1].hist(risk_scores[y_test == 1], bins=50, alpha=0.5, label='Future Anomaly', color='red')
    axes[0, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Risk Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Risk Score Distribution')
    axes[0, 1].legend()

    # Precision-Recall at different thresholds
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    axes[1, 0].plot(thresholds, precisions, marker='o', label='Precision')
    axes[1, 0].plot(thresholds, recalls, marker='s', label='Recall')
    axes[1, 0].set_xlabel('Risk Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Lead time distribution for correct predictions
    if correct_predictions.sum() > 0:
        axes[1, 1].hist(warning_lead_times, bins=20, color='orange', edgecolor='black')
        axes[1, 1].axvline(x=warning_lead_times.mean(), color='red', linestyle='--',
                           label=f'Mean: {warning_lead_times.mean():.1f} steps')
        axes[1, 1].set_xlabel('Warning Lead Time (timesteps)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Early Warning Lead Time Distribution')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictive_evaluation.png'), dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plots saved to {os.path.join(save_dir, 'predictive_evaluation.png')}")
    plt.close()

    # Save metrics
    metrics = {
        'threshold_analysis': results,
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred,
                                                       target_names=['No Future Anomaly', 'Future Anomaly'],
                                                       output_dict=True)
    }

    if correct_predictions.sum() > 0:
        metrics['early_warning'] = {
            'avg_lead_time_steps': float(warning_lead_times.mean()),
            'avg_lead_time_minutes': float(warning_lead_times.mean() / 10),
            'median_lead_time_steps': float(np.median(warning_lead_times)),
            'min_lead_time_steps': int(warning_lead_times.min()),
            'max_lead_time_steps': int(warning_lead_times.max())
        }

    with open(os.path.join(save_dir, 'predictive_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    """Main training pipeline"""

    # Load data
    df = load_and_preprocess_data(DATA_PATH)

    # Create pivot
    pivot, labels = create_sensor_pivot(df)

    # Normalize
    normalized, scalers = normalize_by_sensor_type(pivot)

    # Create PREDICTIVE sequences (key difference!)
    X, y, lead_times = create_predictive_sequences(
        normalized, labels, WINDOW_SIZE, PREDICTION_HORIZON, STRIDE
    )

    # Split data
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    X_train, X_temp, y_train, y_temp, lt_train, lt_temp = train_test_split(
        X, y, lead_times, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, lt_val, lt_test = train_test_split(
        X_temp, y_temp, lt_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set:   {X_val.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Build model
    input_shape = (WINDOW_SIZE, X.shape[2])
    model = build_predictive_cnn(input_shape)

    # Class weights
    n_normal = (y_train == 0).sum()
    n_anomaly = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: n_normal / n_anomaly if n_anomaly > 0 else 1.0}
    print(f"\nClass weights: {class_weight}")

    # Compile
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
            os.path.join(MODEL_DIR, 'best_predictive_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    print("\n" + "="*70)
    print("TRAINING PREDICTIVE MODEL")
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

    # Plot history
    plot_training_history(history, os.path.join(MODEL_DIR, 'training_history.png'))

    # Evaluate
    metrics = evaluate_predictive_model(model, X_test, y_test, lt_test, MODEL_DIR)

    # Save everything
    print("\n" + "="*70)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*70)

    model.save(os.path.join(MODEL_DIR, 'predictive_cnn_model.keras'))
    print(f"Model saved to {os.path.join(MODEL_DIR, 'predictive_cnn_model.keras')}")

    config = {
        'window_size': WINDOW_SIZE,
        'prediction_horizon': PREDICTION_HORIZON,
        'prediction_horizon_minutes': PREDICTION_HORIZON / 10,
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

    with open(os.path.join(MODEL_DIR, 'predictive_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved")

    joblib.dump(scalers, os.path.join(MODEL_DIR, 'scalers.pkl'))
    print(f"Scalers saved")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel predicts anomalies {PREDICTION_HORIZON/10:.1f} minutes in advance")
    print(f"All artifacts saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
