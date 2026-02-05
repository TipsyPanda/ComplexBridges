# -*- coding: utf-8 -*-
"""
visualize_data.py
Visualization of the HBTA bridge sensor data from data_100Hz.h5
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Open the HDF5 file
h5_file = 'data_100Hz.h5'
data_h5 = h5py.File(h5_file, 'r')

# Print the structure of the file
print("=" * 60)
print("HDF5 FILE STRUCTURE")
print("=" * 60)

def print_structure(name, obj):
    """Print the structure of the HDF5 file."""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{indent}{name}/")

data_h5.visititems(print_structure)

# Get list of recordings
recordings = list(data_h5.keys())
print("\n" + "=" * 60)
print(f"AVAILABLE RECORDINGS ({len(recordings)} total)")
print("=" * 60)
for rec in recordings[:10]:
    print(f"  - {rec}")
if len(recordings) > 10:
    print(f"  ... and {len(recordings) - 10} more")

# Select first recording for visualization
recording = recordings[0]
print(f"\nVisualizing data from: {recording}")

# Get sensor groups
sensor_groups = list(data_h5[recording].keys())
print(f"Sensor groups: {sensor_groups}")

# Sampling rate
fs = 100  # Hz (from filename)

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(f'Bridge Sensor Data Visualization\nRecording: {recording}', fontsize=14)

# Plot 1: Acceleration from AG01 (global accelerometer)
if 'acceleration' in data_h5[recording]:
    acc_sensors = list(data_h5[recording]['acceleration'].keys())

    # Plot AG01 y and z channels
    if 'AG01' in acc_sensors:
        ag01_y = data_h5[recording]['acceleration']['AG01']['y'][:]
        ag01_z = data_h5[recording]['acceleration']['AG01']['z'][:]
        t = np.arange(len(ag01_y)) / fs

        axes[0, 0].plot(t, ag01_y, 'b-', linewidth=0.5, alpha=0.8)
        axes[0, 0].set_title('AG01 - Global Accelerometer (Y-direction)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Acceleration')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(t, ag01_z, 'r-', linewidth=0.5, alpha=0.8)
        axes[0, 1].set_title('AG01 - Global Accelerometer (Z-direction)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Acceleration')
        axes[0, 1].grid(True, alpha=0.3)

# Plot 2: Acceleration from AL01 (local accelerometer)
if 'acceleration' in data_h5[recording] and 'AL01' in acc_sensors:
    al01_z = data_h5[recording]['acceleration']['AL01']['z'][:]
    t = np.arange(len(al01_z)) / fs

    axes[1, 0].plot(t, al01_z, 'g-', linewidth=0.5, alpha=0.8)
    axes[1, 0].set_title('AL01 - Local Accelerometer (Z-direction)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Acceleration')
    axes[1, 0].grid(True, alpha=0.3)

# Plot 3: Strain data
if 'strain' in data_h5[recording]:
    strain_sensors = list(data_h5[recording]['strain'].keys())

    if 'SB01' in strain_sensors:
        sb01_x = data_h5[recording]['strain']['SB01']['x'][:]
        t = np.arange(len(sb01_x)) / fs

        axes[1, 1].plot(t, sb01_x, 'm-', linewidth=0.5, alpha=0.8)
        axes[1, 1].set_title('SB01 - Strain Gage on Lower Chord Beam (X-direction)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Strain')
        axes[1, 1].grid(True, alpha=0.3)

# Plot 4: Multiple accelerometers comparison
if 'acceleration' in data_h5[recording]:
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, sensor in enumerate(['AG01', 'AG02', 'AG03', 'AG04', 'AG05']):
        if sensor in acc_sensors:
            data = data_h5[recording]['acceleration'][sensor]['z'][:]
            t = np.arange(len(data)) / fs
            axes[2, 0].plot(t, data + i*0.5, color=colors[i], linewidth=0.5,
                           alpha=0.8, label=sensor)

    axes[2, 0].set_title('Multiple Global Accelerometers (Z-direction, offset for clarity)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Acceleration (offset)')
    axes[2, 0].legend(loc='upper right', fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

# Plot 5: FFT of acceleration data
if 'acceleration' in data_h5[recording] and 'AG01' in acc_sensors:
    ag01_z = data_h5[recording]['acceleration']['AG01']['z'][:]
    n = len(ag01_z)
    fft_data = np.abs(np.fft.rfft(ag01_z))
    freqs = np.fft.rfftfreq(n, 1/fs)

    axes[2, 1].plot(freqs, fft_data, 'k-', linewidth=0.5)
    axes[2, 1].set_title('FFT of AG01 Z-direction Acceleration')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude')
    axes[2, 1].set_xlim(0, 50)  # Show up to 50 Hz
    axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
if 'acceleration' in data_h5[recording] and 'AG01' in acc_sensors:
    ag01_z = data_h5[recording]['acceleration']['AG01']['z'][:]
    print(f"AG01 Z-direction:")
    print(f"  Duration: {len(ag01_z)/fs:.2f} seconds")
    print(f"  Samples: {len(ag01_z)}")
    print(f"  Mean: {np.mean(ag01_z):.6f}")
    print(f"  Std: {np.std(ag01_z):.6f}")
    print(f"  Min: {np.min(ag01_z):.6f}")
    print(f"  Max: {np.max(ag01_z):.6f}")

data_h5.close()
print("\nVisualization saved to 'visualization_output.png'")
