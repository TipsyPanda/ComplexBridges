"""
Z24 Bridge Dataset Loader for CNN-LSTM Encoder-Decoder

This module loads the Z24 bridge monitoring dataset and prepares it for use
with the CNN-LSTM encoder-decoder anomaly prediction model.

Dataset Info:
- Source: EMPA/SIMCES Project (Z24 Bridge, Koppigen, Switzerland)
- Original sampling rate: 100 Hz
- Data: Accelerometer measurements from ~30 channels
- Damage states: Folders 01 (healthy) → 02-17 (progressive damage)

Usage:
    # Full dataset
    from z24_loader import Z24DataLoader

    loader = Z24DataLoader('/home/yanni/data/z24')
    df, labels = loader.load_pdt_data()
    X, y = loader.create_sequences(df, labels, input_steps=100, output_steps=600)

    # Demo dataset (for fast prototyping)
    # Create once:
    loader.save_demo_dataset('/home/yanni/data/z24/z24_demo.npz')

    # Load in notebook:
    data = Z24DataLoader.load_demo_dataset('/home/yanni/data/z24/z24_demo.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta
import joblib
import zipfile
from io import BytesIO


class Z24DataLoader:
    """
    Load and preprocess Z24 bridge dataset for CNN-LSTM encoder-decoder model.

    Attributes:
        data_dir: Path to extracted Z24 dataset
        target_sample_rate: Desired output sample rate (default 10 Hz)
        original_sample_rate: Z24 native sample rate (100 Hz)
    """

    # Default channels to use (representative accelerometers)
    # Format: LocationID + Direction (V=Vertical, L=Longitudinal, T=Transverse)
    DEFAULT_CHANNELS = ['124V', '125V', '224V', '225V', '324V']

    # Damage state mapping (folder number → description)
    DAMAGE_STATES = {
        '01': 'healthy',
        '02': 'settlement_20mm',
        '03': 'settlement_40mm',
        '04': 'settlement_80mm',
        '05': 'settlement_95mm',
        '06': 'tilt_foundation',
        '07': 'tendon_rupture_1',
        '08': 'tendon_rupture_2',
        '09': 'tendon_rupture_3',
        '10': 'tendon_rupture_4',
        '11': 'concrete_spalling_1',
        '12': 'concrete_spalling_2',
        '13': 'landslide_simulation',
        '14': 'anchor_failure_1',
        '15': 'anchor_failure_2',
        '16': 'anchor_failure_3',
        '17': 'final_damage',
    }

    def __init__(self, data_dir: str, target_sample_rate: int = 10):
        """
        Initialize the Z24 data loader.

        Args:
            data_dir: Path to the extracted Z24 dataset directory
            target_sample_rate: Target sample rate for output (default 10 Hz)
        """
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.original_sample_rate = 100
        self.downsample_factor = self.original_sample_rate // target_sample_rate

        # Validate directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")

        # Find PDT directories
        self.pdt_dirs = []
        for subdir in ['pdt_01-08', 'pdt_09_17']:
            path = os.path.join(data_dir, subdir)
            if os.path.exists(path):
                self.pdt_dirs.append(path)

        # Find EMS directories (healthy baseline data)
        self.ems_dirs = []
        for subdir in ['Z24ems1', 'Z24ems2', 'Z24ems3']:
            path = os.path.join(data_dir, subdir)
            if os.path.exists(path):
                self.ems_dirs.append(path)

        if not self.pdt_dirs and not self.ems_dirs:
            raise ValueError(f"No PDT or EMS directories found in {data_dir}")

    def _load_mat_file(self, filepath: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load a single .mat file.

        Args:
            filepath: Path to .mat file

        Returns:
            Tuple of (data array, channel names list)
        """
        mat = sio.loadmat(filepath)
        data = mat['data']  # Shape: (n_samples, n_channels)

        # Channel names - handle different formats
        labelshulp = mat['labelshulp']
        if labelshulp.ndim == 1:
            channels = [str(ch).strip() for ch in labelshulp]
        else:
            channels = [str(ch[0]).strip() for ch in labelshulp.flatten()]

        return data, channels

    def _get_damage_label(self, folder_name: str) -> int:
        """
        Map folder name to binary damage label.

        Args:
            folder_name: Folder name (e.g., '01', '02', ...)

        Returns:
            0 for healthy, 1 for damaged
        """
        # Folder 01 is healthy baseline, all others are damaged
        return 0 if folder_name == '01' else 1

    def _get_damage_state(self, folder_name: str) -> str:
        """
        Get damage state description for a folder.

        Args:
            folder_name: Folder name (e.g., '01', '02', ...)

        Returns:
            Damage state description string
        """
        return self.DAMAGE_STATES.get(folder_name, f'unknown_{folder_name}')

    def _downsample(self, data: np.ndarray) -> np.ndarray:
        """
        Downsample data from original to target sample rate.
        Uses decimation (taking every nth sample).

        Args:
            data: Input data array (n_samples, n_channels)

        Returns:
            Downsampled data array
        """
        return data[::self.downsample_factor]

    def _find_common_channels(self, all_channels: List[List[str]]) -> List[str]:
        """
        Find channels that are common across all files.

        Args:
            all_channels: List of channel lists from each file

        Returns:
            List of common channel names
        """
        if not all_channels:
            return []

        common = set(all_channels[0])
        for channels in all_channels[1:]:
            common &= set(channels)

        return sorted(list(common))

    def get_mat_files(self, test_type: str = 'avt') -> List[Dict]:
        """
        Get list of all .mat files with metadata.

        Args:
            test_type: 'avt' (ambient) or 'fvt' (forced) or 'both'

        Returns:
            List of dicts with filepath, damage_state, damage_label, test_type
        """
        files = []

        for pdt_dir in self.pdt_dirs:
            for folder in sorted(os.listdir(pdt_dir)):
                folder_path = os.path.join(pdt_dir, folder)
                if not os.path.isdir(folder_path):
                    continue

                # Skip non-numeric folders
                if not folder.isdigit():
                    continue

                damage_label = self._get_damage_label(folder)
                damage_state = self._get_damage_state(folder)

                # Check both avt and fvt subdirectories
                for subdir in ['avt', 'fvt']:
                    if test_type != 'both' and subdir != test_type:
                        continue

                    subdir_path = os.path.join(folder_path, subdir)
                    if not os.path.isdir(subdir_path):
                        continue

                    for filename in sorted(os.listdir(subdir_path)):
                        if filename.endswith('.mat'):
                            files.append({
                                'filepath': os.path.join(subdir_path, filename),
                                'damage_folder': folder,
                                'damage_state': damage_state,
                                'damage_label': damage_label,
                                'test_type': subdir,
                                'filename': filename
                            })

        return files

    def _load_ems_aaa_file(self, zip_path: str, aaa_filename: str) -> np.ndarray:
        """
        Load a single .AAA acceleration file from an EMS zip archive.

        Args:
            zip_path: Path to the zip file
            aaa_filename: Name of the .aaa file inside the zip

        Returns:
            numpy array of acceleration values
        """
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(aaa_filename) as f:
                lines = f.read().decode('latin-1').strip().split('\n')
                # Line 1 contains number of samples
                n_samples = int(lines[1].strip())
                # Data starts at line 3 (0=label, 1=n_samples, 2=interval)
                # Only read n_samples lines to avoid footer
                data_lines = lines[3:3+n_samples]
                data = np.array([float(line.strip()) for line in data_lines])
                return data

    def load_ems_data(
        self,
        channels: List[str] = ['03', '05', '06', '07', '10'],
        max_files: Optional[int] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load EMS (Environmental Monitoring System) healthy baseline data.

        The EMS data contains 10 months of measurements from the healthy bridge.
        Each zip file contains one hourly measurement with multiple channels.

        Args:
            channels: List of channel numbers to load (e.g., ['03', '05', '06'])
            max_files: Maximum number of zip files to load (for testing)
            max_samples: Maximum total samples to load (to limit memory usage)
            verbose: Print progress messages

        Returns:
            Tuple of (sensor_df, anomaly_labels):
                - sensor_df: DataFrame with columns for each sensor channel
                - anomaly_labels: Series with all zeros (healthy)
        """
        if not self.ems_dirs:
            raise ValueError("No EMS directories found")

        # Collect all zip files
        zip_files = []
        for ems_dir in self.ems_dirs:
            for filename in sorted(os.listdir(ems_dir)):
                if filename.endswith('.zip'):
                    zip_files.append(os.path.join(ems_dir, filename))

        if max_files:
            zip_files = zip_files[:max_files]

        if verbose:
            print(f"Found {len(zip_files)} EMS zip files")

        all_data = []
        files_loaded = 0
        total_samples = 0

        for i, zip_path in enumerate(zip_files):
            # Check if we have enough samples
            if max_samples and total_samples >= max_samples:
                if verbose:
                    print(f"  Reached {max_samples:,} samples limit, stopping.")
                break

            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    # Get list of .aaa files in this zip
                    aaa_files = [n for n in z.namelist() if n.lower().endswith('.aaa')]

                    # Find files matching our channels
                    channel_data = {}
                    for ch in channels:
                        # EMS files are named like 01C1403.aaa (channel 03)
                        matching = [f for f in aaa_files if f.lower().endswith(f'{ch}.aaa')]
                        if matching:
                            data = self._load_ems_aaa_file(zip_path, matching[0])
                            # Downsample
                            data = data[::self.downsample_factor]
                            channel_data[ch] = data

                    # Only include if we got all channels
                    if len(channel_data) == len(channels):
                        # Stack channels into array
                        min_len = min(len(d) for d in channel_data.values())
                        stacked = np.column_stack([channel_data[ch][:min_len] for ch in channels])
                        all_data.append(stacked)
                        files_loaded += 1
                        total_samples += len(stacked)

                if verbose and (i + 1) % 200 == 0:
                    print(f"  Processed {i + 1}/{len(zip_files)} files, {total_samples:,} samples...")

            except Exception as e:
                if verbose and i < 5:
                    print(f"  Error loading {zip_path}: {e}")

        if not all_data:
            raise ValueError("No EMS data loaded. Check channel numbers.")

        # Concatenate all data
        data_concat = np.vstack(all_data)

        # Trim to max_samples if specified
        if max_samples and len(data_concat) > max_samples:
            data_concat = data_concat[:max_samples]

        # Create DataFrame with channel names
        column_names = [f'EMS_CH{ch}' for ch in channels]
        df = pd.DataFrame(data_concat, columns=column_names)

        # All EMS data is healthy (label=0)
        labels = pd.Series(np.zeros(len(df), dtype=int), name='anomaly')

        if verbose:
            print(f"\n✅ Loaded {len(df):,} EMS samples from {files_loaded} files")
            print(f"   Shape: {df.shape}")
            print(f"   Sample rate: {self.target_sample_rate} Hz")
            print(f"   Duration: {len(df) / self.target_sample_rate / 60:.1f} minutes")
            print(f"   All samples are HEALTHY (label=0)")

        return df, labels

    def load_combined_data(
        self,
        ems_channels: List[str] = ['03', '05', '06', '07', '10'],
        max_ems_files: Optional[int] = None,
        max_pdt_files: Optional[int] = None,
        target_samples_per_class: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load combined EMS (healthy) and PDT (damaged) data.

        Args:
            ems_channels: Channel numbers to load from EMS
            max_ems_files: Max EMS files to load
            max_pdt_files: Max PDT files to load
            target_samples_per_class: Target samples per class (for balancing)
            verbose: Print progress

        Returns:
            Tuple of (sensor_df, anomaly_labels)
        """
        if verbose:
            print("=" * 60)
            print("Loading Combined EMS + PDT Data")
            print("=" * 60)

        # Load EMS (healthy) data
        if verbose:
            print("\n[1/2] Loading EMS (healthy) data...")
        ems_df, ems_labels = self.load_ems_data(
            channels=ems_channels,
            max_files=max_ems_files,
            verbose=verbose
        )

        # Load PDT (damaged) data - only folders 02-17
        if verbose:
            print("\n[2/2] Loading PDT (damaged) data...")
        pdt_df, pdt_labels = self.load_pdt_data(
            max_files=max_pdt_files,
            verbose=verbose
        )
        # Filter to only damaged (label=1)
        damaged_mask = pdt_labels == 1
        pdt_df = pdt_df[damaged_mask].reset_index(drop=True)
        pdt_labels = pdt_labels[damaged_mask].reset_index(drop=True)

        # Balance classes if requested
        if target_samples_per_class:
            n_healthy = min(len(ems_df), target_samples_per_class)
            n_damaged = min(len(pdt_df), target_samples_per_class)
            n_per_class = min(n_healthy, n_damaged)

            ems_df = ems_df.iloc[:n_per_class].reset_index(drop=True)
            ems_labels = ems_labels.iloc[:n_per_class].reset_index(drop=True)
            pdt_df = pdt_df.iloc[:n_per_class].reset_index(drop=True)
            pdt_labels = pdt_labels.iloc[:n_per_class].reset_index(drop=True)

        # Rename columns to match
        ems_df.columns = [f'CH{i}' for i in range(len(ems_df.columns))]
        pdt_df.columns = [f'CH{i}' for i in range(len(pdt_df.columns))]

        # Combine
        combined_df = pd.concat([ems_df, pdt_df], ignore_index=True)
        combined_labels = pd.concat([ems_labels, pdt_labels], ignore_index=True)

        if verbose:
            print(f"\n✅ Combined dataset:")
            print(f"   Total samples: {len(combined_df):,}")
            print(f"   Healthy: {(combined_labels == 0).sum():,} ({(combined_labels == 0).mean()*100:.1f}%)")
            print(f"   Damaged: {(combined_labels == 1).sum():,} ({(combined_labels == 1).mean()*100:.1f}%)")

        return combined_df, combined_labels

    def load_pdt_data(
        self,
        channels: Optional[List[str]] = None,
        test_type: str = 'avt',
        max_files: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load all PDT (Progressive Damage Test) data.

        Args:
            channels: List of channel names to load (None = auto-detect common channels)
            test_type: 'avt' (ambient), 'fvt' (forced), or 'both'
            max_files: Maximum number of files to load (for testing)
            verbose: Print progress messages

        Returns:
            Tuple of (sensor_df, anomaly_labels):
                - sensor_df: DataFrame with columns for each sensor channel
                - anomaly_labels: Series with binary anomaly labels (0=healthy, 1=damaged)
        """
        mat_files = self.get_mat_files(test_type)

        if max_files:
            mat_files = mat_files[:max_files]

        if verbose:
            print(f"Found {len(mat_files)} .mat files")

        # First pass: find common channels if not specified
        if channels is None:
            if verbose:
                print("Auto-detecting common channels...")
            all_channels = []
            for file_info in mat_files[:20]:  # Sample first 20 files
                try:
                    _, ch = self._load_mat_file(file_info['filepath'])
                    all_channels.append(ch)
                except Exception as e:
                    print(f"Warning: Could not read {file_info['filepath']}: {e}")

            common = self._find_common_channels(all_channels)

            # Try to use default channels if available
            channels = [ch for ch in self.DEFAULT_CHANNELS if ch in common]
            if not channels:
                # Fall back to first 5 common channels
                channels = common[:5]

            if verbose:
                print(f"Using channels: {channels}")

        # Second pass: load data
        all_data = []
        all_labels = []

        for i, file_info in enumerate(mat_files):
            try:
                data, file_channels = self._load_mat_file(file_info['filepath'])

                # Extract selected channels
                channel_indices = []
                for ch in channels:
                    if ch in file_channels:
                        channel_indices.append(file_channels.index(ch))
                    else:
                        # Channel not in this file - skip file
                        break

                if len(channel_indices) != len(channels):
                    if verbose and i < 5:
                        print(f"  Skipping {file_info['filename']}: missing channels")
                    continue

                # Extract and downsample
                selected_data = data[:, channel_indices]
                downsampled = self._downsample(selected_data)

                # Create labels array (same label for all samples in file)
                n_samples = len(downsampled)
                labels = np.full(n_samples, file_info['damage_label'])

                all_data.append(downsampled)
                all_labels.append(labels)

                if verbose and (i + 1) % 50 == 0:
                    print(f"  Loaded {i + 1}/{len(mat_files)} files...")

            except Exception as e:
                if verbose:
                    print(f"  Error loading {file_info['filepath']}: {e}")

        if not all_data:
            raise ValueError("No data loaded. Check channels and file paths.")

        # Concatenate all data
        data_concat = np.vstack(all_data)
        labels_concat = np.hstack(all_labels)

        # Create DataFrame
        df = pd.DataFrame(data_concat, columns=channels)
        labels = pd.Series(labels_concat, name='anomaly')

        if verbose:
            print(f"\n✅ Loaded {len(df):,} samples from {len(all_data)} files")
            print(f"   Shape: {df.shape}")
            print(f"   Sample rate: {self.target_sample_rate} Hz")
            print(f"   Duration: {len(df) / self.target_sample_rate / 60:.1f} minutes")
            print(f"   Healthy samples: {(labels == 0).sum():,} ({(labels == 0).mean()*100:.1f}%)")
            print(f"   Damaged samples: {(labels == 1).sum():,} ({(labels == 1).mean()*100:.1f}%)")

        return df, labels

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
        """
        Normalize sensor data using StandardScaler.

        Args:
            df: Sensor DataFrame

        Returns:
            Tuple of (normalized_df, scalers_dict)
        """
        normalized = df.copy()
        scalers = {}

        for col in df.columns:
            scaler = StandardScaler()
            normalized[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

        return normalized, scalers

    def create_sequences(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        input_steps: int = 100,
        output_steps: int = 600,
        output_sample_rate: int = 20,
        stride: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for CNN-LSTM encoder-decoder model.

        This matches the format expected by the notebook's model:
        - Input: (n_samples, input_steps, n_features) - sensor history
        - Output: (n_samples, n_output_steps, 1) - future risk scores

        Args:
            data: Normalized sensor DataFrame
            labels: Anomaly labels Series
            input_steps: Number of input timesteps (default 100 = 10s at 10Hz)
            output_steps: Number of output timesteps before downsampling (default 600 = 60s)
            output_sample_rate: Downsample output to this rate (default 20 = every 2s)
            stride: Sliding window stride (default 10)

        Returns:
            Tuple of (X, y) arrays ready for model training
        """
        data_array = data.values
        labels_array = labels.values

        X_list = []
        y_list = []

        n_output_steps = output_steps // output_sample_rate
        max_idx = len(data_array) - input_steps - output_steps

        for i in range(0, max_idx, stride):
            # Input: Historical window
            X_seq = data_array[i:i+input_steps]

            # Output: Future anomaly labels (downsampled)
            future_start = i + input_steps
            future_labels = labels_array[future_start:future_start + output_steps]

            # Downsample: Take maximum in each interval
            y_seq = []
            for j in range(n_output_steps):
                start_idx = j * output_sample_rate
                end_idx = (j + 1) * output_sample_rate
                y_seq.append(future_labels[start_idx:end_idx].max())

            X_list.append(X_seq)
            y_list.append(y_seq)

        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, n_output_steps, 1)

        return X, y

    def temporal_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data chronologically (no leakage).

        Args:
            X: Input sequences
            y: Output sequences
            train_ratio: Training set ratio
            val_ratio: Validation set ratio (test = 1 - train - val)

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_demo_subset(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        samples_per_state: int = 6000,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a balanced, interleaved demo subset for proper temporal splitting.

        The demo subset interleaves healthy and damaged chunks so that a temporal
        split will include both classes in train/val/test sets.

        Pattern: H-D-H-D-H-D-H-D-H-D (5 healthy chunks, 5 damaged chunks)

        Args:
            df: Full sensor DataFrame
            labels: Full anomaly labels Series
            samples_per_state: Number of samples to take from healthy and damaged
            verbose: Print progress messages

        Returns:
            Tuple of (demo_df, demo_labels)
        """
        # Get indices for each class
        healthy_idx = np.where(labels.values == 0)[0]
        damaged_idx = np.where(labels.values == 1)[0]

        if len(healthy_idx) == 0 or len(damaged_idx) == 0:
            raise ValueError("Data must contain both healthy and damaged samples")

        # Calculate chunk sizes (n_chunks alternating pairs)
        n_chunks = 5  # Creates 10 segments: H-D-H-D-H-D-H-D-H-D
        # Use the same size for both classes to ensure 50/50 balance
        max_per_chunk = min(
            samples_per_state // n_chunks,
            len(healthy_idx) // n_chunks,
            len(damaged_idx) // n_chunks
        )
        healthy_per_chunk = max_per_chunk
        damaged_per_chunk = max_per_chunk

        # Build interleaved segments
        segments = []
        healthy_spacing = len(healthy_idx) // n_chunks
        damaged_spacing = len(damaged_idx) // n_chunks

        for i in range(n_chunks):
            # Healthy chunk - sample from different parts of healthy data
            h_start = healthy_idx[i * healthy_spacing]
            h_end = h_start + healthy_per_chunk
            segments.append((h_start, min(h_end, len(df)), 'healthy'))

            # Damaged chunk - sample from different parts of damaged data
            d_start = damaged_idx[i * damaged_spacing]
            d_end = d_start + damaged_per_chunk
            segments.append((d_start, min(d_end, len(df)), 'damaged'))

        # Concatenate interleaved segments
        demo_dfs = [df.iloc[start:end] for start, end, _ in segments]
        demo_labels_list = [labels.iloc[start:end] for start, end, _ in segments]

        demo_df = pd.concat(demo_dfs, ignore_index=True)
        demo_labels = pd.concat(demo_labels_list, ignore_index=True)

        if verbose:
            print(f"\n📦 Demo subset created (interleaved):")
            print(f"   Total samples: {len(demo_df):,}")
            print(f"   Healthy: {(demo_labels == 0).sum():,} ({(demo_labels == 0).mean()*100:.1f}%)")
            print(f"   Damaged: {(demo_labels == 1).sum():,} ({(demo_labels == 1).mean()*100:.1f}%)")
            print(f"   Duration: {len(demo_df) / self.target_sample_rate / 60:.1f} minutes")
            print(f"   Pattern: {n_chunks} healthy + {n_chunks} damaged chunks interleaved")

        return demo_df, demo_labels

    def save_demo_dataset(
        self,
        output_path: str,
        samples_per_state: int = 6000,
        test_type: str = 'avt',
        verbose: bool = True
    ) -> str:
        """
        Create and save a demo dataset for fast prototyping.

        Saves a .npz file containing:
        - X_train, y_train, X_val, y_val, X_test, y_test: Ready-to-use arrays
        - df: Raw sensor DataFrame
        - labels: Anomaly labels
        - scalers: Fitted StandardScalers for each channel
        - config: Dataset configuration

        Args:
            output_path: Path to save the .npz file
            samples_per_state: Samples per damage state (default 6000 = 10 min each)
            test_type: 'avt', 'fvt', or 'both'
            verbose: Print progress

        Returns:
            Path to saved file
        """
        if verbose:
            print("=" * 60)
            print("Creating Z24 Demo Dataset")
            print("=" * 60)

        # Load full data
        df, labels = self.load_pdt_data(test_type=test_type, verbose=verbose)

        # Create demo subset
        demo_df, demo_labels = self.create_demo_subset(
            df, labels, samples_per_state=samples_per_state, verbose=verbose
        )

        # Normalize
        normalized, scalers = self.normalize(demo_df)

        # Create sequences with stride=1 for more data
        X, y = self.create_sequences(normalized, demo_labels, stride=1)

        # Split (60/20/20 for balanced val/test)
        X_train, y_train, X_val, y_val, X_test, y_test = self.temporal_split(
            X, y, train_ratio=0.6, val_ratio=0.2
        )

        if verbose:
            print(f"\n📊 Sequence shapes:")
            print(f"   X_train: {X_train.shape}")
            print(f"   X_val:   {X_val.shape}")
            print(f"   X_test:  {X_test.shape}")

        # Save configuration
        config = {
            'sample_rate_hz': self.target_sample_rate,
            'original_sample_rate_hz': self.original_sample_rate,
            'input_steps': 100,
            'output_steps': 600,
            'output_sample_rate': 20,
            'n_output_steps': 30,
            'channels': list(demo_df.columns),
            'test_type': test_type,
            'samples_per_state': samples_per_state,
            'created_at': datetime.now().isoformat(),
        }

        # Save as .npz
        np.savez_compressed(
            output_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            df=demo_df.values,
            df_columns=demo_df.columns.values,
            labels=demo_labels.values,
            config=config
        )

        # Save scalers separately (can't easily serialize in npz)
        scalers_path = output_path.replace('.npz', '_scalers.pkl')
        joblib.dump(scalers, scalers_path)

        if verbose:
            print(f"\n✅ Demo dataset saved:")
            print(f"   Data: {output_path}")
            print(f"   Scalers: {scalers_path}")
            print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

        return output_path

    def save_demo_dataset_with_ems(
        self,
        output_path: str,
        samples_per_class: int = 500000,
        verbose: bool = True
    ) -> str:
        """
        Create and save a demo dataset using EMS (healthy) + PDT (damaged) data.

        This uses the full EMS baseline data for healthy samples, providing
        much more training data than using only PDT folder 01.

        Args:
            output_path: Path to save the .npz file
            samples_per_class: Target samples per class (healthy/damaged)
            verbose: Print progress

        Returns:
            Path to saved file
        """
        if verbose:
            print("=" * 60)
            print("Creating Z24 Demo Dataset with EMS Data")
            print("=" * 60)

        # Load EMS (healthy) data - limit to avoid OOM
        if verbose:
            print("\n[1/3] Loading EMS (healthy) data...")
        ems_df, ems_labels = self.load_ems_data(
            max_samples=samples_per_class + 100000,  # Load slightly more than needed
            verbose=verbose
        )

        # Load PDT (damaged) data
        if verbose:
            print("\n[2/3] Loading PDT (damaged) data...")
        pdt_df, pdt_labels = self.load_pdt_data(verbose=verbose)

        # Filter PDT to only damaged samples
        damaged_mask = pdt_labels == 1
        pdt_damaged_df = pdt_df[damaged_mask].reset_index(drop=True)
        pdt_damaged_labels = pdt_labels[damaged_mask].reset_index(drop=True)

        # Balance classes
        n_healthy = min(len(ems_df), samples_per_class)
        n_damaged = min(len(pdt_damaged_df), samples_per_class)
        n_per_class = min(n_healthy, n_damaged)

        if verbose:
            print(f"\n[3/3] Balancing classes to {n_per_class:,} samples each...")

        # Sample from healthy (EMS)
        healthy_df = ems_df.iloc[:n_per_class].reset_index(drop=True)
        healthy_labels = pd.Series(np.zeros(n_per_class, dtype=int), name='anomaly')

        # Sample from damaged (PDT)
        damaged_df = pdt_damaged_df.iloc[:n_per_class].reset_index(drop=True)
        damaged_labels = pd.Series(np.ones(n_per_class, dtype=int), name='anomaly')

        # Rename columns to match (5 channels)
        healthy_df.columns = [f'CH{i}' for i in range(len(healthy_df.columns))]
        damaged_df.columns = [f'CH{i}' for i in range(len(damaged_df.columns))]

        # Interleave for proper temporal splitting
        n_chunks = 10
        chunk_size = n_per_class // n_chunks

        interleaved_dfs = []
        interleaved_labels = []

        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size

            # Add healthy chunk
            interleaved_dfs.append(healthy_df.iloc[start:end])
            interleaved_labels.append(healthy_labels.iloc[start:end])

            # Add damaged chunk
            interleaved_dfs.append(damaged_df.iloc[start:end])
            interleaved_labels.append(damaged_labels.iloc[start:end])

        demo_df = pd.concat(interleaved_dfs, ignore_index=True)
        demo_labels = pd.concat(interleaved_labels, ignore_index=True)

        if verbose:
            print(f"\n📦 Demo subset created (interleaved EMS+PDT):")
            print(f"   Total samples: {len(demo_df):,}")
            print(f"   Healthy: {(demo_labels == 0).sum():,} ({(demo_labels == 0).mean()*100:.1f}%)")
            print(f"   Damaged: {(demo_labels == 1).sum():,} ({(demo_labels == 1).mean()*100:.1f}%)")
            print(f"   Duration: {len(demo_df) / self.target_sample_rate / 60:.1f} minutes")

        # Normalize
        normalized, scalers = self.normalize(demo_df)

        # Create sequences with stride=1 for maximum data
        X, y = self.create_sequences(normalized, demo_labels, stride=1)

        # Split (60/20/20)
        X_train, y_train, X_val, y_val, X_test, y_test = self.temporal_split(
            X, y, train_ratio=0.6, val_ratio=0.2
        )

        if verbose:
            print(f"\n📊 Sequence shapes:")
            print(f"   X_train: {X_train.shape}")
            print(f"   X_val:   {X_val.shape}")
            print(f"   X_test:  {X_test.shape}")

        # Save configuration
        config = {
            'sample_rate_hz': self.target_sample_rate,
            'original_sample_rate_hz': self.original_sample_rate,
            'input_steps': 100,
            'output_steps': 600,
            'output_sample_rate': 20,
            'n_output_steps': 30,
            'channels': list(demo_df.columns),
            'data_source': 'EMS+PDT',
            'samples_per_class': n_per_class,
            'created_at': datetime.now().isoformat(),
        }

        # Save as .npz
        np.savez_compressed(
            output_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            df=demo_df.values,
            df_columns=demo_df.columns.values,
            labels=demo_labels.values,
            config=config
        )

        # Save scalers separately
        scalers_path = output_path.replace('.npz', '_scalers.pkl')
        joblib.dump(scalers, scalers_path)

        if verbose:
            print(f"\n✅ Demo dataset saved:")
            print(f"   Data: {output_path}")
            print(f"   Scalers: {scalers_path}")
            print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

        return output_path

    @staticmethod
    def load_demo_dataset(path: str) -> Dict:
        """
        Load a saved demo dataset.

        Args:
            path: Path to .npz file

        Returns:
            Dict with X_train, y_train, X_val, y_val, X_test, y_test,
            config, and optionally df, labels, scalers
        """
        data = np.load(path, allow_pickle=True)

        result = {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_val': data['X_val'],
            'y_val': data['y_val'],
            'X_test': data['X_test'],
            'y_test': data['y_test'],
            'config': data['config'].item(),
        }

        # Optional fields (v1 has df/labels, v2 doesn't to save memory)
        if 'df' in data.files and 'df_columns' in data.files:
            result['df'] = pd.DataFrame(data['df'], columns=data['df_columns'])
        if 'labels' in data.files:
            result['labels'] = pd.Series(data['labels'], name='anomaly')

        # Load scalers if available
        scalers_path = path.replace('.npz', '_scalers.pkl')
        if os.path.exists(scalers_path):
            result['scalers'] = joblib.load(scalers_path)

        return result

    def _load_ems_temperature(self, zip_path: str) -> Optional[Dict[str, float]]:
        """
        Load temperature data from EMS .env file.

        Args:
            zip_path: Path to the EMS zip file

        Returns:
            Dict with temperature values, or None if not available
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                env_files = [n for n in z.namelist() if n.lower().endswith('.env')]
                if not env_files:
                    return None

                with z.open(env_files[0]) as f:
                    lines = f.read().decode('latin-1').strip().split('\n')
                    if len(lines) < 2:
                        return None

                    headers = lines[0].split()

                    # Headers include units as separate tokens: "TDT1 [°C] TDT2 [°C]"
                    # So header index i corresponds to data index i//2 for label tokens
                    # Find TDT column indices
                    tdt_indices = {}
                    for i, h in enumerate(headers):
                        if h == 'TDT2':
                            tdt_indices['TDT2'] = i // 2  # Data column index
                        elif h == 'TDT1' and 'TDT1' not in tdt_indices:
                            tdt_indices['TDT1'] = i // 2

                    if not tdt_indices:
                        return None

                    # Average the 10 scans (lines 1-10)
                    values_list = []
                    for line in lines[1:11]:
                        try:
                            row_values = [float(x) for x in line.split()]
                            values_list.append(row_values)
                        except ValueError:
                            continue

                    if not values_list:
                        return None

                    avg_values = np.mean(values_list, axis=0)

                    # Extract temperature - prefer TDT2, fallback to TDT1
                    temp_data = {}
                    for key in ['TDT2', 'TDT1']:
                        if key in tdt_indices:
                            idx = tdt_indices[key]
                            if idx < len(avg_values):
                                val = avg_values[idx]
                                # Filter out invalid readings (e.g., 3276.7 is error code)
                                if -50 < val < 60:
                                    temp_data['deck_temp'] = val
                                    break

                    return temp_data if temp_data else None

        except Exception:
            return None

    def load_ems_data_v2(
        self,
        channels: List[str] = ['03', '05', '06', '07', '10'],
        max_files: Optional[int] = None,
        max_samples: Optional[int] = None,
        include_temp: bool = True,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load EMS data with optional temperature feature (v2).

        Args:
            channels: List of channel numbers to load
            max_files: Maximum number of zip files to load
            max_samples: Maximum total samples to load
            include_temp: Whether to include deck temperature as a feature
            verbose: Print progress messages

        Returns:
            Tuple of (sensor_df, anomaly_labels)
        """
        if not self.ems_dirs:
            raise ValueError("No EMS directories found")

        # Collect all zip files
        zip_files = []
        for ems_dir in self.ems_dirs:
            for filename in sorted(os.listdir(ems_dir)):
                if filename.endswith('.zip'):
                    zip_files.append(os.path.join(ems_dir, filename))

        if max_files:
            zip_files = zip_files[:max_files]

        if verbose:
            print(f"Found {len(zip_files)} EMS zip files")
            if include_temp:
                print(f"Including temperature as feature")

        all_data = []
        files_loaded = 0
        total_samples = 0
        temp_count = 0

        for i, zip_path in enumerate(zip_files):
            if max_samples and total_samples >= max_samples:
                if verbose:
                    print(f"  Reached {max_samples:,} samples limit, stopping.")
                break

            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    aaa_files = [n for n in z.namelist() if n.lower().endswith('.aaa')]

                    channel_data = {}
                    for ch in channels:
                        matching = [f for f in aaa_files if f.lower().endswith(f'{ch}.aaa')]
                        if matching:
                            data = self._load_ems_aaa_file(zip_path, matching[0])
                            data = data[::self.downsample_factor]
                            channel_data[ch] = data

                    if len(channel_data) == len(channels):
                        min_len = min(len(d) for d in channel_data.values())
                        stacked = np.column_stack([channel_data[ch][:min_len] for ch in channels])

                        # Add temperature if requested
                        if include_temp:
                            temp_data = self._load_ems_temperature(zip_path)
                            if temp_data and 'deck_temp' in temp_data:
                                temp_col = np.full(min_len, temp_data['deck_temp'])
                                stacked = np.column_stack([stacked, temp_col])
                                temp_count += 1
                            else:
                                # Use NaN for missing temperature
                                temp_col = np.full(min_len, np.nan)
                                stacked = np.column_stack([stacked, temp_col])

                        all_data.append(stacked)
                        files_loaded += 1
                        total_samples += len(stacked)

                if verbose and (i + 1) % 200 == 0:
                    print(f"  Processed {i + 1}/{len(zip_files)} files, {total_samples:,} samples...")

            except Exception as e:
                if verbose and i < 5:
                    print(f"  Error loading {zip_path}: {e}")

        if not all_data:
            raise ValueError("No EMS data loaded. Check channel numbers.")

        data_concat = np.vstack(all_data)

        if max_samples and len(data_concat) > max_samples:
            data_concat = data_concat[:max_samples]

        # Create column names
        column_names = [f'CH{i}' for i in range(len(channels))]
        if include_temp:
            column_names.append('TEMP')

        df = pd.DataFrame(data_concat, columns=column_names)

        # Fill NaN temperatures with median
        if include_temp and df['TEMP'].isna().any():
            median_temp = df['TEMP'].median()
            df['TEMP'] = df['TEMP'].fillna(median_temp)
            if verbose:
                print(f"  Filled {df['TEMP'].isna().sum()} NaN temperatures with median={median_temp:.1f}°C")

        labels = pd.Series(np.zeros(len(df), dtype=int), name='anomaly')

        if verbose:
            print(f"\n✅ Loaded {len(df):,} EMS samples from {files_loaded} files")
            print(f"   Shape: {df.shape}")
            print(f"   Sample rate: {self.target_sample_rate} Hz")
            print(f"   Duration: {len(df) / self.target_sample_rate / 60:.1f} minutes")
            if include_temp:
                print(f"   Temperature: {temp_count}/{files_loaded} files had valid temp")
                print(f"   Temp range: {df['TEMP'].min():.1f}°C to {df['TEMP'].max():.1f}°C")

        return df, labels

    def _create_sequences_chunked(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        input_steps: int,
        output_steps: int,
        output_sample_rate: int,
        stride: int,
        chunk_size: int = 10000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences in chunks to avoid OOM.

        Processes data in chunks, creates sequences for each chunk,
        and concatenates results.
        """
        import gc

        n_output_steps = output_steps // output_sample_rate
        total_window = input_steps + output_steps

        # Calculate total sequences
        n_total = (len(data) - total_window) // stride + 1
        if n_total <= 0:
            raise ValueError(f"Data too short for sequences: {len(data)} < {total_window}")

        # Pre-allocate output arrays
        n_features = data.shape[1]
        X_all = np.zeros((n_total, input_steps, n_features), dtype=np.float32)
        y_all = np.zeros((n_total, n_output_steps, 1), dtype=np.float32)

        # Process in chunks
        seq_idx = 0
        for chunk_start in range(0, n_total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_total)

            for i in range(chunk_start, chunk_end):
                data_start = i * stride
                X_all[seq_idx] = data[data_start:data_start + input_steps]

                # Output labels
                future_start = data_start + input_steps
                future_labels = labels[future_start:future_start + output_steps]

                for j in range(n_output_steps):
                    start_idx = j * output_sample_rate
                    end_idx = (j + 1) * output_sample_rate
                    y_all[seq_idx, j, 0] = future_labels[start_idx:end_idx].max()

                seq_idx += 1

            gc.collect()

        return X_all, y_all

    def save_demo_dataset_v2(
        self,
        output_path: str,
        samples_per_class: int = 100000,
        include_temp: bool = True,
        verbose: bool = True
    ) -> str:
        """
        Create and save a v2 demo dataset with 50Hz sampling and optional temperature.

        Memory-efficient incremental processing for large datasets.

        This version uses:
        - 50 Hz sampling rate (captures modal frequencies up to 25 Hz)
        - Temperature as additional feature (from EMS .env files)
        - EMS data for healthy, PDT (folders 02-17) for damaged
        - **Concatenated healthy→damaged data to create transition sequences**
        - Chunked processing to avoid OOM

        The key difference from v1: sequences are created from a CONTINUOUS
        healthy→damaged timeline, so sequences can span the transition and
        have mixed labels. This prevents data leakage where the model learns
        to distinguish data sources rather than structural health.

        Args:
            output_path: Path to save the .npz file
            samples_per_class: Target samples per class (healthy/damaged)
            include_temp: Whether to include temperature feature
            verbose: Print progress

        Returns:
            Path to saved file
        """
        import gc

        if verbose:
            print("=" * 60)
            print("Creating Z24 Demo Dataset v2 (50Hz + Temperature)")
            print("=" * 60)
            print(f"Sample rate: {self.target_sample_rate} Hz")
            print(f"Memory-efficient incremental processing enabled")
            print(f"Creating continuous healthy→damaged timeline for transitions")

        # Sequence parameters
        input_steps = int(10 * self.target_sample_rate)  # 10 seconds
        output_steps = int(60 * self.target_sample_rate)  # 60 seconds
        output_sample_rate = int(2 * self.target_sample_rate)  # 2 second intervals
        n_output_steps = output_steps // output_sample_rate  # 30 output predictions
        stride = 10  # Larger stride for more diverse sequences

        # Determine number of channels
        n_accel_channels = 5
        n_channels = n_accel_channels + (1 if include_temp else 0)
        col_names = [f'CH{i}' for i in range(n_accel_channels)]
        if include_temp:
            col_names.append('TEMP')

        total_window = input_steps + output_steps

        if verbose:
            print(f"\n📊 Sequence parameters:")
            print(f"   Input: {input_steps} steps ({input_steps/self.target_sample_rate:.0f}s)")
            print(f"   Output: {n_output_steps} predictions ({output_steps/self.target_sample_rate:.0f}s horizon)")
            print(f"   Stride: {stride}")
            print(f"   Channels: {n_channels} ({col_names})")

        # First pass: Load EMS sample to fit scalers and get temperature stats
        if verbose:
            print("\n[1/5] Loading EMS sample for scaler fitting...")

        ems_sample, _ = self.load_ems_data_v2(
            max_samples=50000,
            include_temp=include_temp,
            verbose=False
        )
        ems_sample.columns = col_names

        # Fit scalers on sample
        scalers = {}
        for col in col_names:
            scaler = StandardScaler()
            scaler.fit(ems_sample[[col]])
            scalers[col] = scaler

        median_temp = ems_sample['TEMP'].median() if include_temp else None
        del ems_sample
        gc.collect()

        if verbose:
            print(f"   Scalers fitted on 50K samples")
            if include_temp:
                print(f"   Median temperature: {median_temp:.1f}°C")

        # Second pass: Count available data
        if verbose:
            print("\n[2/5] Counting available data...")

        ems_zip_files = []
        for ems_dir in self.ems_dirs:
            for filename in sorted(os.listdir(ems_dir)):
                if filename.endswith('.zip'):
                    ems_zip_files.append(os.path.join(ems_dir, filename))

        pdt_df_temp, pdt_labels_temp = self.load_pdt_data(verbose=False)
        n_pdt_damaged = (pdt_labels_temp == 1).sum()
        del pdt_df_temp, pdt_labels_temp
        gc.collect()

        n_per_class = min(samples_per_class, n_pdt_damaged)

        if verbose:
            print(f"   EMS files available: {len(ems_zip_files)}")
            print(f"   PDT damaged samples: {n_pdt_damaged:,}")
            print(f"   Target per class: {n_per_class:,}")

        # Third pass: Load PDT damaged data
        if verbose:
            print("\n[3/5] Loading PDT damaged data...")

        pdt_df, pdt_labels = self.load_pdt_data(verbose=False)
        damaged_mask = pdt_labels == 1
        pdt_damaged = pdt_df[damaged_mask].values[:n_per_class]

        if include_temp:
            temp_col = np.full((len(pdt_damaged), 1), median_temp)
            pdt_damaged = np.hstack([pdt_damaged, temp_col])

        del pdt_df, pdt_labels, damaged_mask
        gc.collect()

        if verbose:
            print(f"   Loaded {len(pdt_damaged):,} damaged samples")

        # Fourth pass: Load EMS healthy data incrementally
        if verbose:
            print("\n[4/5] Loading EMS healthy data and creating timeline...")

        # We'll create multiple healthy→damaged segments for variety
        n_segments = 5  # Create 5 transition timelines
        segment_size = n_per_class // n_segments

        X_all = []
        y_all = []
        total_sequences = 0

        ems_file_idx = 0

        for seg_idx in range(n_segments):
            if verbose:
                print(f"   Segment {seg_idx + 1}/{n_segments}...")

            # Load EMS chunk for this segment
            ems_chunk_data = []
            ems_chunk_samples = 0
            target_samples = segment_size

            while ems_chunk_samples < target_samples and ems_file_idx < len(ems_zip_files):
                zip_path = ems_zip_files[ems_file_idx]
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        aaa_files = [n for n in z.namelist() if n.lower().endswith('.aaa')]
                        channels = ['03', '05', '06', '07', '10']

                        channel_data = {}
                        for ch in channels:
                            matching = [f for f in aaa_files if f.lower().endswith(f'{ch}.aaa')]
                            if matching:
                                data = self._load_ems_aaa_file(zip_path, matching[0])
                                data = data[::self.downsample_factor]
                                channel_data[ch] = data

                        if len(channel_data) == len(channels):
                            min_len = min(len(d) for d in channel_data.values())
                            stacked = np.column_stack([channel_data[ch][:min_len] for ch in channels])

                            if include_temp:
                                temp_data = self._load_ems_temperature(zip_path)
                                if temp_data and 'deck_temp' in temp_data:
                                    temp_val = temp_data['deck_temp']
                                else:
                                    temp_val = median_temp
                                temp_col_arr = np.full((min_len, 1), temp_val)
                                stacked = np.hstack([stacked, temp_col_arr])

                            ems_chunk_data.append(stacked)
                            ems_chunk_samples += len(stacked)

                except Exception:
                    pass

                ems_file_idx += 1

            if not ems_chunk_data:
                break

            # Combine EMS chunk
            ems_chunk = np.vstack(ems_chunk_data)[:segment_size]
            del ems_chunk_data
            gc.collect()

            # Get corresponding PDT segment
            pdt_start = seg_idx * segment_size
            pdt_end = min(pdt_start + segment_size, len(pdt_damaged))
            pdt_chunk = pdt_damaged[pdt_start:pdt_end]

            # Create continuous timeline: HEALTHY → DAMAGED
            # This allows sequences to span the transition
            timeline_data = np.vstack([ems_chunk, pdt_chunk])
            timeline_labels = np.concatenate([
                np.zeros(len(ems_chunk), dtype=int),
                np.ones(len(pdt_chunk), dtype=int)
            ])

            del ems_chunk, pdt_chunk
            gc.collect()

            # Normalize timeline
            timeline_norm = np.zeros_like(timeline_data, dtype=np.float32)
            for i, col in enumerate(col_names):
                timeline_norm[:, i] = scalers[col].transform(timeline_data[:, i:i+1]).flatten()

            del timeline_data
            gc.collect()

            # Create sequences from this timeline
            # Sequences will naturally span the healthy→damaged transition
            if len(timeline_norm) > total_window:
                X_seg, y_seg = self._create_sequences_chunked(
                    timeline_norm, timeline_labels,
                    input_steps, output_steps, output_sample_rate, stride
                )
                X_all.append(X_seg)
                y_all.append(y_seg)
                total_sequences += len(X_seg)

                if verbose:
                    # Count mixed sequences in this segment
                    y_means = y_seg.mean(axis=(1, 2))
                    mixed = np.sum((y_means > 0) & (y_means < 1))
                    print(f"      Created {len(X_seg):,} sequences ({mixed:,} mixed-label)")

            del timeline_norm, timeline_labels
            gc.collect()

        del pdt_damaged
        gc.collect()

        # Fifth pass: Concatenate and split
        if verbose:
            print(f"\n[5/5] Concatenating and splitting...")

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        del X_all, y_all
        gc.collect()

        # Shuffle to mix segments (important for train/val/test split)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Split (60/20/20)
        n_samples = len(X)
        train_end = int(n_samples * 0.6)
        val_end = int(n_samples * 0.8)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        del X, y
        gc.collect()

        if verbose:
            print(f"\n📊 Final sequence shapes:")
            print(f"   X_train: {X_train.shape}")
            print(f"   X_val:   {X_val.shape}")
            print(f"   X_test:  {X_test.shape}")

            # Analyze label distribution
            def analyze_labels(y_data, name):
                y_means = y_data.mean(axis=(1, 2))
                all_healthy = np.sum(y_means == 0)
                all_damaged = np.sum(y_means == 1)
                mixed = np.sum((y_means > 0) & (y_means < 1))
                print(f"   {name}: {all_healthy} all-healthy, {all_damaged} all-damaged, {mixed} mixed")

            print(f"\n📊 Sequence label analysis:")
            analyze_labels(y_train, "Train")
            analyze_labels(y_val, "Val")
            analyze_labels(y_test, "Test")

            # Class balance
            train_anomaly_pct = y_train.mean() * 100
            val_anomaly_pct = y_val.mean() * 100
            test_anomaly_pct = y_test.mean() * 100
            print(f"\n📊 Class balance (anomaly %):")
            print(f"   Train: {train_anomaly_pct:.1f}%")
            print(f"   Val:   {val_anomaly_pct:.1f}%")
            print(f"   Test:  {test_anomaly_pct:.1f}%")

        # Save configuration
        config = {
            'sample_rate_hz': self.target_sample_rate,
            'original_sample_rate_hz': self.original_sample_rate,
            'input_steps': input_steps,
            'output_steps': output_steps,
            'output_sample_rate': output_sample_rate,
            'n_output_steps': n_output_steps,
            'channels': col_names,
            'include_temp': include_temp,
            'data_source': 'EMS+PDT_v2_transitions',
            'samples_per_class': n_per_class,
            'n_segments': n_segments,
            'created_at': datetime.now().isoformat(),
        }

        # Save as .npz
        np.savez_compressed(
            output_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            config=config
        )

        # Save scalers separately
        scalers_path = output_path.replace('.npz', '_scalers.pkl')
        joblib.dump(scalers, scalers_path)

        if verbose:
            print(f"\n✅ Demo dataset v2 saved:")
            print(f"   Data: {output_path}")
            print(f"   Scalers: {scalers_path}")
            print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

        return output_path


def main():
    """Example usage of Z24DataLoader."""
    import argparse

    parser = argparse.ArgumentParser(description='Load Z24 bridge dataset')
    parser.add_argument('--data-dir', default='/home/yanni/data/z24',
                        help='Path to Z24 dataset')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum files to load (for testing)')
    parser.add_argument('--test-type', default='avt', choices=['avt', 'fvt', 'both'],
                        help='Test type to load')
    parser.add_argument('--create-demo', action='store_true',
                        help='Create and save demo dataset')
    parser.add_argument('--demo-output', default='/home/yanni/data/z24/z24_demo.npz',
                        help='Output path for demo dataset')
    parser.add_argument('--demo-samples', type=int, default=6000,
                        help='Samples per state for demo (default 6000 = 10 min each)')
    args = parser.parse_args()

    loader = Z24DataLoader(args.data_dir)

    if args.create_demo:
        # Create and save demo dataset
        loader.save_demo_dataset(
            output_path=args.demo_output,
            samples_per_state=args.demo_samples,
            test_type=args.test_type
        )
        return

    # Load data
    df, labels = loader.load_pdt_data(
        test_type=args.test_type,
        max_files=args.max_files
    )

    # Normalize
    normalized, scalers = loader.normalize(df)

    # Create sequences
    X, y = loader.create_sequences(normalized, labels)

    print(f"\n📊 Sequence shapes:")
    print(f"   X: {X.shape} (samples, input_steps, features)")
    print(f"   y: {y.shape} (samples, output_steps, 1)")

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = loader.temporal_split(X, y)

    print(f"\n📊 Split sizes:")
    print(f"   Train: {len(X_train)}")
    print(f"   Val:   {len(X_val)}")
    print(f"   Test:  {len(X_test)}")


if __name__ == '__main__':
    main()
