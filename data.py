import os
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Tuple, List, Dict


# Configuration constants
SESSION_CONFIGS = {
    'audio': {
        'class_ranges': {
            'bad': '1', 'go': '2', 'good': '3', 'happy': '4', 'hello': '5',
            'help': '6', 'no': '7', 'stop': '8', 'thanks': '9', 'yes': '10'
        },
        'baseline_mode': 'eyes_closed',
        'baseline_codes': [102]
    },
    'read': {
        'class_ranges': {
            'bad': (1, 10), 'go': (11, 20), 'good': (21, 30), 'happy': (31, 40),
            'hello': (41, 50), 'help': (51, 60), 'no': (61, 70), 'stop': (71, 80),
            'thanks': (81, 90), 'yes': (91, 100)
        },
        'baseline_mode': 'eyes_open',
        'baseline_codes': [101]
    }
}


def compute_subject_baseline(raw, events, baseline_codes: List[int]) -> np.ndarray:
    """Compute baseline using specified event codes"""
    baseline_events = [e for e in events if e[2] in baseline_codes]
    if not baseline_events:
        return None
    
    baseline_epochs = mne.Epochs(
        raw, events=np.array(baseline_events), event_id=None,
        tmin=0, tmax=2.0, baseline=None, preload=True
    )
    return np.mean(baseline_epochs.get_data(), axis=(0, 2))


def build_event_mapping(original_event_id: Dict, class_ranges: Dict, 
                       session_type: str) -> Tuple[Dict, Dict, Dict]:
    """Build mappings between original events, classes, and standardized event IDs"""
    standardized_event_id = {}
    file_event_id = {}
    class_mapping = {}
    next_event_id = 1000
    
    if session_type == 'audio':
        for class_name, event_str in class_ranges.items():
            if event_str in original_event_id:
                key = f"{class_name}_{event_str}"
                if key not in standardized_event_id:
                    standardized_event_id[key] = next_event_id
                    next_event_id += 1
                file_event_id[key] = standardized_event_id[key]
                class_mapping[original_event_id[event_str]] = (class_name, standardized_event_id[key])
    else:  # read
        for class_name, (start_code, end_code) in class_ranges.items():
            for code in range(start_code, end_code + 1):
                if str(code) in original_event_id:
                    key = f"{class_name}_{code}"
                    if key not in standardized_event_id:
                        standardized_event_id[key] = next_event_id
                        next_event_id += 1
                    file_event_id[key] = standardized_event_id[key]
                    class_mapping[original_event_id[str(code)]] = (class_name, standardized_event_id[key])
    
    return standardized_event_id, file_event_id, class_mapping


def load_and_preprocess_data_with_subjects(edf_paths: List[str], 
                                          session_type: str = 'read') -> Tuple:
    """
    Load and preprocess EEG data from EDF files
    
    Parameters:
    -----------
    edf_paths : list
        List of EDF file paths
    session_type : str
        'audio' for imagined speech or 'read' for silent reading
    
    Returns:
    --------
    X : np.ndarray
        EEG data (samples, channels, time_points)
    Y : np.ndarray
        Labels
    subjects : np.ndarray
        Subject IDs for each sample
    class_names : list
        List of class names
    combined_epochs : mne.Epochs
        Combined epochs object
    """
    if session_type not in SESSION_CONFIGS:
        raise ValueError(f"session_type must be 'audio' or 'read', got '{session_type}'")
    
    config = SESSION_CONFIGS[session_type]
    class_ranges = config['class_ranges']
    baseline_codes = config['baseline_codes']
    
    all_epochs = []
    all_metadata = []
    subject_ids = []
    standardized_event_id = {}
    next_event_id = 1000
    
    for edf_path in edf_paths:
        print(f"Loading: {edf_path}")
        subject_id = os.path.basename(edf_path).split('_')[0]
        
        # Load and preprocess raw data
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.filter(l_freq=1., h_freq=50., fir_design='firwin', verbose=False)
        raw.resample(250)
        
        events, original_event_id = mne.events_from_annotations(raw)
        baseline_mean = compute_subject_baseline(raw, events, baseline_codes)
        
        # Build event mappings
        _, file_event_id, class_mapping = build_event_mapping(
            original_event_id, class_ranges, session_type
        )
        
        # Update global standardized_event_id
        for key, local_id in file_event_id.items():
            if key not in standardized_event_id:
                standardized_event_id[key] = next_event_id
                next_event_id += 1
        
        # Filter valid events
        valid_events = np.array([
            [ev[0], ev[1], class_mapping[ev[2]][1]]
            for ev in events if ev[2] in class_mapping
        ])
        
        if len(valid_events) == 0:
            print(f"Warning: No valid events in {edf_path}, skipping...")
            continue
        
        # Create epochs
        epochs = mne.Epochs(
            raw, events=valid_events, event_id=file_event_id,
            tmin=0, tmax=0.999, baseline=None, preload=True, verbose=False
        )
        
        # Apply baseline correction
        if baseline_mean is not None:
            epochs._data -= baseline_mean[:, np.newaxis]
        
        # Create metadata
        metadata = pd.DataFrame({
            'event_id': [e[2] for e in epochs.events],
            'class': [next(cn for orig_id, (cn, std_id) in class_mapping.items() 
                          if std_id == e[2]) for e in epochs.events],
            'source_file': edf_path,
            'subject_id': subject_id
        })
        
        all_epochs.append(epochs)
        all_metadata.append(metadata)
        subject_ids.extend([subject_id] * len(epochs.events))
        
        print(f"Loaded {len(epochs)} epochs from subject {subject_id}")
    
    if not all_epochs:
        raise RuntimeError("No valid data found in provided EDF paths")
    
    # Combine all epochs
    combined_epochs = mne.concatenate_epochs(all_epochs)
    combined_epochs.metadata = pd.concat(all_metadata, ignore_index=True)
    print(f"Total: {len(combined_epochs)} epochs")
    
    # Organize data by class
    class_names = list(class_ranges.keys())
    class_to_index = {cls: i for i, cls in enumerate(class_names)}
    
    X_list, Y_list, subject_list = [], [], []
    for class_name in class_names:
        idxs = combined_epochs.metadata['class'] == class_name
        if idxs.any():
            data = combined_epochs[idxs.values].get_data()
            X_list.append(data)
            Y_list.append(np.full(len(data), class_to_index[class_name]))
            subject_list.extend(combined_epochs.metadata.loc[idxs, 'subject_id'].values)
    
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    subjects = np.array(subject_list)
    
    print(f"\nFinal dataset: {X.shape}")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(Y)))}")
    print(f"Unique subjects: {np.unique(subjects)}")
    print(f"Samples per subject: {dict(zip(*np.unique(subjects, return_counts=True)))}")
    
    return X, Y, subjects, class_names, combined_epochs


def preprocess_and_tensorize(X: np.ndarray, Y: np.ndarray, 
                             n_channels: int = 64, 
                             n_timepoints: int = 250) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess EEG data: select channels/timepoints, normalize, add channel dim
    
    Parameters:
    -----------
    X : np.ndarray
        Raw EEG data (samples, channels, time_points)
    Y : np.ndarray
        Labels
    n_channels : int
        Number of channels to use (default: 64)
    n_timepoints : int
        Number of time points to use (default: 250)
    
    Returns:
    --------
    X_tensor : torch.Tensor
        Preprocessed data (samples, 1, channels, time_points)
    Y_tensor : torch.Tensor
        Labels as long tensor
    """
    # Select channels and timepoints
    X = X[:, :n_channels, :n_timepoints]
    
    # Normalize each sample individually
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    
    # Add channel dimension (batch_size, 1, channels, time_points)
    X = X[:, np.newaxis, :, :]
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    
    return X_tensor, Y_tensor


def create_data_loaders(X_train: torch.Tensor, Y_train: torch.Tensor,
                       X_val: torch.Tensor, Y_val: torch.Tensor,
                       X_test: torch.Tensor, Y_test: torch.Tensor,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from tensors"""
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def prepare_data_loaders_within_subject(X, Y, subjects, target_subject, 
                                        args, random_seed=42):
    """Prepare data loaders for within-subject evaluation"""
    # Filter data for target subject
    subject_mask = subjects == target_subject
    X_subject = X[subject_mask]
    Y_subject = Y[subject_mask]
    
    print(f"Subject {target_subject}: {len(X_subject)} samples")
    
    # Preprocess and tensorize
    X_tensor, Y_tensor = preprocess_and_tensorize(X_subject, Y_subject)
    
    print(f"Class distribution: {dict(zip(*np.unique(Y_tensor.numpy(), return_counts=True)))}")
    
    # Split: 80% train, 10% val, 10% test
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_tensor, Y_tensor, test_size=0.2, random_state=random_seed, stratify=Y_tensor
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=random_seed, stratify=Y_temp
    )
    
    print(f"Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return create_data_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, 
                              args['batch_size'])


def prepare_data_loaders_cross_validation(X, Y, subjects, train_idx, val_idx, 
                                          test_idx, args):
    """Prepare data loaders for k-fold cross-validation"""
    # Preprocess and tensorize
    X_tensor, Y_tensor = preprocess_and_tensorize(X, Y)
    
    # Split using indices
    X_train, X_val, X_test = X_tensor[train_idx], X_tensor[val_idx], X_tensor[test_idx]
    Y_train, Y_val, Y_test = Y_tensor[train_idx], Y_tensor[val_idx], Y_tensor[test_idx]
    
    print(f"Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train classes: {np.bincount(Y_train.numpy())}")
    print(f"Val classes: {np.bincount(Y_val.numpy())}")
    print(f"Test classes: {np.bincount(Y_test.numpy())}")
    
    return create_data_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, 
                              args['batch_size'])


def prepare_data_loaders_loso(X, Y, subjects, test_subject, args):
    """Prepare data loaders for Leave-One-Subject-Out cross-validation"""
    # Preprocess and tensorize
    X_tensor, Y_tensor = preprocess_and_tensorize(X, Y)
    
    # Split by subject
    train_mask = subjects != test_subject
    test_mask = subjects == test_subject
    
    X_test, Y_test = X_tensor[test_mask], Y_tensor[test_mask]
    
    # Split training subjects into train and validation
    train_subjects = subjects[train_mask]
    unique_train_subjects = np.unique(train_subjects)
    
    train_subjs, val_subjs = train_test_split(
        unique_train_subjects, test_size=0.2, random_state=42
    )
    
    # Create masks for train and validation
    val_mask = np.isin(subjects, val_subjs) & train_mask
    train_mask_final = train_mask & ~val_mask
    
    X_train = X_tensor[train_mask_final]
    Y_train = Y_tensor[train_mask_final]
    X_val = X_tensor[val_mask]
    Y_val = Y_tensor[val_mask]
    
    print(f"Test subject {test_subject}:")
    print(f"  Train: {len(X_train)} samples ({len(train_subjs)} subjects)")
    print(f"  Val: {len(X_val)} samples ({len(val_subjs)} subjects)")
    print(f"  Test: {len(X_test)} samples")
    
    return create_data_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, 
                              args['batch_size'])