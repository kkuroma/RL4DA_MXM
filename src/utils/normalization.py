import numpy as np
import json
from typing import Dict, Tuple, Optional, Any
import os


class DataNormalizer:
    """
    Centralized data normalization for ENKF RL environment.

    Handles normalization to [-3, 3] range with proper unnormalization capabilities.
    Stores normalization factors for consistency across training/evaluation.
    """

    def __init__(self, target_range: Tuple[float, float] = (-3.0, 3.0)):
        """
        Initialize normalizer.

        Args:
            target_range: Target range for normalized values (min, max)
        """
        self.target_min, self.target_max = target_range
        self.target_range = self.target_max - self.target_min

        # Store normalization factors for each data type
        self.norm_factors: Dict[str, Dict[str, float]] = {}

    def fit(self, data: Dict[str, np.ndarray], percentile_range: Tuple[float, float] = (1.0, 99.0)):
        """
        Compute normalization factors from data.

        Args:
            data: Dictionary of data arrays to compute normalization from
            percentile_range: Use percentiles instead of min/max for robustness
        """
        self.norm_factors = {}

        for key, values in data.items():
            if isinstance(values, (list, tuple)):
                # Handle list of arrays (e.g., multiple trajectories)
                all_values = np.concatenate([np.array(v).flatten() for v in values])
            else:
                all_values = np.array(values).flatten()

            # Remove NaN/inf values
            all_values = all_values[np.isfinite(all_values)]

            if len(all_values) == 0:
                raise ValueError(f"No valid values found for key '{key}'")

            # Use percentiles for robust normalization
            data_min = np.percentile(all_values, percentile_range[0])
            data_max = np.percentile(all_values, percentile_range[1])

            # Ensure non-zero range
            if np.abs(data_max - data_min) < 1e-10:
                data_max = data_min + 1.0

            self.norm_factors[key] = {
                'min': float(data_min),
                'max': float(data_max),
                'range': float(data_max - data_min)
            }

    def normalize(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """
        Normalize data to target range.

        Args:
            data: Data to normalize
            data_type: Type of data (must exist in norm_factors)

        Returns:
            Normalized data in target range
        """
        if data_type not in self.norm_factors:
            raise ValueError(f"No normalization factors found for data type '{data_type}'")

        factors = self.norm_factors[data_type]

        # Normalize to [0, 1]
        normalized = (data - factors['min']) / factors['range']

        # Scale to target range
        normalized = normalized * self.target_range + self.target_min

        # Clip to target range (with small buffer for numerical stability)
        buffer = 0.1
        normalized = np.clip(normalized,
                           self.target_min - buffer,
                           self.target_max + buffer)

        return normalized

    def unnormalize(self, normalized_data: np.ndarray, data_type: str) -> np.ndarray:
        """
        Convert normalized data back to original scale.

        Args:
            normalized_data: Data in normalized range
            data_type: Type of data (must exist in norm_factors)

        Returns:
            Data in original scale
        """
        if data_type not in self.norm_factors:
            raise ValueError(f"No normalization factors found for data type '{data_type}'")

        factors = self.norm_factors[data_type]

        # Scale from target range to [0, 1]
        unnormalized = (normalized_data - self.target_min) / self.target_range

        # Scale to original range
        unnormalized = unnormalized * factors['range'] + factors['min']

        return unnormalized

    def save(self, filepath: str):
        """Save normalization factors to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.norm_factors, f, indent=2)

    def load(self, filepath: str):
        """Load normalization factors from JSON file."""
        with open(filepath, 'r') as f:
            self.norm_factors = json.load(f)

    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics for debugging."""
        stats = {}
        for data_type, factors in self.norm_factors.items():
            stats[data_type] = {
                'original_range': [factors['min'], factors['max']],
                'target_range': [self.target_min, self.target_max],
                'scale_factor': factors['range'] / self.target_range
            }
        return stats


def create_enkf_normalizer(paths: list, config: dict) -> DataNormalizer:
    """
    Create and fit normalizer for ENKF data.

    Args:
        paths: List of precomputed ENKF paths
        config: Configuration dictionary

    Returns:
        Fitted DataNormalizer instance
    """
    normalizer = DataNormalizer(target_range=(-3.0, 3.0))

    # Collect all data for normalization
    all_data = {
        'true_states': [],
        'forecast_states': [],
        'analysis_states': [],
        'observations': [],
        'ensemble_mean': [],
        'ensemble_std': [],
        'forecast_error': [],
        'analysis_error': []
    }

    print("Computing normalization factors from all paths...")

    for path in paths:
        # True states
        all_data['true_states'].append(path['true_states'])

        # Forecast states (all ensemble members)
        forecast_flat = path['forecast_states'].reshape(-1, path['forecast_states'].shape[-1])
        all_data['forecast_states'].append(forecast_flat)

        # Analysis states (all ensemble members)
        analysis_flat = path['analysis_states'].reshape(-1, path['analysis_states'].shape[-1])
        all_data['analysis_states'].append(analysis_flat)

        # Observations
        all_data['observations'].append(path['observations'])

        # Ensemble statistics
        ensemble_mean = np.mean(path['forecast_states'], axis=2)
        ensemble_std = np.std(path['forecast_states'], axis=2)
        all_data['ensemble_mean'].append(ensemble_mean)
        all_data['ensemble_std'].append(ensemble_std)

        # Errors
        forecast_error = ensemble_mean - path['true_states']
        analysis_mean = np.mean(path['analysis_states'], axis=2)
        analysis_error = analysis_mean - path['true_states']
        all_data['forecast_error'].append(forecast_error)
        all_data['analysis_error'].append(analysis_error)

    # Fit normalizer
    normalizer.fit(all_data)

    # Print normalization statistics
    stats = normalizer.get_stats()
    print("\nNormalization Statistics:")
    for data_type, stat in stats.items():
        print(f"  {data_type}:")
        print(f"    Original range: [{stat['original_range'][0]:.3f}, {stat['original_range'][1]:.3f}]")
        print(f"    Scale factor: {stat['scale_factor']:.3f}")

    return normalizer


def compute_temporal_features(
    true_states: np.ndarray,
    forecast_states: np.ndarray,
    analysis_states: np.ndarray,
    observations: np.ndarray,
    normalizer: DataNormalizer,
    temporal_window: int = 3
) -> Dict[str, np.ndarray]:
    """
    Compute temporal features for LSTM input.

    Args:
        true_states: Shape (T, N)
        forecast_states: Shape (T, N, N_ens)
        analysis_states: Shape (T, N, N_ens)
        observations: Shape (T, N_obs)
        normalizer: Fitted normalizer
        temporal_window: Number of timesteps for temporal sequence

    Returns:
        Dictionary with temporal and current features
    """
    T, N = true_states.shape
    N_ens = forecast_states.shape[2]

    # Compute ensemble statistics
    ensemble_mean = np.mean(forecast_states, axis=2)  # (T, N)
    ensemble_std = np.std(forecast_states, axis=2)    # (T, N)

    # Compute errors
    forecast_error = ensemble_mean - true_states  # (T, N)
    analysis_mean = np.mean(analysis_states, axis=2)
    analysis_error = analysis_mean - true_states  # (T, N)

    # Normalize all components
    norm_true = normalizer.normalize(true_states, 'true_states')
    norm_forecast_error = normalizer.normalize(forecast_error, 'forecast_error')
    norm_ensemble_std = normalizer.normalize(ensemble_std, 'ensemble_std')
    norm_analysis_error = normalizer.normalize(analysis_error, 'analysis_error')
    norm_observations = normalizer.normalize(observations, 'observations')
    norm_ensemble_mean = normalizer.normalize(ensemble_mean, 'ensemble_mean')
    norm_forecast_flat = normalizer.normalize(
        forecast_states.reshape(-1, N), 'forecast_states'
    ).reshape(T, N, N_ens)

    # Build temporal features (for LSTM)
    # Each timestep contains: [analysis_error, forecast_error, ensemble_std]
    temporal_features = []
    for t in range(temporal_window-1, T):
        seq = []
        for dt in range(temporal_window-1):  # t-2, t-1 (not including t)
            idx = t - (temporal_window-2) + dt
            step_features = np.concatenate([
                norm_analysis_error[idx],     # Previous analysis error
                norm_forecast_error[idx],     # Previous forecast error
                norm_ensemble_std[idx],       # Previous uncertainty
            ])
            seq.append(step_features)
        temporal_features.append(np.array(seq))

    temporal_features = np.array(temporal_features)  # (T-window+1, window-1, feature_dim)

    # Build current state features (simultaneous at timestep t)
    current_features = []
    for t in range(temporal_window-1, T):
        step_features = np.concatenate([
            norm_forecast_flat[t].flatten(),  # Current ensemble forecast
            norm_true[t],                     # Current truth (for training)
            norm_observations[t],             # Current observations
            norm_ensemble_mean[t],            # Current ensemble mean
            norm_ensemble_std[t],             # Current ensemble std
        ])
        current_features.append(step_features)

    current_features = np.array(current_features)  # (T-window+1, feature_dim)

    return {
        'temporal_sequence': temporal_features,
        'current_state': current_features,
        'valid_timesteps': T - temporal_window + 1,
        'target_analysis': norm_true[temporal_window-1:]  # Target for RL actions
    }