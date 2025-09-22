import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import random
import sys
import os

# Add src to path for imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from utils.normalization import DataNormalizer, compute_temporal_features


class ENKFEnvironment(gym.Env):
    """
    RL Environment for ENKF with proper temporal structure.

    The environment respects the physics of ENKF and maintains proper temporal ordering:
    - Temporal sequences are processed with LSTM (previous analysis/error states)
    - Current timestep data is processed simultaneously (ensemble, observations, truth)
    - Actions predict analysis corrections in normalized space
    """

    def __init__(
        self,
        paths: List[Dict[str, np.ndarray]],
        normalizer: DataNormalizer,
        config: Dict[str, Any],
        eval_path_idx: int = 0,
        device: str = 'cpu'
    ):
        super().__init__()

        self.paths = paths
        self.normalizer = normalizer
        self.config = config
        self.eval_path_idx = eval_path_idx
        self.device = device

        # Environment configuration
        self.temporal_window = config.get('temporal_window', 3)
        self.max_episode_length = config.get('max_episode_length', 1000)
        self.early_termination_threshold = config.get('early_termination_threshold', 5.0)

        # Curriculum learning setup
        self.active_paths = list(range(1, len(paths)))  # All paths except eval (path 0)
        self.path_queue = self.active_paths.copy()  # Queue for sequential path selection
        self.current_path_queue_idx = 0

        # Extract dimensions from first path
        first_path = paths[0]
        self.N = first_path['true_states'].shape[1]  # State dimension
        self.N_ens = first_path['forecast_states'].shape[2]  # Ensemble size
        self.N_obs = first_path['observations'].shape[1]  # Observation dimension

        # Compute feature dimensions
        self._compute_feature_dimensions()

        # Define observation and action spaces
        self._setup_spaces()

        # Episode state
        self.reset()

    def _compute_feature_dimensions(self):
        """Compute dimensions for temporal and current features."""
        # Temporal sequence features per timestep: [analysis_error, forecast_error, ensemble_std]
        self.temporal_feature_dim = 3 * self.N

        # Current state features: [forecast_ensemble, truth, observations, ensemble_mean, ensemble_std]
        self.current_feature_dim = (
            self.N * self.N_ens +  # Flattened ensemble forecast
            self.N +               # Truth
            self.N_obs +           # Observations
            self.N +               # Ensemble mean
            self.N                 # Ensemble std
        )

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space: Dict with temporal sequence and current state
        self.observation_space = spaces.Dict({
            'temporal_sequence': spaces.Box(
                low=-3.0,
                high=3.0,
                shape=(self.temporal_window - 1, self.temporal_feature_dim),
                dtype=np.float32
            ),
            'current_state': spaces.Box(
                low=-3.0,
                high=3.0,
                shape=(self.current_feature_dim,),
                dtype=np.float32
            )
        })

        # Action space: Analysis correction in normalized space
        self.action_space = spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(self.N,),
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, is_eval: bool = False, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment for new episode."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Select path for this episode
        if is_eval:
            self.current_path_idx = self.eval_path_idx
        else:
            # Use curriculum-based sequential path selection
            if not self.path_queue:
                # Reset queue if empty
                self.path_queue = self.active_paths.copy()
                self.current_path_queue_idx = 0

            # Get next path from queue (sequential order)
            self.current_path_idx = self.path_queue[self.current_path_queue_idx]
            self.current_path_queue_idx = (self.current_path_queue_idx + 1) % len(self.path_queue)

        self.current_path = self.paths[self.current_path_idx]

        # Precompute temporal features for this path
        self.temporal_data = compute_temporal_features(
            true_states=self.current_path['true_states'],
            forecast_states=self.current_path['forecast_states'],
            analysis_states=self.current_path['analysis_states'],
            observations=self.current_path['observations'],
            normalizer=self.normalizer,
            temporal_window=self.temporal_window
        )

        # Episode state
        self.timestep = 0
        self.episode_length = min(
            self.temporal_data['valid_timesteps'],
            self.max_episode_length
        )
        self.cumulative_reward = 0.0
        self.rmse_history = []

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take environment step."""
        action = np.array(action, dtype=np.float32)

        # Clip action to valid range
        action = np.clip(action, -3.0, 3.0)

        # Compute reward
        reward = self._compute_reward(action)
        self.cumulative_reward += reward

        # Update timestep
        self.timestep += 1

        # Check termination conditions
        terminated = self.timestep >= self.episode_length
        truncated = self._check_early_termination(action)

        # Get next observation
        if not (terminated or truncated):
            obs = self._get_observation()
        else:
            # Return dummy observation for terminated episodes
            obs = {
                'temporal_sequence': np.zeros((self.temporal_window - 1, self.temporal_feature_dim), dtype=np.float32),
                'current_state': np.zeros(self.current_feature_dim, dtype=np.float32)
            }

        info = self._get_info()
        if terminated or truncated:
            info['episode'] = {
                'r': self.cumulative_reward / max(self.timestep, 1),  # Average reward
                'l': self.timestep,
                'path_idx': self.current_path_idx,
                'final_rmse': self.rmse_history[-1] if self.rmse_history else float('inf'),
                'mean_rmse': np.mean(self.rmse_history) if self.rmse_history else float('inf')
            }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        if self.timestep >= self.episode_length:
            # Return dummy observation for out-of-bounds timesteps
            return {
                'temporal_sequence': np.zeros((self.temporal_window - 1, self.temporal_feature_dim), dtype=np.float32),
                'current_state': np.zeros(self.current_feature_dim, dtype=np.float32)
            }

        return {
            'temporal_sequence': self.temporal_data['temporal_sequence'][self.timestep].astype(np.float32),
            'current_state': self.temporal_data['current_state'][self.timestep].astype(np.float32)
        }

    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute reward based on analysis quality.

        Args:
            action: Predicted analysis correction (normalized)

        Returns:
            Reward in range approximately [-1, 1]
        """
        if self.timestep >= self.episode_length:
            return 0.0

        # Get true analysis target (normalized)
        true_analysis_norm = self.temporal_data['target_analysis'][self.timestep]

        # Compute RMSE in normalized space
        rmse_norm = np.sqrt(np.mean((action - true_analysis_norm) ** 2))
        self.rmse_history.append(rmse_norm)

        # Convert RMSE to reward
        # RMSE in normalized space is roughly [0, 6] -> map to [-1, 1]
        # Good predictions (low RMSE) get positive rewards
        max_rmse = 2.0  # Maximum expected RMSE in normalized space
        reward = 1.0 - (rmse_norm / max_rmse)
        reward = np.clip(reward, -1.0, 1.0)
        
        return float(reward)

    def _check_early_termination(self, action: np.ndarray) -> bool:
        """Check if episode should terminate early due to instability."""
        # Check for NaN or infinite actions
        if not np.isfinite(action).all():
            return True

        # Check if RMSE is too high (model diverging)
        if len(self.rmse_history) > 0:
            if self.rmse_history[-1] > self.early_termination_threshold:
                return True

        # Check for consistent poor performance
        if len(self.rmse_history) >= 10:
            recent_rmse = np.mean(self.rmse_history[-10:])
            if recent_rmse > 2.0:  # Poor performance threshold
                return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        info = {
            'timestep': self.timestep,
            'path_idx': self.current_path_idx,
            'episode_length': self.episode_length,
            'cumulative_reward': self.cumulative_reward,
            'active_paths_count': len(self.active_paths),
            'queue_position': self.current_path_queue_idx
        }

        if self.rmse_history:
            info['current_rmse'] = self.rmse_history[-1]
            info['mean_rmse'] = np.mean(self.rmse_history)

        return info

    def render(self, mode: str = 'human') -> None:
        """Render environment (not implemented)."""
        pass

    def close(self) -> None:
        """Close environment."""
        pass

    def set_active_paths(self, active_paths: List[int]) -> None:
        """
        Update active paths for curriculum learning.
        New paths are inserted at the front of the queue.
        """
        self.active_paths = active_paths

        # If we have new paths that weren't in the queue, insert them at the front
        if len(active_paths) > len(self.path_queue):
            new_paths = [p for p in active_paths if p not in self.path_queue]
            # Insert new paths at the front so they are used immediately
            self.path_queue = new_paths + self.path_queue
        else:
            # Update existing queue
            self.path_queue = active_paths.copy()

        # Reset index if we're beyond the queue length
        if self.current_path_queue_idx >= len(self.path_queue):
            self.current_path_queue_idx = 0


class ENKFEvaluationEnvironment(ENKFEnvironment):
    """Evaluation environment that always uses the evaluation path."""

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset for evaluation (always use eval path)."""
        return super().reset(seed=seed, is_eval=True, **kwargs)


def create_enkf_env(paths: List[Dict[str, np.ndarray]], normalizer: DataNormalizer, config: Dict[str, Any]) -> ENKFEnvironment:
    """Create ENKF training environment."""
    return ENKFEnvironment(paths, normalizer, config)


def create_enkf_eval_env(paths: List[Dict[str, np.ndarray]], normalizer: DataNormalizer, config: Dict[str, Any]) -> ENKFEvaluationEnvironment:
    """Create ENKF evaluation environment."""
    return ENKFEvaluationEnvironment(paths, normalizer, config)


class CurriculumManager:
    """
    Manages curriculum learning by progressively adding more training paths.
    """

    def __init__(self, total_paths: int, start_paths: int = 1, episodes_per_level: int = 100):
        self.total_paths = total_paths
        self.start_paths = start_paths
        self.episodes_per_level = episodes_per_level

        self.current_level = 1
        self.episodes_completed = 0
        self.episodes_needed = episodes_per_level
        self.successful_episodes = 0

    def update(self, episode_reward: float, episode_length: int) -> bool:
        """
        Update curriculum state after episode completion.

        Args:
            episode_reward: Average reward for the episode
            episode_length: Length of completed episode

        Returns:
            True if curriculum level increased
        """
        self.episodes_completed += 1

        # Count as successful if episode completed without early termination
        # and achieved reasonable performance
        if episode_reward > -0.5 and episode_length > 100:
            self.successful_episodes += 1

        # Check if we should advance to next level
        if self.episodes_completed >= self.episodes_needed:
            success_rate = self.successful_episodes / self.episodes_completed

            if success_rate >= 0.7 and self.current_level < self.total_paths:  # 70% success rate
                self.current_level += 1
                self.episodes_completed = 0
                self.successful_episodes = 0
                self.episodes_needed = max(1, self.episodes_per_level // self.current_level)
                return True

        return False

    def get_active_paths(self) -> List[int]:
        """Get list of currently active training paths."""
        return list(range(1, min(self.current_level + 1, self.total_paths)))

    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        return {
            'current_level': self.current_level,
            'active_paths': len(self.get_active_paths()),
            'episodes_completed': self.episodes_completed,
            'episodes_needed': self.episodes_needed,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(self.episodes_completed, 1)
        }