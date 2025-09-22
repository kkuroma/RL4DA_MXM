import os
import json
import numpy as np
import time
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import sys

# Add src to path for imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from envs.enkf_env import CurriculumManager


class TensorBoardLoggingCallback(BaseCallback):
    """
    Enhanced TensorBoard logging callback for ENKF RL training.

    Logs comprehensive metrics including:
    - Training rewards and episode statistics
    - RMSE metrics
    - Curriculum learning progress
    - Model diagnostics
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_rmse = []
        self.timestep_counter = 0

    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Log hyperparameters - simplified approach without HParam class
        try:
            hparam_dict = {
                'learning_rate': float(self.model.learning_rate) if hasattr(self.model, 'learning_rate') else 0.0,
                'batch_size': float(getattr(self.model, 'batch_size', 256)),
                'n_steps': float(getattr(self.model, 'n_steps', 2048)),
                'gamma': float(getattr(self.model, 'gamma', 0.99)),
                'ent_coef': float(getattr(self.model, 'ent_coef', 0.1)),
            }
            # Log each hyperparameter separately
            for key, value in hparam_dict.items():
                self.logger.record(f"hparams/{key}", value)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log hyperparameters: {e}")

    def _on_step(self) -> bool:
        """Called at each step."""
        self.timestep_counter += 1

        # Log step-level metrics from info
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]

            # Episode completion logging
            if 'episode' in ep_info:
                episode_data = ep_info['episode']

                # Basic episode metrics
                episode_reward = episode_data.get('r', 0)
                episode_length = episode_data.get('l', 0)
                final_rmse = episode_data.get('final_rmse', float('inf'))
                mean_rmse = episode_data.get('mean_rmse', float('inf'))
                path_idx = episode_data.get('path_idx', -1)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_rmse.append(mean_rmse)

                # Log to TensorBoard
                self.logger.record("episode/reward", episode_reward)
                self.logger.record("episode/length", episode_length)
                self.logger.record("episode/final_rmse", final_rmse)
                self.logger.record("episode/mean_rmse", mean_rmse)
                self.logger.record("episode/path_idx", path_idx)

                # Log curriculum-related metrics if available
                if 'active_paths_count' in episode_data:
                    self.logger.record("curriculum/active_paths_count", episode_data['active_paths_count'])
                if 'queue_position' in episode_data:
                    self.logger.record("curriculum/queue_position", episode_data['queue_position'])

                # Running averages
                if len(self.episode_rewards) >= 10:
                    recent_rewards = self.episode_rewards[-10:]
                    recent_lengths = self.episode_lengths[-10:]
                    recent_rmse = self.episode_rmse[-10:]

                    self.logger.record("episode/reward_mean_10", np.mean(recent_rewards))
                    self.logger.record("episode/length_mean_10", np.mean(recent_lengths))
                    self.logger.record("episode/rmse_mean_10", np.mean(recent_rmse))

                if len(self.episode_rewards) >= 100:
                    recent_rewards = self.episode_rewards[-100:]
                    recent_lengths = self.episode_lengths[-100:]
                    recent_rmse = self.episode_rmse[-100:]

                    self.logger.record("episode/reward_mean_100", np.mean(recent_rewards))
                    self.logger.record("episode/length_mean_100", np.mean(recent_lengths))
                    self.logger.record("episode/rmse_mean_100", np.mean(recent_rmse))

        # Log model diagnostics every 1000 steps
        if self.timestep_counter % 1000 == 0:
            self._log_model_diagnostics()

        return True

    def _log_model_diagnostics(self) -> None:
        """Log model-specific diagnostics."""
        try:
            # Policy network gradients
            if hasattr(self.model.policy, 'features_extractor'):
                features_extractor = self.model.policy.features_extractor

                # LSTM diagnostics
                if hasattr(features_extractor, 'temporal_lstm'):
                    lstm = features_extractor.temporal_lstm

                    # Log gradient norms
                    total_norm = 0
                    param_count = 0
                    for name, param in lstm.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                            self.logger.record(f"gradients/lstm_{name}", param_norm.item())

                    if param_count > 0:
                        total_norm = total_norm ** (1. / 2)
                        self.logger.record("gradients/lstm_total_norm", total_norm)

            # Action distribution diagnostics
            if hasattr(self.model.policy, 'log_std'):
                log_std = self.model.policy.log_std
                std = log_std.exp()
                self.logger.record("policy/action_std_mean", std.mean().item())
                self.logger.record("policy/action_std_max", std.max().item())
                self.logger.record("policy/action_std_min", std.min().item())

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log model diagnostics: {e}")


class CurriculumCallback(BaseCallback):
    """
    Curriculum learning callback that progressively adds training paths.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        save_path: str,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.config = config
        self.save_path = save_path

        # Curriculum settings
        self.enable_curriculum = config.get('enable_curriculum', True)
        start_paths = config.get('curriculum_start_paths', 1)
        max_paths = config.get('curriculum_max_paths', 99)
        episodes_per_level = config.get('curriculum_episodes_per_path', 100)

        # Initialize curriculum manager
        self.curriculum = CurriculumManager(
            total_paths=max_paths,
            start_paths=start_paths,
            episodes_per_level=episodes_per_level
        )

        # State tracking
        self.state_file = os.path.join(save_path, "curriculum_state.json")
        self._load_state()

        # Episode tracking
        self.episodes_this_session = 0

    def _load_state(self) -> None:
        """Load curriculum state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.curriculum.current_level = state.get('current_level', 1)
                self.curriculum.episodes_completed = state.get('episodes_completed', 0)
                self.curriculum.successful_episodes = state.get('successful_episodes', 0)
                if self.verbose > 0:
                    print(f"Loaded curriculum state: Level {self.curriculum.current_level}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to load curriculum state: {e}. Starting fresh.")

    def _save_state(self) -> None:
        """Save curriculum state to disk."""
        state = {
            'current_level': self.curriculum.current_level,
            'episodes_completed': self.curriculum.episodes_completed,
            'successful_episodes': self.curriculum.successful_episodes,
            'episodes_this_session': self.episodes_this_session
        }
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _on_step(self) -> bool:
        """Called at each step."""
        if not self.enable_curriculum:
            return True

        # Check for episode completion
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]

            if 'episode' in ep_info:
                episode_data = ep_info['episode']
                episode_reward = episode_data.get('r', 0)
                episode_length = episode_data.get('l', 0)

                self.episodes_this_session += 1

                # Update curriculum
                level_increased = self.curriculum.update(episode_reward, episode_length)

                if level_increased:
                    active_paths = self.curriculum.get_active_paths()
                    if self.verbose > 0:
                        print(f"\n🎓 Curriculum level increased to {self.curriculum.current_level}")
                        print(f"   Active training paths: {len(active_paths)} {active_paths}")

                    # Log curriculum progress
                    stats = self.curriculum.get_stats()
                    for key, value in stats.items():
                        self.logger.record(f"curriculum/{key}", value)

                    # Update environment if possible
                    self._update_environment_paths(active_paths)

                # Save state periodically
                if self.episodes_this_session % 10 == 0:
                    self._save_state()

                # Log curriculum stats every episode
                stats = self.curriculum.get_stats()
                for key, value in stats.items():
                    self.logger.record(f"curriculum/{key}", value)

        return True

    def _update_environment_paths(self, active_paths: list) -> None:
        """Update active paths in the training environment."""
        try:
            # Get training environment from model
            training_env = self.training_env if hasattr(self, 'training_env') else None
            if training_env is None and hasattr(self.model, 'env'):
                training_env = self.model.env

            if training_env is not None:
                if hasattr(training_env, 'set_active_paths'):
                    training_env.set_active_paths(active_paths)
                elif hasattr(training_env, 'envs'):
                    # VecEnv case - update all environments
                    for env in training_env.envs:
                        if hasattr(env, 'set_active_paths'):
                            env.set_active_paths(active_paths)
                elif hasattr(training_env, 'venv') and hasattr(training_env.venv, 'envs'):
                    # DummyVecEnv case
                    for env in training_env.venv.envs:
                        if hasattr(env, 'set_active_paths'):
                            env.set_active_paths(active_paths)
                else:
                    if self.verbose > 0:
                        print(f"Training environment type: {type(training_env)}")
                        print(f"Available methods: {[m for m in dir(training_env) if not m.startswith('_')]}")
            else:
                if self.verbose > 0:
                    print("Warning: Could not access training environment")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not update environment paths: {e}")
                import traceback
                traceback.print_exc()

    def _on_training_end(self) -> None:
        """Called when training ends."""
        self._save_state()


class ModelCheckpointCallback(BaseCallback):
    """
    Enhanced model checkpoint callback with additional metadata.
    """

    def __init__(
        self,
        save_path: str,
        save_freq: int = 10000,
        name_prefix: str = "rl_model",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.save_freq == 0:
            # Save model
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps")
            self.model.save(model_path)

            # Save metadata
            metadata = {
                'timesteps': self.n_calls,
                'save_time': time.time(),
                'model_class': self.model.__class__.__name__,
            }

            # Add training metrics if available
            if hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
                metadata['recent_mean_reward'] = np.mean(self.episode_rewards[-10:])

            metadata_path = model_path + "_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")

        return True


def create_callbacks(config: Dict[str, Any], save_path: str, eval_env) -> list:
    """
    Create training callbacks based on configuration.

    Args:
        config: Configuration dictionary
        save_path: Path for saving callback state
        eval_env: Evaluation environment

    Returns:
        List of configured callbacks
    """
    callbacks = []

    # Always add TensorBoard logging
    tb_callback = TensorBoardLoggingCallback(verbose=1)
    callbacks.append(tb_callback)

    # Add curriculum learning if enabled
    if config.get('enable_curriculum', True):
        curriculum_callback = CurriculumCallback(
            config=config,
            save_path=save_path,
            verbose=1
        )
        callbacks.append(curriculum_callback)

    # Add model checkpointing
    checkpoint_callback = ModelCheckpointCallback(
        save_path=os.path.join(save_path, "checkpoints"),
        save_freq=config.get('save_freq', 50000),
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # Add evaluation callback
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best_model"),
        log_path=os.path.join(save_path, "eval_logs"),
        eval_freq=config.get('eval_freq', 10000),
        n_eval_episodes=config.get('eval_episodes', 10),
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)

    return callbacks