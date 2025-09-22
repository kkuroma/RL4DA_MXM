import os
import sys
import importlib.util
import argparse
import torch
from typing import Dict, Any, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add src to path for imports
src_path = os.path.dirname(os.path.dirname(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)

from utils.data_generation import load_or_generate_data, verify_data_quality, print_data_statistics
from envs.enkf_env import create_enkf_env, create_enkf_eval_env
from agents.policies import TemporalLSTMActorCriticPolicy
from training.callbacks import create_callbacks


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def setup_device(config: Dict[str, Any]) -> str:
    """Setup and validate device configuration."""
    use_cuda = config.get("use_cuda", False)

    if use_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        if use_cuda:
            print("CUDA requested but not available, using CPU")
        else:
            print("Using CPU device")

    return device


def create_environments(paths, normalizer, config, device):
    """Create training and evaluation environments."""
    # Training environment
    train_env = create_enkf_env(paths, normalizer, config)
    train_env = Monitor(train_env)

    # Evaluation environment (always uses path 0)
    eval_env = create_enkf_eval_env(paths, normalizer, config)
    eval_env = Monitor(eval_env)

    return train_env, eval_env


def setup_tensorboard_logging(save_path: str) -> str:
    """Setup TensorBoard logging directory."""
    tensorboard_log_dir = os.path.join(save_path, "tensorboard")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    return tensorboard_log_dir


def create_ppo_agent(env, config: Dict[str, Any], tensorboard_log_dir: str, device: str) -> PPO:
    """Create PPO agent with temporal LSTM policy."""

    # Temporal LSTM configuration
    temporal_window = config.get('temporal_window', 3)
    lstm_hidden_size = config.get('lstm_hidden_size', 512)
    num_lstm_layers = config.get('num_lstm_layers', 2)
    features_dim = config.get('features_dim', 512)

    # Policy configuration
    policy_kwargs = {
        "temporal_window": temporal_window,
        "lstm_hidden_size": lstm_hidden_size,
        "num_lstm_layers": num_lstm_layers,
        "features_dim": features_dim,
        "net_arch": dict(
            pi=[256, 256],  # Policy network architecture
            vf=[256, 256]   # Value network architecture
        ),
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True,
        "log_std_init": 0.0,  # Initial exploration
    }

    # PPO hyperparameters - optimized for LSTM and ENKF task
    ppo_config = {
        "policy": TemporalLSTMActorCriticPolicy,
        "env": env,
        "learning_rate": config.get("learning_rate", 3e-4),  # Conservative for LSTM
        "n_steps": config.get("n_steps", 2048),
        "batch_size": config.get("batch_size", 64),  # Smaller batches for LSTM
        "n_epochs": config.get("n_epochs", 10),
        "gamma": config.get("gamma", 0.99),
        "gae_lambda": config.get("gae_lambda", 0.95),
        "clip_range": config.get("clip_range", 0.2),
        "ent_coef": config.get("ent_coef", 0.01),  # Small entropy for precision tasks
        "vf_coef": config.get("vf_coef", 0.5),
        "max_grad_norm": config.get("max_grad_norm", 0.5),  # Important for LSTM stability
        "policy_kwargs": policy_kwargs,
        "tensorboard_log": tensorboard_log_dir,
        "device": device,
        "verbose": 1,
    }

    print("Creating PPO agent with configuration:")
    for key, value in ppo_config.items():
        if key != "policy_kwargs":
            print(f"  {key}: {value}")

    print("Policy configuration:")
    for key, value in policy_kwargs.items():
        print(f"  {key}: {value}")

    return PPO(**ppo_config)


def train_model(
    config_dir: str,
    force_regenerate: bool = False,
    max_timesteps: int = None
) -> PPO:
    """
    Main training function.

    Args:
        config_dir: Directory containing config.py
        force_regenerate: Force data regeneration
        max_timesteps: Override total timesteps (for testing)

    Returns:
        Trained PPO model
    """
    print(f"Starting training with config from: {config_dir}")

    # Load configuration
    config_path = os.path.join(config_dir, "config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    print(f"Loaded configuration for: {config.get('l96_params', {})}")

    # Setup device
    device = setup_device(config)

    # Load or generate data
    print("\n=== Data Loading/Generation ===")
    paths, normalizer = load_or_generate_data(config, config_dir, force_regenerate)

    # Verify data quality
    if not verify_data_quality(paths, config):
        raise RuntimeError("Data quality verification failed!")

    # Print data statistics
    print_data_statistics(paths, normalizer)

    # Create environments
    print("\n=== Environment Setup ===")
    train_env, eval_env = create_environments(paths, normalizer, config, device)
    print(f"Created training and evaluation environments")
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")

    # Setup training infrastructure
    save_path = os.path.join(config_dir, "training_results")
    os.makedirs(save_path, exist_ok=True)

    tensorboard_log_dir = setup_tensorboard_logging(save_path)
    print(f"TensorBoard logs: {tensorboard_log_dir}")

    # Create PPO agent
    print("\n=== Model Creation ===")
    model = create_ppo_agent(train_env, config, tensorboard_log_dir, device)

    # Setup callbacks
    callbacks = create_callbacks(config, save_path, eval_env)
    print(f"Created {len(callbacks)} training callbacks")

    # Training
    total_timesteps = max_timesteps or config.get("total_timesteps", 1000000)
    print(f"\n=== Training Start ===")
    print(f"Training for {total_timesteps} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False
        )

        print("\n=== Training Complete ===")

        # Save final model
        final_model_path = os.path.join(save_path, "final_model")
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Save normalizer for inference
        normalizer_path = os.path.join(save_path, "normalizer.json")
        normalizer.save(normalizer_path)
        print(f"Normalizer saved to: {normalizer_path}")

        return model

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current model
        interrupt_model_path = os.path.join(save_path, "interrupted_model")
        model.save(interrupt_model_path)
        print(f"Interrupted model saved to: {interrupt_model_path}")
        return model

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train RL agent for ENKF")
    parser.add_argument("--config_dir", type=str, required=True,
                       help="Directory containing config.py")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regenerate all data")
    parser.add_argument("--max-timesteps", type=int, default=None,
                       help="Override total timesteps (for testing)")
    parser.add_argument("--test", action="store_true",
                       help="Run short test training (10k timesteps)")

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.max_timesteps = 10000
        print("Running in test mode (10k timesteps)")

    try:
        model = train_model(
            config_dir=args.config_dir,
            force_regenerate=args.force_regenerate,
            max_timesteps=args.max_timesteps
        )
        print("Training completed successfully!")
        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())