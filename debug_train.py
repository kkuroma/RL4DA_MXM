#!/usr/bin/env python3
"""
Short debugging training script for TorchRL multi-agent PPO

This is a minimal version of the training script for quick testing and debugging.
It runs for a very short time with minimal logging.

Usage:
    python debug_train.py --dir=logs/L96_1 [--device=cpu] [--steps=5]
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from tqdm import tqdm

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'scripts'))

from train_torchrl import (
    make_env,
    make_ppo_models,
    load_global_config,
)
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


def debug_train(log_dir: str, device: str = "cpu", num_steps: int = 5):
    """
    Minimal training loop for debugging.

    Args:
        log_dir: Path to log directory
        device: Device to use (cpu or cuda:0)
        num_steps: Number of training steps to run
    """
    print("="*60)
    print("DEBUG TRAINING MODE")
    print("="*60)
    print(f"Log directory: {log_dir}")
    print(f"Device: {device}")
    print(f"Number of steps: {num_steps}")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load config
    print("Loading configuration...")
    global_config = load_global_config()
    rl_config = global_config.get("rl_multiagent", global_config.get("rl", {}))
    print(f"✓ Config loaded")
    print()

    # Create environment
    print("Creating environment...")
    env = make_env(log_dir, device=device, is_eval=False)
    print(f"✓ Environment created")
    print(f"  Number of agents: {env.n_agents}")
    print(f"  N (state dim): {env.base_env.N}")
    print(f"  N_ens (ensemble size): {env.base_env.N_ens}")
    print(f"  Observation shape per agent: {env.observation_spec['observation'].shape}")
    print(f"  Expected obs dim: {3*env.base_env.N*env.base_env.N_ens + env.base_env.N}")
    print(f"  Action shape per agent: {env.action_spec.shape}")
    print(f"  Architecture: Centralized observations, Decentralized actions (CTDE)")
    print()

    # Create models
    print("Creating policy and value networks...")
    policy_module, value_module = make_ppo_models(env, device)
    print(f"✓ Models created")
    print(f"  Policy params: {sum(p.numel() for p in policy_module.parameters()):,}")
    print(f"  Value params: {sum(p.numel() for p in value_module.parameters()):,}")
    print()

    # Hyperparameters for debugging (small values for speed)
    frames_per_batch = 500  # Small batch for quick testing
    num_epochs = 2
    num_minibatches = 4
    lr = 1e-3  # Higher LR for faster updates in debug mode

    clip_epsilon = rl_config.get("clip_range", 0.2)
    gamma = rl_config.get("gamma", 0.99)
    lmbda = rl_config.get("gae_lambda", 0.95)
    entropy_coef = rl_config.get("ent_coef", 0.01)
    critic_coef = rl_config.get("vf_coef", 0.5)

    print("Training hyperparameters:")
    print(f"  Frames per batch: {frames_per_batch}")
    print(f"  Num epochs: {num_epochs}")
    print(f"  Num minibatches: {num_minibatches}")
    print(f"  Learning rate: {lr}")
    print()

    # Create collector
    print("Creating data collector...")
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * num_steps,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )
    print(f"✓ Collector created")
    print()

    # Create replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        batch_size=frames_per_batch // num_minibatches,
    )

    # Create advantage module
    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        average_gae=True,
    )

    # Create loss module
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=True,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef,
        loss_critic_type="smooth_l1",
        normalize_advantage=True,
    )

    # Create optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=lr)

    # Training loop
    print("="*60)
    print("Starting training loop...")
    print("="*60)

    total_frames = 0
    pbar = tqdm(total=num_steps, desc="Training steps")

    for step_idx, tensordict_data in enumerate(collector):
        if step_idx >= num_steps:
            break

        print(f"\nStep {step_idx + 1}/{num_steps}")
        print(f"  Collected {tensordict_data.numel()} frames")

        # Compute advantages
        with torch.no_grad():
            advantage_module(tensordict_data)
        print(f"  ✓ Computed advantages")

        # Flatten for replay buffer
        data_view = tensordict_data.reshape(-1)

        # Train for multiple epochs
        epoch_losses = []
        for epoch in range(num_epochs):
            replay_buffer.empty()
            replay_buffer.extend(data_view)

            minibatch_losses = []
            for mb_idx in range(num_minibatches):
                subdata = replay_buffer.sample()

                # Compute loss
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Backprop
                optimizer.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optimizer.step()

                minibatch_losses.append(loss_value.item())

            avg_epoch_loss = np.mean(minibatch_losses)
            epoch_losses.append(avg_epoch_loss)

        avg_loss = np.mean(epoch_losses)
        print(f"  ✓ Trained for {num_epochs} epochs")
        print(f"  Average loss: {avg_loss:.4f}")

        # Get rewards if available
        if "reward" in tensordict_data.keys():
            total_reward = tensordict_data["reward"].sum().item()
            mean_reward = tensordict_data["reward"].mean().item()
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Mean reward: {mean_reward:.4f}")

        total_frames += tensordict_data.numel()
        pbar.update(1)

    pbar.close()

    # Cleanup
    collector.shutdown()

    print()
    print("="*60)
    print("DEBUG TRAINING COMPLETED")
    print("="*60)
    print(f"Total frames collected: {total_frames}")
    print()
    print("If this completed successfully, you can run full training with:")
    print(f"  python scripts/train_torchrl.py --dir={log_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Debug training for TorchRL multi-agent PPO')
    parser.add_argument('--dir', default='logs/L96_1', help='Log directory')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda:0)')
    parser.add_argument('--steps', type=int, default=5, help='Number of training steps')
    args = parser.parse_args()

    log_dir = os.path.abspath(args.dir)

    # Check if directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        print(f"Please generate data first:")
        print(f"  python scripts/generate_data.py --dir={args.dir}")
        sys.exit(1)

    # Check required files
    required_files = [
        os.path.join(log_dir, "config.py"),
        os.path.join(log_dir, "precomputed_paths"),
        os.path.join(log_dir, "norm_dict.json"),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease generate data first:")
        print(f"  python scripts/generate_data.py --dir={args.dir}")
        sys.exit(1)

    try:
        debug_train(log_dir, args.device, args.steps)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
