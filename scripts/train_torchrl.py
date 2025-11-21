#!/usr/bin/env python3
"""
Multi-Agent RL Training Script for ENKF using TorchRL

This script implements multi-agent PPO training using TorchRL's native multi-agent support.
Based on: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html

Usage:
    python scripts/train_torchrl.py --dir=logs/L96_1 [options]

Arguments:
    --dir               Log directory (required, e.g., logs/L96_1)
    --num-workers       Number of parallel collectors (default: 4)
    --num-gpus          Number of GPUs to use (0 for CPU, default: 1)
    --total-frames      Total frames to collect (default: 1_000_000)
    --frames-per-batch  Frames per batch (default: 4000)
    --num-epochs        PPO epochs per batch (default: 10)
    --lr                Learning rate (default: from config or 3e-4)
    --seed              Random seed (default: 42)
"""

import os
import sys
import argparse
import importlib.util
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.envs import (
    EnvBase,
    TransformedEnv,
    InitTracker,
    StepCounter,
    check_env_specs,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions import Independent, Normal
from tqdm import tqdm

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from rl.env import MultiAgentEnkfEnvironment


def setup_logging(log_dir):
    """Set up logging to file and console"""
    log_file = os.path.join(log_dir, "run_torchrl.log")

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_global_config():
    """Load global configuration from logs/global_config.py"""
    config_path = os.path.join(project_root, "logs", "global_config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Global config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("global_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


class TorchRLMultiAgentEnv(EnvBase):
    """
    TorchRL-compatible wrapper for MultiAgentEnkfEnvironment.

    Converts dict-based multi-agent outputs to TorchRL's tensor-based format
    with proper batch dimensions for multi-agent learning.

    TensorDict structure:
        - observation: (n_agents, obs_dim)
        - action: (n_agents, action_dim)
        - reward: (n_agents, 1)
        - done: (n_agents, 1)
    """

    batch_locked = False

    def __init__(self, logs_dir_path: str, device: str = "cpu", is_eval: bool = False):
        super().__init__(device=device, batch_size=[])

        # Create the base environment
        self.base_env = MultiAgentEnkfEnvironment(logs_dir_path, is_eval=is_eval)
        self.n_agents = len(self.base_env.agent_ids)
        self.agent_ids = self.base_env.agent_ids

        # Get observation and action dimensions
        obs_dim = self.base_env.observation_space.shape[0]
        action_dim = self.base_env.action_space.shape[0]

        # Define specs with multi-agent dimension
        from torchrl.data import Composite, Unbounded, Binary

        # Observation spec: (n_agents, obs_dim)
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(self.n_agents, obs_dim),
                dtype=torch.float32,
                device=device,
            ),
            shape=[],  # Empty batch size for single environment
        )

        # Action spec: (n_agents, action_dim)
        self.action_spec = Unbounded(
            shape=(self.n_agents, action_dim),
            dtype=torch.float32,
            device=device,
        )

        # Reward spec: (n_agents, 1)
        self.reward_spec = Composite(
            reward=Unbounded(
                shape=(self.n_agents, 1),
                dtype=torch.float32,
                device=device,
            ),
            shape=[],  # Empty batch size for single environment
        )

        # Done spec: (n_agents, 1) - per-agent done flags
        self.done_spec = Composite(
            done=Binary(
                n=1,
                shape=(self.n_agents, 1),
                dtype=torch.bool,
                device=device,
            ),
            shape=[],  # Empty batch size for single environment
        )

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:
        """Reset environment and return initial observations"""
        obs_dict, info_dict = self.base_env.reset()

        # Stack observations from all agents
        obs_list = [obs_dict[agent_id] for agent_id in self.agent_ids]
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)

        # Create TensorDict with proper structure
        # batch_size=[] means single environment, n_agents is part of the tensor shape
        td = TensorDict(
            {
                "observation": obs_tensor,  # shape: (n_agents, obs_dim)
                "done": torch.zeros(self.n_agents, 1, dtype=torch.bool, device=self.device),  # shape: (n_agents, 1)
            },
            batch_size=[],  # Empty batch_size for single environment
            device=self.device,
        )

        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute actions and return next state"""
        # Extract actions and convert to dict format
        actions = tensordict["action"].cpu().numpy()
        action_dict = {agent_id: actions[i] for i, agent_id in enumerate(self.agent_ids)}

        # Step the base environment
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.base_env.step(action_dict)

        # Convert outputs to tensors
        obs_list = [obs_dict[agent_id] for agent_id in self.agent_ids]
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)

        reward_list = [reward_dict[agent_id] for agent_id in self.agent_ids]
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Combine terminated and truncated into done
        done_list = [terminated_dict[agent_id] or truncated_dict[agent_id] for agent_id in self.agent_ids]
        done_tensor = torch.tensor(done_list, dtype=torch.bool, device=self.device).unsqueeze(-1)

        # Create output TensorDict
        # TorchRL automatically handles "next" nesting - just return the new state directly
        # The environment framework will wrap this in "next" for you
        # batch_size=[] means single environment, n_agents is part of the tensor shape
        td = TensorDict(
            {
                "observation": obs_tensor,   # shape: (n_agents, obs_dim) - this becomes next.observation
                "reward": reward_tensor,     # shape: (n_agents, 1) - stays at root
                "done": done_tensor,         # shape: (n_agents, 1) - this becomes next.done
            },
            batch_size=[],  # Empty batch_size for single environment
            device=self.device,
        )

        return td

    def _set_seed(self, seed: int):
        """Set random seed"""
        torch.manual_seed(seed)
        np.random.seed(seed)


def make_env(logs_dir_path: str, device: str, is_eval: bool = False):
    """Create and wrap the environment with necessary transforms"""
    env = TorchRLMultiAgentEnv(logs_dir_path, device=device, is_eval=is_eval)

    # Add transforms for tracking
    env = TransformedEnv(env)
    env.append_transform(InitTracker())  # Track episode initialization
    env.append_transform(StepCounter(max_steps=500))  # Track steps (from config)

    return env


def make_ppo_models(
    env: EnvBase,
    device: str,
    hidden_sizes: tuple = (256, 256),
    activation: nn.Module = nn.Tanh,
) -> tuple:
    """
    Create policy and value networks for multi-agent PPO.

    Following TorchRL multi-agent tutorial approach with parameter sharing.
    Each agent has the same network architecture (parameter sharing).

    Args:
        env: The environment
        device: Device to place models on
        hidden_sizes: Hidden layer sizes for MLPs
        activation: Activation function

    Returns:
        policy_module: The policy network
        value_module: The value network
    """
    n_agents = env.n_agents
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]

    # Build policy network (shared parameters across agents)
    # Input: (n_agents, obs_dim) -> Output: (n_agents, 2*action_dim) for mean and log_std
    policy_net = nn.Sequential(
        nn.Linear(obs_dim, hidden_sizes[0]),
        activation(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        activation(),
        nn.Linear(hidden_sizes[1], 2 * action_dim),
    ).to(device)

    # Wrap policy in TensorDictModule
    policy_module = TensorDictModule(
        policy_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    # Split output into loc and scale, and create distribution
    class SplitMeanStd(nn.Module):
        """Split concatenated mean and log_std into separate tensors"""
        def __init__(self, action_dim):
            super().__init__()
            self.action_dim = action_dim

        def forward(self, loc_scale):
            loc, scale = loc_scale.chunk(2, dim=-1)
            scale = torch.nn.functional.softplus(scale) + 1e-4  # Ensure positive std
            return loc, scale

    split_module = TensorDictModule(
        SplitMeanStd(action_dim),
        in_keys=[("loc", "scale")],  # This won't work, need to adjust
        out_keys=["loc", "scale"],
    )

    # Actually, let's rebuild this more cleanly
    class PolicyNetwork(nn.Module):
        """Policy network that outputs mean and std"""
        def __init__(self, obs_dim, action_dim, hidden_sizes, activation):
            super().__init__()
            self.action_dim = action_dim
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_sizes[0]),
                activation(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                activation(),
                nn.Linear(hidden_sizes[1], 2 * action_dim),
            )

        def forward(self, observation):
            out = self.net(observation)
            loc, scale = out.chunk(2, dim=-1)
            scale = torch.nn.functional.softplus(scale) + 1e-4
            return loc, scale

    policy_net = PolicyNetwork(obs_dim, action_dim, hidden_sizes, activation).to(device)

    policy_module = TensorDictModule(
        policy_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    # Create probabilistic actor with TanhNormal distribution
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -1.0,
            "high": 1.0,
            "tanh_loc": False,
        },
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )

    # Build value network (shared parameters across agents)
    # Input: (n_agents, obs_dim) -> Output: (n_agents, 1)
    value_net = nn.Sequential(
        nn.Linear(obs_dim, hidden_sizes[0]),
        activation(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        activation(),
        nn.Linear(hidden_sizes[1], 1),
    ).to(device)

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return policy_module, value_module


def main():
    parser = argparse.ArgumentParser(description='Train multi-agent RL for ENKF with TorchRL')
    parser.add_argument('--dir', required=True, help='Log directory (e.g., logs/L96_1)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel collectors')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs (0 for CPU)')
    parser.add_argument('--total-frames', type=int, default=1_000_000, help='Total frames to collect')
    parser.add_argument('--frames-per-batch', type=int, default=4000, help='Frames per batch')
    parser.add_argument('--num-epochs', type=int, default=10, help='PPO epochs per batch')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: from config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Convert to absolute path
    log_dir = Path(os.path.abspath(args.dir))

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("="*60)
    logger.info(f"TorchRL Multi-Agent Training for {log_dir}")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("="*60)

    # Load configurations
    global_config = load_global_config()
    rl_config = global_config.get("rl_multiagent", global_config.get("rl", {}))

    # Set device
    if args.num_gpus > 0 and torch.cuda.is_available():
        device = "cuda:0"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Training hyperparameters (from config or args)
    total_frames = args.total_frames
    frames_per_batch = args.frames_per_batch
    num_epochs = args.num_epochs
    num_minibatches = rl_config.get("sgd_minibatch_size", 128) * num_epochs // frames_per_batch
    if num_minibatches < 1:
        num_minibatches = 8
    lr = args.lr if args.lr is not None else rl_config.get("learning_rate", 3e-4)

    # PPO parameters (from config)
    clip_epsilon = rl_config.get("clip_range", 0.2)
    gamma = rl_config.get("gamma", 0.99)
    lmbda = rl_config.get("gae_lambda", 0.95)
    entropy_coef = rl_config.get("ent_coef", 0.01)
    critic_coef = rl_config.get("vf_coef", 0.5)

    logger.info(f"Training parameters:")
    logger.info(f"  Total frames: {total_frames:,}")
    logger.info(f"  Frames per batch: {frames_per_batch:,}")
    logger.info(f"  Num epochs: {num_epochs}")
    logger.info(f"  Num minibatches: {num_minibatches}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Clip epsilon: {clip_epsilon}")
    logger.info(f"  Gamma: {gamma}")
    logger.info(f"  Lambda (GAE): {lmbda}")
    logger.info(f"  Entropy coefficient: {entropy_coef}")
    logger.info(f"  Critic coefficient: {critic_coef}")

    # Create directories
    checkpoint_dir = log_dir / "checkpoints_torchrl"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create environment
    logger.info("Creating environment...")
    env = make_env(str(log_dir), device=device, is_eval=False)

    logger.info(f"Environment created:")
    logger.info(f"  Number of agents: {env.n_agents}")
    logger.info(f"  Observation shape: {env.observation_spec['observation'].shape}")
    logger.info(f"  Action shape: {env.action_spec.shape}")

    # Check environment specs
    try:
        check_env_specs(env)
        logger.info("Environment specs check: PASSED")
    except Exception as e:
        logger.warning(f"Environment specs check failed: {e}")

    # Create policy and value networks
    logger.info("Creating policy and value networks...")
    policy_module, value_module = make_ppo_models(env, device)

    # Count parameters
    policy_params = sum(p.numel() for p in policy_module.parameters())
    value_params = sum(p.numel() for p in value_module.parameters())
    logger.info(f"  Policy parameters: {policy_params:,}")
    logger.info(f"  Value parameters: {value_params:,}")

    # Create data collector
    logger.info("Creating data collector...")
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,  # No limit on trajectory length
    )

    # Create replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        batch_size=frames_per_batch // num_minibatches,
    )

    # Create advantage module (GAE)
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
    logger.info("="*60)
    logger.info("Starting training...")
    logger.info("="*60)

    total_frames_collected = 0
    pbar = tqdm(total=total_frames, desc="Training")

    for batch_idx, tensordict_data in enumerate(collector):
        # Compute advantages
        with torch.no_grad():
            advantage_module(tensordict_data)

        # Flatten data for sampling
        data_view = tensordict_data.reshape(-1)

        # Train for multiple epochs
        for epoch in range(num_epochs):
            # Clear replay buffer and add new data
            replay_buffer.empty()
            replay_buffer.extend(data_view)

            # Train on minibatches
            for minibatch_idx in range(num_minibatches):
                subdata = replay_buffer.sample()

                # Compute loss
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Backprop and optimize
                optimizer.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optimizer.step()

        # Update progress
        frames_in_batch = tensordict_data.numel()
        total_frames_collected += frames_in_batch
        pbar.update(frames_in_batch)

        # Logging
        if "done" in tensordict_data.keys():
            done_indices = tensordict_data["done"].squeeze(-1)
            if done_indices.any():
                episode_rewards = tensordict_data["reward"][done_indices].sum(dim=0).mean()
                logger.info(
                    f"Batch {batch_idx:4d} | Frames: {total_frames_collected:8,} | "
                    f"Mean Episode Reward: {episode_rewards.item():8.4f} | "
                    f"Loss: {loss_value.item():8.4f}"
                )

        # Save checkpoint periodically
        if batch_idx % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{total_frames_collected}.pt"
            torch.save({
                'batch_idx': batch_idx,
                'total_frames': total_frames_collected,
                'policy_state_dict': policy_module.state_dict(),
                'value_state_dict': value_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    pbar.close()

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'batch_idx': batch_idx,
        'total_frames': total_frames_collected,
        'policy_state_dict': policy_module.state_dict(),
        'value_state_dict': value_module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    logger.info(f"Final model saved: {final_path}")

    # Cleanup
    collector.shutdown()
    logger.info("="*60)
    logger.info(f"Training completed at: {datetime.now()}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
