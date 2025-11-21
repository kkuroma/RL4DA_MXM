"""
TorchRL-compatible Multi-Agent ENKF Environment

This wrapper converts the Gymnasium-based MultiAgentEnkfEnvironment
to TorchRL's TensorDict-based API for use with TorchRL's PPO implementation.
"""

import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import CompositeSpec, Bounded, Unbounded, Categorical
from torchrl.envs import EnvBase
from typing import Optional

from rl.env import MultiAgentEnkfEnvironment


class TorchRLMultiAgentEnkfEnv(EnvBase):
    """
    TorchRL wrapper for Multi-Agent ENKF Environment.

    Converts Gymnasium dict-based API to TorchRL's TensorDict-based API.
    Handles batching dimension (num_envs=1 for single environment).

    For multi-agent in TorchRL, we use a "shared policy" approach where:
    - All agents share the same policy network
    - Observations, actions, and rewards are stacked along agent dimension
    - Agent index is implicit in the tensor shape
    """

    batch_locked = False  # Allow dynamic batching

    def __init__(
        self,
        logs_dir_path: str,
        is_eval: bool = False,
        device: str = "cpu",
        batch_size: Optional[torch.Size] = None,
    ):
        """
        Initialize TorchRL-compatible environment.

        Args:
            logs_dir_path: Path to logs directory with precomputed data
            is_eval: Whether this is evaluation mode
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size (default: torch.Size([]) for unbatched)
        """
        super().__init__(device=device, batch_size=batch_size or torch.Size([]))

        # Create underlying Gymnasium environment
        self.gym_env = MultiAgentEnkfEnvironment(logs_dir_path, is_eval)

        # Get environment parameters
        self.num_agents = self.gym_env.N_ens
        self.obs_dim_per_agent = self.gym_env.obs_dim_per_agent
        self.action_dim_per_agent = self.gym_env.N
        self.agent_ids = self.gym_env.agent_ids

        # Define specs
        self._make_specs()

        # Internal state
        self._current_obs = None
        self._step_count = 0

    def _make_specs(self):
        """Create TensorDict specs for observations, actions, rewards, and done flags."""

        # Observation spec: (num_agents, obs_dim_per_agent)
        observation_spec = Bounded(
            low=-5.0,
            high=5.0,
            shape=(self.num_agents, self.obs_dim_per_agent),
            dtype=torch.float32,
            device=self.device,
        )

        # Action spec: (num_agents, action_dim_per_agent)
        action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, self.action_dim_per_agent),
            dtype=torch.float32,
            device=self.device,
        )

        # Reward spec: (num_agents, 1) - individual rewards per agent
        reward_spec = Bounded(
            low=-4.0,
            high=1.0,
            shape=(self.num_agents, 1),
            dtype=torch.float32,
            device=self.device,
        )

        # Done spec: (num_agents, 1) - broadcast to match reward shape
        done_spec = Categorical(
            n=2,
            shape=(self.num_agents, 1),
            dtype=torch.bool,
            device=self.device,
        )

        # Create composite specs
        self.observation_spec = CompositeSpec(
            observation=observation_spec,
            shape=self.batch_size,
        )

        self.action_spec = CompositeSpec(
            action=action_spec,
            shape=self.batch_size,
        )

        self.reward_spec = CompositeSpec(
            reward=reward_spec,
            shape=self.batch_size,
        )

        self.done_spec = CompositeSpec(
            done=done_spec,
            terminated=done_spec.clone(),
            shape=self.batch_size,
        )

        # Full spec combines all
        self.full_done_spec = self.done_spec
        self.full_action_spec = self.action_spec
        self.full_observation_spec = self.observation_spec
        self.full_reward_spec = self.reward_spec

    def _obs_dict_to_tensor(self, obs_dict: dict) -> torch.Tensor:
        """
        Convert Gymnasium observation dict to TorchRL tensor.

        Args:
            obs_dict: Dict mapping agent_id -> observation array

        Returns:
            torch.Tensor: Stacked observations of shape (num_agents, obs_dim)
        """
        obs_list = [obs_dict[agent_id] for agent_id in self.agent_ids]
        obs_array = np.stack(obs_list, axis=0)  # (num_agents, obs_dim)
        return torch.from_numpy(obs_array).float().to(self.device)

    def _action_tensor_to_dict(self, action_tensor: torch.Tensor) -> dict:
        """
        Convert TorchRL action tensor to Gymnasium action dict.

        Args:
            action_tensor: Tensor of shape (num_agents, action_dim)

        Returns:
            dict: Dict mapping agent_id -> action array
        """
        action_array = action_tensor.cpu().numpy()
        action_dict = {
            agent_id: action_array[i]
            for i, agent_id in enumerate(self.agent_ids)
        }
        return action_dict

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        """
        Reset the environment.

        Args:
            tensordict: Optional tensordict with reset parameters (e.g., seed)

        Returns:
            TensorDict: Initial observation tensordict
        """
        # Extract seed if provided
        seed = None
        if tensordict is not None and "seed" in tensordict.keys():
            seed = tensordict["seed"].item()

        # Reset Gymnasium environment
        obs_dict, info_dict = self.gym_env.reset(seed=seed)

        # Convert to TensorDict
        obs_tensor = self._obs_dict_to_tensor(obs_dict)

        self._current_obs = obs_tensor
        self._step_count = 0

        # Create output tensordict
        out_tensordict = TensorDict(
            {
                "observation": obs_tensor,
                "done": torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device),
                "terminated": torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return out_tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Step the environment.

        Args:
            tensordict: TensorDict containing 'action' key

        Returns:
            TensorDict: Next observation, reward, done, and terminated flags
        """
        # Extract action from tensordict
        action_tensor = tensordict["action"]

        # Convert to dict format for Gymnasium env
        action_dict = self._action_tensor_to_dict(action_tensor)

        # Step Gymnasium environment
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.gym_env.step(action_dict)

        # Convert observations to tensor
        obs_tensor = self._obs_dict_to_tensor(obs_dict)

        # Convert rewards to tensor: (num_agents, 1)
        reward_list = [reward_dict[agent_id] for agent_id in self.agent_ids]
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Get done flags (all agents done together, broadcast to match reward shape)
        done = terminated_dict["__all__"] or truncated_dict["__all__"]
        terminated = terminated_dict["__all__"]

        done_tensor = torch.full((self.num_agents, 1), done, dtype=torch.bool, device=self.device)
        terminated_tensor = torch.full((self.num_agents, 1), terminated, dtype=torch.bool, device=self.device)

        self._current_obs = obs_tensor
        self._step_count += 1

        # Create output tensordict
        out_tensordict = TensorDict(
            {
                "observation": obs_tensor,
                "reward": reward_tensor,
                "done": done_tensor,
                "terminated": terminated_tensor,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return out_tensordict

    def _set_seed(self, seed: Optional[int] = None):
        """Set random seed."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)


def create_torchrl_multiagent_enkf_env(
    logs_dir_path: str,
    eval_mode: bool = False,
    device: str = "cpu",
) -> TorchRLMultiAgentEnkfEnv:
    """
    Factory function to create TorchRL-compatible Multi-Agent ENKF environment.

    Args:
        logs_dir_path: Path to logs directory
        eval_mode: Whether to use evaluation mode
        device: Device ('cpu' or 'cuda')

    Returns:
        TorchRLMultiAgentEnkfEnv: TorchRL-compatible environment
    """
    return TorchRLMultiAgentEnkfEnv(
        logs_dir_path=logs_dir_path,
        is_eval=eval_mode,
        device=device,
    )
