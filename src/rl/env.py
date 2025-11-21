import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import os
import json
import importlib.util


class MultiAgentEnkfEnvironment(gym.Env):
    """
    Multi-Agent RL Environment for ENKF

    Each of N_ens agents controls one ensemble member (N-dimensional action).
    Agents receive individual observations and rewards based on their ensemble member's RMSE.

    This design reduces action space from (N * N_ens) to N per agent,
    making the learning problem more tractable for large ensembles.
    """

    def __init__(self, logs_dir_path: str, is_eval: bool = False):
        self.logs_dir_path = logs_dir_path
        self.is_eval = is_eval

        # load parameters
        self._load_config()
        self._load_parameters()

        # data and normalization
        self.data_paths = sorted([os.path.join(self.logs_dir_path, "precomputed_paths", name)
                                 for name in os.listdir(os.path.join(self.logs_dir_path, "precomputed_paths"))])
        if is_eval:
            self.data_paths = [self.data_paths[0]]
        else:
            self.data_paths.pop(0)
        self.data_path_cache = {}
        self.norm_dict = json.load(open(os.path.join(self.logs_dir_path, "norm_dict.json"), "r"))

        # episode state
        self.timestep = 0
        self.episode_halted = False
        self.halt_reason = ""
        self.curriculum_track = {
            'best_reward': -1e10,
            'current_reward': 0,
            'current_episode_length': 0,
            'num_passed_instaces': 0
        }
        self.active_paths = [0]
        self.current_path_idx = 0
        self.xa_prev = None
        self.last_path_used = 0
        self.current_path = self._get_current_path()

        # Multi-agent specific: agent IDs
        self.agent_ids = [f"agent_{i}" for i in range(self.N_ens)]

        # setup env
        self._setup_env()

    def _get_current_path(self):
        self.last_path_used += 1
        # cache hit
        if self.active_paths[self.current_path_idx] in self.data_path_cache:
            self.data_path_cache[self.active_paths[self.current_path_idx]]['last_used'] = self.last_path_used
            return self.data_path_cache[self.active_paths[self.current_path_idx]]['data']
        # cache full - use LRU
        if len(self.data_path_cache) >= self.config['cache_size']:
            least_recent = min(self.data_path_cache, key=lambda k: self.data_path_cache[k]['last_used'])
            del self.data_path_cache[least_recent]
        data = np.load(self.data_paths[self.active_paths[self.current_path_idx]], allow_pickle=True).item()
        self.data_path_cache[self.active_paths[self.current_path_idx]] = {'data': data, 'last_used': self.last_path_used}
        return data

    def _load_config(self):
        config_path = os.path.join(self.logs_dir_path, "config.py")
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        self.config = config_module.config

    def _load_parameters(self):
        self.N = self.config['params']['N']
        self.N_ens = self.config['num_ensembles']
        self.dtda = self.config['dtda']
        self.steps_per_oda = int(self.config['oda'] / self.dtda)
        self.episode_length = self.config.get('max_episode_length', 250)
        # Each agent observes: ALL xa_prev (N*N_ens), ALL xb (N*N_ens), ALL dxa (N*N_ens), observation (N)
        # Total: 3*N*N_ens + N
        self.obs_dim_per_agent = 3 * self.N * self.N_ens + self.N
        self.model_params = self.config['params']
        self.model_class = self.config['model_class']

    def _setup_env(self):
        """
        For multi-agent: each agent has its own observation and action space.
        Observation: [xa_prev (flattened N*N_ens), xb (flattened N*N_ens), dxa (flattened N*N_ens), xo (N)]
        Action: xa_i (N-dimensional vector for this ensemble member)

        All agents see the ENTIRE state space (centralized observations, decentralized actions).
        """
        # Define the observation and action space for a single agent
        single_obs_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.obs_dim_per_agent,),
            dtype=np.float32
        )
        single_action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.N,),  # Each agent controls N dimensions (one ensemble member)
            dtype=np.float32
        )

        # For RLlib multi-agent, we need Dict spaces
        # However, we'll keep single spaces as attributes for convenience
        # RLlib will use the spaces from the policy config
        self.observation_space = single_obs_space
        self.action_space = single_action_space

    def _get_background(self, xa):
        '''get normalized xb given normalized xa'''
        xa = xa * self.norm_dict["analysis_states"]
        xb = np.zeros((self.N, self.N_ens))
        for l in range(self.N_ens):
            model = self.model_class(self.model_params, self.dtda, False)
            model.initialize(xa[:, l])
            for _ in range(self.steps_per_oda):
                xb[:, l], _ = model.step()
        return xb / self.norm_dict['background_states']

    def _get_derivatives(self, xa):
        '''get normalized dxa given normalized xa'''
        xa = xa * self.norm_dict["analysis_states"]
        dxa = np.zeros((self.N, self.N_ens))
        for l in range(self.N_ens):
            model = self.model_class(self.model_params, self.dtda, False)
            dxa[:, l] = model.derivatives(0, xa[:, l])
        return dxa / self.norm_dict['derivatives']

    def reset(self, seed=None, **kwargs):
        '''Reset the environment and return observations for all agents'''
        self.timestep = 0
        if self.curriculum_track['current_reward'] > self.curriculum_track['best_reward']:
            self.curriculum_track['best_reward'] = self.curriculum_track['current_reward']
            if self.curriculum_track['current_episode_length'] >= 0.9 * self.episode_length:
                self.curriculum_track['num_passed_instaces'] += 1
        if self.curriculum_track['num_passed_instaces'] >= 100 / len(self.active_paths):
            # advance curriculum
            self.active_paths.append(len(self.active_paths))
            self.curriculum_track['best_reward'] = -1e10
            self.curriculum_track['num_passed_instaces'] = 0
        self.current_path_idx = (self.current_path_idx + 1) % len(self.active_paths)
        self.current_path = self._get_current_path()
        self.episode_halted = False
        self.halt_reason = ""
        self.curriculum_track['current_reward'] = 0
        self.curriculum_track['current_episode_length'] = 0

        obs_dict = self._get_observations()
        info_dict = {agent_id: {} for agent_id in self.agent_ids}

        return obs_dict, info_dict

    def _get_observations(self):
        '''Get observations for all agents'''
        # Initialize xa_prev if needed
        if True:  # self.xa_prev is None:
            self.xa_prev = self.current_path["previous_analysis"][self.timestep] / self.norm_dict["analysis_states"]

        xb = self._get_background(self.xa_prev)
        xo = self.current_path["observations"][self.timestep] / self.norm_dict["observations"]
        dxa = self._get_derivatives(self.xa_prev)

        # Check for extreme values
        full_obs = np.concatenate([self.xa_prev, xb, dxa, xo.reshape(self.N, 1)], axis=1)
        if np.any(np.isnan(full_obs)) or np.any(np.isinf(full_obs)):
            self.episode_halted = True
            self.halt_reason = "Extreme values found in observations"

        # Flatten all ensemble data for shared observation
        # xa_prev: (N, N_ens) -> flatten to (N*N_ens,)
        # xb: (N, N_ens) -> flatten to (N*N_ens,)
        # dxa: (N, N_ens) -> flatten to (N*N_ens,)
        # xo: (N,)
        xa_prev_flat = self.xa_prev.flatten(order='F')  # Flatten column-wise
        xb_flat = xb.flatten(order='F')
        dxa_flat = dxa.flatten(order='F')

        # Print observation component ranges every 100 steps
        if self.timestep % 100 == 0:
            print(f"\n=== Observation Component Ranges (timestep {self.timestep}) ===")
            print(f"xa_prev: [{xa_prev_flat.min():.4f}, {xa_prev_flat.max():.4f}], mean={xa_prev_flat.mean():.4f}, std={xa_prev_flat.std():.4f}")
            print(f"xb:      [{xb_flat.min():.4f}, {xb_flat.max():.4f}], mean={xb_flat.mean():.4f}, std={xb_flat.std():.4f}")
            print(f"dxa:     [{dxa_flat.min():.4f}, {dxa_flat.max():.4f}], mean={dxa_flat.mean():.4f}, std={dxa_flat.std():.4f}")
            print(f"xo:      [{xo.min():.4f}, {xo.max():.4f}], mean={xo.mean():.4f}, std={xo.std():.4f}")

        # Create shared observation: [xa_prev_flat, xb_flat, dxa_flat, xo]
        # Total: 3*N*N_ens + N
        shared_obs = np.concatenate([
            xa_prev_flat,  # N*N_ens dims
            xb_flat,       # N*N_ens dims
            dxa_flat,      # N*N_ens dims
            xo             # N dims
        ]).astype(np.float32)

        # All agents receive the same observation (centralized observations)
        obs_dict = {}
        for agent_id in self.agent_ids:
            obs_dict[agent_id] = shared_obs.copy()

        return obs_dict

    def step(self, action_dict: Dict[str, np.ndarray]):
        """
        Step function for multi-agent environment.

        Args:
            action_dict: Dict mapping agent_id -> action (N-dimensional array)

        Returns:
            obs_dict: Dict mapping agent_id -> observation
            reward_dict: Dict mapping agent_id -> reward
            terminated_dict: Dict mapping agent_id -> terminated flag
            truncated_dict: Dict mapping agent_id -> truncated flag
            info_dict: Dict mapping agent_id -> info dict
        """
        # Check if episode is already done (safety check to prevent out of bounds access)
        if self.timestep >= self.episode_length:
            obs_dict = {agent_id: np.zeros(self.obs_dim_per_agent, dtype=np.float32) for agent_id in self.agent_ids}
            reward_dict = {agent_id: 0.0 for agent_id in self.agent_ids}
            terminated_dict = {agent_id: True for agent_id in self.agent_ids}
            terminated_dict["__all__"] = True
            truncated_dict = {agent_id: False for agent_id in self.agent_ids}
            truncated_dict["__all__"] = False
            info_dict = {agent_id: {} for agent_id in self.agent_ids}
            return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

        xa_true = self.current_path["analysis_states"][self.timestep].astype(np.float32) / self.norm_dict["analysis_states"]

        # Handle episode halt
        if self.episode_halted:
            obs_dict = self._get_observations()
            reward_dict = {agent_id: -4.0 for agent_id in self.agent_ids}
            terminated_dict = {agent_id: True for agent_id in self.agent_ids}
            terminated_dict["__all__"] = True
            truncated_dict = {agent_id: False for agent_id in self.agent_ids}
            truncated_dict["__all__"] = False
            info_dict = {agent_id: {'halt_reason': self.halt_reason} for agent_id in self.agent_ids}
            return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

        # Reconstruct full action from individual agent actions
        xa_predicted = np.zeros((self.N, self.N_ens), dtype=np.float32)
        for i, agent_id in enumerate(self.agent_ids):
            if self.timestep == 0:
                # At timestep 0, use ground truth
                xa_predicted[:, i] = xa_true[:, i]
            else:
                # Use agent's action
                xa_predicted[:, i] = action_dict[agent_id]

        # Calculate individual rewards for each agent based on its ensemble member's RMSE
        reward_dict = {}
        total_reward = 0.0
        for i, agent_id in enumerate(self.agent_ids):
            rmse_i = np.sqrt(np.mean((xa_true[:, i] - xa_predicted[:, i]) ** 2))
            reward_i = 1.0 - rmse_i
            reward_dict[agent_id] = float(reward_i)
            total_reward += reward_i

        # Average reward across all agents
        avg_reward = total_reward / self.N_ens
        self.curriculum_track['current_reward'] += avg_reward
        self.curriculum_track['current_episode_length'] += 1

        # Update state
        self.xa_prev = xa_predicted
        self.timestep += 1

        # Check if episode is done
        done = self.timestep >= self.episode_length

        if done:
            # Episode is done, don't try to get next observations (would be out of bounds)
            # Return dummy observations
            next_obs_dict = {agent_id: np.zeros(self.obs_dim_per_agent, dtype=np.float32) for agent_id in self.agent_ids}
        else:
            # Get next observations
            next_obs_dict = self._get_observations()

        terminated_dict = {agent_id: done for agent_id in self.agent_ids}
        terminated_dict["__all__"] = done
        truncated_dict = {agent_id: False for agent_id in self.agent_ids}
        truncated_dict["__all__"] = False
        info_dict = {agent_id: {} for agent_id in self.agent_ids}

        return next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict


def create_multiagent_enkf_env(logs_dir_path: str, eval_mode: bool = False):
    """Factory function to create Multi-Agent ENKF RL environment"""
    return MultiAgentEnkfEnvironment(logs_dir_path, is_eval=eval_mode)