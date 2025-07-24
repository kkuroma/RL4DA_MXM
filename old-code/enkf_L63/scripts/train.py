import os
from os.path import join as osj
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from lorenz63 import L63
from eakf import eakf_solver
from helper import *
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces, Env
from stable_baselines3.common.callbacks import EvalCallback

device = torch.device("cpu")
sol = np.load(osj(os.getcwd(), "data", "example.npy"), allow_pickle=True).item()

# 1) Define a custom MLP extractor using the ensemble strategy
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 8):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, features_dim),
            nn.ReLU(), 
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
    
class IdentityMLPExtractor(nn.Module):
    def __init__(self, features_dim: int):
        super(IdentityMLPExtractor, self).__init__()
        self.features_dim = features_dim
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim
    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)
    def forward_actor(self, features):
        return features
    def forward_critic(self, features):
        return features
    
class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self,observation_space,action_space,lr_schedule,*args,**kwargs,):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **{"features_extractor_class": FeatureExtractor},
        )
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = IdentityMLPExtractor(self.features_dim)

# 2) Define a custom environment
class L63RLAgent(Env):
    def __init__(self, solution_data, thresh=50):
        super(L63RLAgent, self).__init__()
        self.solution_data = solution_data
        self.num_ensembles = solution_data["analysis_states"].shape[-1]
        self.timesteps = solution_data["true_states"].shape[0]
        self.idx = 0
        self.current_step = 0
        self.ensemble_number = 0
        
        # Define observation and action spaces
        vec_size = solution_data["true_states"].shape[1]
        self.observation_space = spaces.Box(low=-thresh, high=thresh, shape=(vec_size*3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-thresh, high=thresh, shape=(vec_size,), dtype=np.float32)
    
    def step(self, action):
        analysis = self.solution_data["analysis_states"][self.idx,:,self.ensemble_number].flatten()
        # Compute reward as negative RMSE
        rmse = np.sqrt(np.mean((analysis - action) ** 2))
        reward = -rmse
        # Update step index and get new observation
        self.current_step += 1
        obs = self._get_obs()
        done = False
        if (self.ensemble_number==0) and (self.idx==0):
            done = True
        return obs, reward, done, done, {}
    
    def reset(self, seed=None, options=None):
        self.idx = 0
        self.current_step = 0
        self.ensemble_number = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        self.idx = (self.idx+1)%self.timesteps
        if self.idx == 0:
            self.ensemble_number = (self.ensemble_number+1)%self.num_ensembles
        obs = np.concatenate([
            self.solution_data["previous_analysis"][self.idx,:,self.ensemble_number],
            self.solution_data["background_states"][self.idx,:,self.ensemble_number],
            self.solution_data["derivatives"][self.idx,:,self.ensemble_number]
        ], axis=0).flatten() # shape = (9,)
        return obs.astype(np.float32)
    
# 3) Train the PPO model
if __name__=="__main__":
    total_timesteps = 10000000
    eval_freq = 20000

    eval_callback = EvalCallback(
        L63RLAgent(sol), 
        best_model_save_path=osj(os.getcwd(), "logs"),
        log_path=osj(os.getcwd(), "logs"), 
        eval_freq=eval_freq, 
        deterministic=True,
        render=False
    )

    model = PPO(
        CustomMLPPolicy,
        L63RLAgent(sol),
        verbose=1,
        device=device,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    model.save("lorenz63")