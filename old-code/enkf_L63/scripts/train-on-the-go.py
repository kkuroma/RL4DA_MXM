import os
from os.path import join as osj
import numpy as np
from helper import *
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces, Env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

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
    def __init__(self, solution_data, thresh=50, allowed_ens=[0]):
        super(L63RLAgent, self).__init__()
        self.solution_data = solution_data
        self.num_ensembles = solution_data["analysis_states"].shape[-1]
        self.num_allowed_ensembles = len(allowed_ens)
        self.timesteps = solution_data["true_states"].shape[0]
        self.idx = 0
        self.current_analysis = None
        self.current_step = 0
        self.ensemble_idx = 0
        self.allowed_ens = allowed_ens
        
        # Define observation and action spaces
        vec_size = solution_data["true_states"].shape[1]
        self.observation_space = spaces.Box(low=-thresh, high=thresh, shape=(vec_size*3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-thresh, high=thresh, shape=(vec_size,), dtype=np.float32)
    
    def step(self, action):
        analysis = self.solution_data["analysis_states"][self.idx,:,self.allowed_ens[self.ensemble_idx]].flatten()
        # Compute reward as negative RMSE
        rmse = np.sqrt(np.mean((analysis - action) ** 2))
        reward = -rmse
        # Update step index and get new observation
        self.current_step += 1
        obs = self._get_obs()
        done = False
        if (self.ensemble_idx==0) and (self.idx==0):
            done = True
        # Set action
        self.analysis = action.flatten()
        return obs, reward, done, done, {}
    
    def reset(self, seed=None, options=None):
        self.idx = 0
        self.current_step = 0
        self.ensemble_idx = 0
        self.current_analysis = None
        return self._get_obs(), {}
    
    def _get_derivs(self, state):
        sigma, rho, beta = 10, 28, 8/3
        x = state[0]
        y = state[1]
        z = state[2]
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return np.array([dxdt, dydt, dzdt])
    
    def _get_obs(self):
        self.idx = (self.idx+1)%self.timesteps
        # if start of run:
        if self.idx == 0:
            self.ensemble_idx = (self.ensemble_idx+1)%self.num_allowed_ensembles
            self.current_analysis = None
        if self.current_analysis is None:
            self.current_analysis = self.solution_data["previous_analysis"][self.idx,:,self.allowed_ens[self.ensemble_idx]]
        derivatives = self._get_derivs(self.current_analysis)
        obs = np.concatenate([
            self.current_analysis.flatten(),
            self.solution_data["background_states"][self.idx,:,self.allowed_ens[self.ensemble_idx]].flatten(),
            derivatives.flatten()
        ], axis=0).flatten() # shape = (9,)
        return obs.astype(np.float32)
    
# 3) Train the PPO model
if __name__=="__main__":
    total_timesteps = 10000000
    eval_freq = 20000

    eval_callback = EvalCallback(
        Monitor(L63RLAgent(sol, allowed_ens=list(range(20)))), 
        best_model_save_path=osj(os.getcwd(), "logs"),
        log_path=osj(os.getcwd(), "logs"), 
        eval_freq=eval_freq, 
        deterministic=True,
        render=False
    )
    
    config = {
        "policy_type": "PPO-shared",
        "total_timesteps": total_timesteps,
        "env_name": "L63-ENKF",
    }
    run = wandb.init(
        entity="kkuroma",
        project="rl4da-mxm",
        config=config,
    )
    
    wandb_callback = WandbCallback(
        verbose=2,
    )
    
    model = PPO(
        CustomMLPPolicy,
        Monitor(L63RLAgent(sol, allowed_ens=list(range(20)))),
        verbose=1,
        device=device,
        tensorboard_log=f"runs/{run.id}"
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback,wandb_callback],
        progress_bar=True
    )
    model.save("lorenz63")