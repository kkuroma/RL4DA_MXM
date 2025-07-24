import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import numpy as np
from typing import Tuple
import gymnasium as gym

class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Simple MLP feature extractor for EAKF environment.
    Expects flattened input and processes it through fully connected layers.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, 
                 hidden_dims: Tuple[int, ...] = (512, 256)):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimension from observation space
        if isinstance(observation_space, gym.spaces.Box):
            input_dim = int(np.prod(observation_space.shape))
        else:
            raise ValueError("Only Box observation spaces are supported")
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer to features_dim
        layers.append([nn.Linear(prev_dim, features_dim), nn.ReLU()])
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Flatten if needed (batch_size, *) -> (batch_size, -1)
        if len(observations.shape) > 2:
            observations = observations.flatten(start_dim=1)
        return self.mlp(observations)

class MLPActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCritic policy using MLP feature extractor for EAKF.
    """
    
    def __init__(self, *args, features_extractor_kwargs=None, **kwargs):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        super().__init__(
            *args,
            features_extractor_class=MLPFeaturesExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )

class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for sequential data in EAKF environment.
    Processes time series of ensemble states and observations.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 lstm_hidden_size: int = 128, num_lstm_layers: int = 2):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimension
        if isinstance(observation_space, gym.spaces.Box):
            # Assume last dimension is the feature dimension for each timestep
            self.seq_len = observation_space.shape[0] if len(observation_space.shape) > 1 else 1
            self.input_size = observation_space.shape[-1] if len(observation_space.shape) > 1 else observation_space.shape[0]
        else:
            raise ValueError("Only Box observation spaces are supported")
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Reshape for LSTM if needed
        if len(observations.shape) == 2:
            # (batch_size, seq_len * features) -> (batch_size, seq_len, features)
            observations = observations.view(batch_size, self.seq_len, self.input_size)
        elif len(observations.shape) == 3:
            # Already in correct shape (batch_size, seq_len, features)
            pass
        else:
            # Flatten and reshape
            observations = observations.flatten(start_dim=1)
            observations = observations.view(batch_size, self.seq_len, self.input_size)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(observations)
        
        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        
        # Project to features_dim
        features = self.output_projection(last_output)
        
        return features

class LSTMActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCritic policy using LSTM feature extractor for EAKF.
    """
    
    def __init__(self, *args, features_extractor_kwargs=None, **kwargs):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        super().__init__(
            *args,
            features_extractor_class=LSTMFeaturesExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )

def create_mlp_agent(env, learning_rate: float = 3e-4, **kwargs):
    """
    Create a PPO agent with MLP policy for EAKF environment.
    
    Args:
        env: EAKF gymnasium environment
        learning_rate: Learning rate for the optimizer
        **kwargs: Additional arguments for PPO
        
    Returns:
        PPO agent with MLP policy
    """
    return PPO(
        MLPActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        verbose=1,
        **kwargs
    )

def create_lstm_agent(env, learning_rate: float = 3e-4, **kwargs):
    """
    Create a RecurrentPPO agent with LSTM policy for EAKF environment.
    
    Args:
        env: EAKF gymnasium environment
        learning_rate: Learning rate for the optimizer
        **kwargs: Additional arguments for RecurrentPPO
        
    Returns:
        RecurrentPPO agent with LSTM policy
    """
    return RecurrentPPO(
        "MlpLstmPolicy",  # Built-in LSTM policy
        env,
        learning_rate=learning_rate,
        verbose=1,
        **kwargs
    )

def create_custom_lstm_agent(env, learning_rate: float = 3e-4, 
                           lstm_hidden_size: int = 128, 
                           num_lstm_layers: int = 2, **kwargs):
    """
    Create a PPO agent with custom LSTM feature extractor for EAKF environment.
    
    Args:
        env: EAKF gymnasium environment
        learning_rate: Learning rate for the optimizer
        lstm_hidden_size: Hidden size of LSTM layers
        num_lstm_layers: Number of LSTM layers
        **kwargs: Additional arguments for PPO
        
    Returns:
        PPO agent with custom LSTM policy
    """
    policy_kwargs = {
        "features_extractor_kwargs": {
            "lstm_hidden_size": lstm_hidden_size,
            "num_lstm_layers": num_lstm_layers
        }
    }
    
    return PPO(
        LSTMActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **kwargs
    )