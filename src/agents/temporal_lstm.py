import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class TemporalLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Temporal LSTM features extractor that respects timestep structure.

    Processes temporal sequences with LSTM while treating same-timestep data simultaneously.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        temporal_window: int = 3,
        lstm_hidden_size: int = 512,
        num_lstm_layers: int = 2,
        features_dim: int = 512,
        **kwargs
    ):
        super().__init__(observation_space, features_dim)

        self.temporal_window = temporal_window
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        # Extract dimensions from observation space
        temporal_seq_dim = observation_space["temporal_sequence"].shape[-1]
        current_state_dim = observation_space["current_state"].shape[-1]

        # LSTM for temporal sequence processing
        self.temporal_lstm = nn.LSTM(
            input_size=temporal_seq_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0.0
        )

        # MLP for current state processing (simultaneous features)
        self.current_state_mlp = nn.Sequential(
            nn.Linear(current_state_dim, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU()
        )

        # Combine temporal and current features
        combined_dim = lstm_hidden_size + lstm_hidden_size // 2
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for name, param in self.temporal_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for LSTM
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        for module in [self.current_state_mlp, self.feature_combiner]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass respecting temporal structure.

        Args:
            observations: Dict with keys:
                - "temporal_sequence": Shape (batch, seq_len, temporal_dim)
                - "current_state": Shape (batch, current_dim)
        """
        temporal_seq = observations["temporal_sequence"]
        current_state = observations["current_state"]

        batch_size = temporal_seq.shape[0]

        # Process temporal sequence with LSTM
        # temporal_seq: (batch, seq_len, temporal_dim)
        lstm_out, (hidden, cell) = self.temporal_lstm(temporal_seq)

        # Use the last hidden state as temporal features
        # hidden: (num_layers, batch, hidden_size) -> take last layer
        temporal_features = hidden[-1]  # (batch, hidden_size)

        # Process current state features (simultaneous at timestep t)
        current_features = self.current_state_mlp(current_state)  # (batch, hidden_size//2)

        # Combine temporal and current features
        combined_features = torch.cat([temporal_features, current_features], dim=1)

        # Final feature processing
        features = self.feature_combiner(combined_features)

        return features


class TemporalLSTMPolicy(nn.Module):
    """
    Complete temporal LSTM policy for ENKF RL.

    Separates temporal modeling (LSTM) from current state processing (MLP).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        temporal_window: int = 3,
        lstm_hidden_size: int = 512,
        num_lstm_layers: int = 2,
        features_dim: int = 512,
        pi_hidden_dims: Tuple[int, ...] = (256, 256),
        vf_hidden_dims: Tuple[int, ...] = (256, 256),
        activation_fn: nn.Module = nn.ReLU,
        log_std_init: float = 0.0,
        **kwargs
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)

        # Features extractor
        self.features_extractor = TemporalLSTMFeaturesExtractor(
            observation_space=observation_space,
            temporal_window=temporal_window,
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
            features_dim=features_dim
        )

        # Policy head (actor)
        pi_layers = []
        input_dim = features_dim
        for hidden_dim in pi_hidden_dims:
            pi_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
            ])
            input_dim = hidden_dim
        pi_layers.append(nn.Linear(input_dim, self.action_dim))
        self.policy_net = nn.Sequential(*pi_layers)

        # Value head (critic)
        vf_layers = []
        input_dim = features_dim
        for hidden_dim in vf_hidden_dims:
            vf_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
            ])
            input_dim = hidden_dim
        vf_layers.append(nn.Linear(input_dim, 1))
        self.value_net = nn.Sequential(*vf_layers)

        # Action distribution parameters
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize policy and value networks."""
        for net in [self.policy_net, self.value_net]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)

        # Last layer of policy net should have smaller weights
        if hasattr(self.policy_net[-1], 'weight'):
            nn.init.orthogonal_(self.policy_net[-1].weight, gain=0.01)

    def extract_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features using temporal LSTM."""
        return self.features_extractor(observations)

    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both policy and value.

        Returns:
            action_mean: Mean actions from policy
            values: State values from critic
        """
        features = self.extract_features(observations)

        action_mean = self.policy_net(features)
        values = self.value_net(features).squeeze(-1)

        return action_mean, values

    def get_action_distribution(self, observations: Dict[str, torch.Tensor]):
        """Get action distribution for sampling."""
        action_mean, _ = self.forward(observations)
        action_std = torch.exp(self.log_std)
        return torch.distributions.Normal(action_mean, action_std)

    def evaluate_actions(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor):
        """Evaluate actions for PPO training."""
        action_mean, values = self.forward(observations)
        action_std = torch.exp(self.log_std)

        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return values, log_probs, entropy