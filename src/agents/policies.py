import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from .temporal_lstm import TemporalLSTMFeaturesExtractor


class TemporalLSTMActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy with temporal LSTM features extractor for ENKF RL.

    This policy is designed specifically for ENKF environments with temporal dependencies.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: callable,
        temporal_window: int = 3,
        lstm_hidden_size: int = 512,
        num_lstm_layers: int = 2,
        features_dim: int = 512,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = TemporalLSTMFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Set up features extractor kwargs with temporal parameters
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        features_extractor_kwargs.update({
            "temporal_window": temporal_window,
            "lstm_hidden_size": lstm_hidden_size,
            "num_lstm_layers": num_lstm_layers,
            "features_dim": features_dim,
        })

        # Default network architecture if not provided
        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs
        )


class SimpleLSTMActorCriticPolicy(ActorCriticPolicy):
    """
    Simpler LSTM-based policy for debugging purposes.
    Use this if TemporalLSTMActorCriticPolicy is too complex initially.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: callable,
        lstm_hidden_size: int = 256,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        **kwargs
    ):
        # Simple features extractor kwargs
        features_extractor_kwargs = {
            "features_dim": lstm_hidden_size,
            "lstm_hidden_size": lstm_hidden_size,
        }

        if net_arch is None:
            net_arch = dict(pi=[128, 128], vf=[128, 128])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )