import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv, InitTracker, StepCounter
from torch.distributions import Normal, Independent
from torch.utils.tensorboard import SummaryWriter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl.torchrl_env import TorchRLMultiAgentEnkfEnv


class MLPPolicy(nn.Module):
    """MLP policy network."""
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, init_log_std=-0.5):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden_dim), activation()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, action_dim)

        # Separate learnable log_std (not state-dependent)
        # init_log_std = -0.5 gives std ≈ 0.606 (reasonable initial exploration)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, observation):
        features = self.backbone(observation)
        mean = self.mean_layer(features)
        # Clamp log_std to prevent explosion
        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        # Expand std to match batch size
        std = std.expand_as(mean)
        return mean, std


class LSTMPolicy(nn.Module):
    """LSTM policy network."""
    def __init__(self, obs_dim, action_dim, hidden_sizes, lstm_hidden_size, lstm_num_layers, activation, init_log_std=-0.5):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, lstm_hidden_size, lstm_num_layers, batch_first=True)
        layers = []
        in_dim = lstm_hidden_size
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden_dim), activation()])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, action_dim)

        # Separate learnable log_std (not state-dependent)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, observation):
        if observation.dim() == 2:
            observation = observation.unsqueeze(1)
        lstm_out, _ = self.lstm(observation)
        features = self.mlp(lstm_out[:, -1, :])
        mean = self.mean_layer(features)
        # Clamp log_std to prevent explosion
        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        # Expand std to match batch size
        std = std.expand_as(mean)
        return mean, std


class MLPValue(nn.Module):
    """MLP value network."""
    def __init__(self, obs_dim, hidden_sizes, activation, num_agents):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden_dim), activation()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_agents))
        self.net = nn.Sequential(*layers)

    def forward(self, observation):
        return self.net(observation)


def load_config(dir_path):
    """Load configuration from directory."""
    config_path = Path(dir_path) / "config.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def create_policy_network(config, obs_dim, action_dim, device):
    """Create policy network based on config."""
    agent_config = config["agent"]
    activation = getattr(nn, agent_config["activation"])
    init_log_std = agent_config.get("init_log_std", -0.5)

    if agent_config["type"] == "MLP":
        policy = MLPPolicy(
            obs_dim, action_dim,
            agent_config["hidden_sizes"],
            activation,
            init_log_std=init_log_std
        )
    elif agent_config["type"] == "LSTM":
        policy = LSTMPolicy(
            obs_dim, action_dim,
            agent_config["hidden_sizes"],
            agent_config["lstm_hidden_size"],
            agent_config["lstm_num_layers"],
            activation,
            init_log_std=init_log_std
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")

    return policy.to(device)


def create_value_network(config, obs_dim, num_agents, device):
    """Create value network."""
    agent_config = config["agent"]
    activation = getattr(nn, agent_config["activation"])
    value = MLPValue(obs_dim, agent_config["hidden_sizes"], activation, num_agents)
    return value.to(device)


def create_policy_module(policy_net, action_spec, device):
    """Wrap policy network in TorchRL modules."""
    policy_module = TensorDictModule(
        policy_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
        log_prob_key="sample_log_prob"
    )

    return policy_module


def setup_training(config, env, device):
    """Initialize all training components."""
    num_agents = env.num_agents
    obs_dim = env.obs_dim_per_agent
    action_dim = env.action_dim_per_agent

    # Create 20 separate policy networks (one per ensemble member)
    policy_nets = nn.ModuleList([
        create_policy_network(config, obs_dim, action_dim, device)
        for _ in range(num_agents)
    ]).to(device)

    # Value network uses single observation (all agents see identical obs)
    value_net = create_value_network(config, obs_dim, num_agents, device)

    def policy_wrapper(observation):
        # observation shapes:
        # - [n_agents, obs_dim]: single env during collection init
        # - [batch, n_agents, obs_dim]: batched collection
        # - [batch, obs_dim]: after flattening for training

        if observation.shape[-1] == obs_dim and len(observation.shape) >= 2 and observation.shape[-2] == num_agents:
            # Has agent dimension: [(...), n_agents, obs_dim]
            batch_shape = observation.shape[:-2]

            # Process each agent independently with its own network
            locs = []
            scales = []
            for agent_idx in range(num_agents):
                agent_obs = observation[..., agent_idx, :]  # [(...), obs_dim]
                loc, scale = policy_nets[agent_idx](agent_obs)  # [(...), action_dim]
                locs.append(loc)
                scales.append(scale)

            # Stack: [(...), n_agents, action_dim]
            loc = torch.stack(locs, dim=-2)
            scale = torch.stack(scales, dim=-2)
            return loc, scale
        else:
            # Training phase after flattening: [batch, obs_dim]
            # Since we flatten [batch, n_agents, ...] -> [batch*n_agents, ...]
            # observation[i] corresponds to agent (i % n_agents)
            batch_size = observation.shape[0]

            # Vectorized approach: process all samples for each agent at once
            locs = []
            scales = []
            for agent_idx in range(num_agents):
                # Get all observations for this agent
                agent_indices = torch.arange(agent_idx, batch_size, num_agents, device=observation.device)
                agent_obs = observation[agent_indices]  # [batch_size/num_agents, obs_dim]

                if agent_obs.shape[0] > 0:
                    loc, scale = policy_nets[agent_idx](agent_obs)
                    locs.append(loc)
                    scales.append(scale)

            # Interleave results back to original order
            loc = torch.cat(locs, dim=0)  # [batch, action_dim]
            scale = torch.cat(scales, dim=0)  # [batch, action_dim]

            # Need to reorder: currently grouped by agent, need to interleave
            # Create proper ordering indices
            num_per_agent = batch_size // num_agents
            remainder = batch_size % num_agents

            reorder_indices = []
            for i in range(num_per_agent):
                for agent_idx in range(num_agents):
                    reorder_indices.append(i * num_agents + agent_idx)
            for agent_idx in range(remainder):
                reorder_indices.append(num_per_agent * num_agents + agent_idx)

            # Undo the grouping by creating inverse permutation
            inverse_perm = torch.empty(batch_size, dtype=torch.long, device=observation.device)
            for new_idx, old_idx in enumerate(reorder_indices):
                inverse_perm[old_idx] = new_idx

            loc = loc[inverse_perm]
            scale = scale[inverse_perm]

            return loc, scale

    policy_module = TensorDictModule(
        policy_wrapper,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
        log_prob_key="sample_log_prob"
    )

    def value_wrapper(observation):
        # observation can be:
        # - [n_agents, obs_dim]: single env during collection init
        # - [batch, n_agents, obs_dim]: batched collection
        # - [batch, obs_dim]: after flattening for training
        # Since all agents see identical obs, we can use any single agent's obs

        if observation.shape[-1] == obs_dim and len(observation.shape) >= 2 and observation.shape[-2] == num_agents:
            # Has agent dimension: [(...), n_agents, obs_dim]
            # All agents have identical obs, so just take the first one
            batch_shape = observation.shape[:-2]
            single_obs = observation[..., 0, :]  # [(...), obs_dim]
            values = value_net(single_obs)  # [(...), num_agents]
            return values.reshape(*batch_shape, num_agents, 1)  # [(...), n_agents, 1]
        else:
            # Training phase: [batch, obs_dim_per_agent]
            # Each obs is already single-agent obs (all identical anyway)
            values = value_net(observation)  # [batch, num_agents]
            return values.mean(dim=-1, keepdim=True)  # [batch, 1]

    value_module = ValueOperator(
        module=value_wrapper,
        in_keys=["observation"]
    )

    train_cfg = config["training"]

    advantage_module = GAE(
        gamma=train_cfg["gamma"],
        lmbda=train_cfg["gae_lambda"],
        value_network=value_module,
        average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=train_cfg["clip_epsilon"],
        entropy_bonus=True,
        entropy_coeff=train_cfg["entropy_coef"],
        critic_coeff=train_cfg["critic_coef"],
        loss_critic_type="smooth_l1",
        normalize_advantage=False
    ).to(device)

    params = list(policy_nets.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=train_cfg["lr"])

    return policy_module, value_module, loss_module, optimizer, advantage_module, policy_nets, value_net


def evaluate(env, policy, device, num_episodes=1, max_steps=500):
    """Run evaluation episodes."""
    from torchrl.envs.utils import step_mdp

    policy.eval()
    total_rewards = []

    with torch.no_grad():
        for _ in range(num_episodes):
            td = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                td = policy(td)
                td = env.step(td)
                episode_reward += td["next", "reward"].sum().item()

                if td["next", "done"].any().item():
                    break

                # Move next state to current for next iteration
                td = step_mdp(td)

            total_rewards.append(episode_reward)

    policy.train()
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards)
    }


def train(dir_path):
    """Main training loop."""
    dir_path = Path(dir_path)
    config = load_config(dir_path)

    device = torch.device(config["device"])
    train_cfg = config["training"]

    checkpoint_dir = dir_path / "checkpoints"
    tensorboard_dir = dir_path / "tensorboard"
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    train_env = TransformedEnv(
        TorchRLMultiAgentEnkfEnv(str(dir_path), is_eval=False, device=device)
    )
    train_env.append_transform(InitTracker())
    train_env.append_transform(StepCounter(max_steps=config["max_episode_length"]))

    eval_env = TransformedEnv(
        TorchRLMultiAgentEnkfEnv(str(dir_path), is_eval=True, device=device)
    )
    eval_env.append_transform(InitTracker())
    eval_env.append_transform(StepCounter(max_steps=config["max_episode_length"]))

    policy_module, value_module, loss_module, optimizer, advantage_module, policy_nets, value_net = setup_training(
        config, train_env, device
    )

    collector = SyncDataCollector(
        train_env,
        policy_module,
        frames_per_batch=train_cfg["frames_per_batch"],
        total_frames=train_cfg["total_frames"],
        device=device,
        storing_device=device,
        max_frames_per_traj=-1
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(train_cfg["frames_per_batch"]),
        batch_size=train_cfg["sgd_minibatch_size"]
    )

    best_eval_reward = -float('inf')
    total_frames = 0

    pbar = tqdm(total=train_cfg["total_frames"], desc="Training", unit="frames")

    for batch_idx, tensordict_data in enumerate(collector):
        total_frames += tensordict_data.numel()

        with torch.no_grad():
            advantage_module(tensordict_data)

        # Flatten agent dimension into batch for training
        # [2000, 20, ...] -> [40000, ...]
        batch_size = tensordict_data.shape[0]
        num_agents = tensordict_data["observation"].shape[1]

        data_flat = TensorDict({}, batch_size=[batch_size * num_agents])
        for key in tensordict_data.keys(True, True):
            val = tensordict_data[key]
            if len(val.shape) >= 2 and val.shape[1] == num_agents:
                # Flatten [batch, n_agents, ...] -> [batch*n_agents, ...]
                new_shape = (batch_size * num_agents,) + val.shape[2:]
                data_flat[key] = val.reshape(new_shape)
            else:
                # Repeat for non-agent tensors (like traj_ids)
                data_flat[key] = val.repeat_interleave(num_agents, dim=0)

        data_view = data_flat

        for epoch in range(train_cfg["num_sgd_iter"]):
            replay_buffer.empty()
            replay_buffer.extend(data_view)

            num_minibatches = len(data_view) // train_cfg["sgd_minibatch_size"]

            for i in range(num_minibatches):
                subdata = replay_buffer.sample().to(device)

                # Debug first minibatch
                if batch_idx == 0 and epoch == 0 and i == 0:
                    print(f"\n=== Sampled Minibatch Shapes ===")
                    for key in subdata.keys(True, True):
                        print(f"{key}: {subdata[key].shape}")

                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"].mean() +
                    loss_vals["loss_critic"].mean() +
                    loss_vals["loss_entropy"].mean()
                )

                # Check for NaN/Inf in losses
                if torch.isnan(loss_value) or torch.isinf(loss_value):
                    print(f"\n!!! NaN/Inf detected at batch {batch_idx}, epoch {epoch}, minibatch {i} !!!")
                    print(f"Total frames so far: {total_frames}")
                    print(f"\nLosses:")
                    print(f"  loss_objective: {loss_vals['loss_objective'].item()}")
                    print(f"  loss_critic: {loss_vals['loss_critic'].item()}")
                    print(f"  loss_entropy: {loss_vals['loss_entropy'].item()}")
                    print(f"\nData ranges:")
                    print(f"  Observation: [{subdata['observation'].min().item():.4f}, {subdata['observation'].max().item():.4f}]")
                    print(f"  Action: [{subdata['action'].min().item():.4f}, {subdata['action'].max().item():.4f}]")
                    print(f"  Reward: [{subdata['next', 'reward'].min().item():.4f}, {subdata['next', 'reward'].max().item():.4f}]")
                    print(f"  State value: [{subdata['state_value'].min().item():.4f}, {subdata['state_value'].max().item():.4f}]")
                    print(f"  Advantage: [{subdata['advantage'].min().item():.4f}, {subdata['advantage'].max().item():.4f}]")
                    if 'loc' in subdata:
                        print(f"  Policy loc: [{subdata['loc'].min().item():.4f}, {subdata['loc'].max().item():.4f}]")
                        print(f"  Policy scale: [{subdata['scale'].min().item():.4f}, {subdata['scale'].max().item():.4f}]")
                    if 'sample_log_prob' in subdata:
                        print(f"  Log prob: [{subdata['sample_log_prob'].min().item():.4f}, {subdata['sample_log_prob'].max().item():.4f}]")
                    raise ValueError("NaN/Inf detected in loss!")

                optimizer.zero_grad()
                loss_value.backward()
                params = list(policy_nets.parameters()) + list(value_net.parameters())

                # Log gradient statistics before clipping
                grad_norms = []
                for p in params:
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm().item())

                if grad_norms:
                    max_grad_before_clip = max(grad_norms)
                    mean_grad_before_clip = sum(grad_norms) / len(grad_norms)
                else:
                    max_grad_before_clip = 0.0
                    mean_grad_before_clip = 0.0

                total_grad_norm = torch.nn.utils.clip_grad_norm_(params, train_cfg["max_grad_norm"])
                optimizer.step()

                # Store for logging (will use last minibatch's values)
                grad_stats = {
                    'grad_norm_total': total_grad_norm.item(),
                    'grad_norm_max': max_grad_before_clip,
                    'grad_norm_mean': mean_grad_before_clip
                }

        mean_reward = tensordict_data["next", "reward"].sum(dim=-1).mean().item()

        writer.add_scalar("train/reward", mean_reward, total_frames)
        writer.add_scalar("train/loss_total", loss_value.item(), total_frames)
        writer.add_scalar("train/loss_policy", loss_vals["loss_objective"].item(), total_frames)
        writer.add_scalar("train/loss_critic", loss_vals["loss_critic"].item(), total_frames)
        writer.add_scalar("train/loss_entropy", loss_vals["loss_entropy"].item(), total_frames)

        # Log value and advantage statistics
        writer.add_scalar("train/value_mean", data_flat["state_value"].mean().item(), total_frames)
        writer.add_scalar("train/value_std", data_flat["state_value"].std().item(), total_frames)
        writer.add_scalar("train/advantage_mean", data_flat["advantage"].mean().item(), total_frames)
        writer.add_scalar("train/advantage_std", data_flat["advantage"].std().item(), total_frames)
        writer.add_scalar("train/policy_std", data_flat["scale"].mean().item(), total_frames)

        # Compute actual entropy (not just loss component)
        if "scale" in data_flat:
            # For Normal distribution: entropy = 0.5 * log(2 * pi * e * sigma^2)
            entropy_value = 0.5 * torch.log(2 * np.pi * np.e * data_flat["scale"] ** 2).sum(dim=-1).mean()
            writer.add_scalar("train/entropy", entropy_value.item(), total_frames)

        pbar.update(tensordict_data.numel())
        pbar.set_postfix({
            "reward": f"{mean_reward:.4f}",
            "loss": f"{loss_value.item():.4f}"
        })

        # Comprehensive logging every batch (not just every 10)
        log_entry = {
            'batch': batch_idx,
            'frames': total_frames,
            'reward': mean_reward,
            'loss_total': loss_value.item(),
            'loss_policy': loss_vals['loss_objective'].item(),
            'loss_critic': loss_vals['loss_critic'].item(),
            'loss_entropy': loss_vals['loss_entropy'].item(),
            'value_mean': data_flat['state_value'].mean().item(),
            'value_max': data_flat['state_value'].max().item(),
            'value_min': data_flat['state_value'].min().item(),
            'value_std': data_flat['state_value'].std().item(),
            'advantage_mean': data_flat['advantage'].mean().item(),
            'advantage_max': data_flat['advantage'].max().item(),
            'advantage_min': data_flat['advantage'].min().item(),
            'advantage_std': data_flat['advantage'].std().item(),
            'action_min': data_flat['action'].min().item(),
            'action_max': data_flat['action'].max().item(),
            'action_saturated_count': (data_flat['action'].abs() > 0.99).sum().item(),
            'action_saturated_pct': (data_flat['action'].abs() > 0.99).float().mean().item() * 100,
            'loc_mean': data_flat['loc'].mean().item(),
            'loc_max': data_flat['loc'].max().item(),
            'loc_min': data_flat['loc'].min().item(),
            'loc_std': data_flat['loc'].std().item(),
            'scale_mean': data_flat['scale'].mean().item(),
            'scale_max': data_flat['scale'].max().item(),
            'scale_min': data_flat['scale'].min().item(),
            'log_prob_mean': data_flat['sample_log_prob'].mean().item(),
            'log_prob_max': data_flat['sample_log_prob'].max().item(),
            'log_prob_min': data_flat['sample_log_prob'].min().item(),
            'learned_log_std': policy_nets[0].log_std.data.mean().item(),
            'learned_std': torch.exp(policy_nets[0].log_std.data.mean()).item(),
            'grad_norm_total': grad_stats['grad_norm_total'],
            'grad_norm_max': grad_stats['grad_norm_max'],
            'grad_norm_mean': grad_stats['grad_norm_mean'],
        }

        # Print to console every 10 batches
        if batch_idx % 10 == 0:
            print(f"\n=== Training Stats (batch {batch_idx}, frames {total_frames}) ===")
            print(f"  Reward: {log_entry['reward']:.4f}")
            print(f"  Loss: {log_entry['loss_total']:.4f} (policy: {log_entry['loss_policy']:.4f}, critic: {log_entry['loss_critic']:.4f}, entropy: {log_entry['loss_entropy']:.4f})")
            print(f"  Value: mean={log_entry['value_mean']:.4f}, max={log_entry['value_max']:.4f}, min={log_entry['value_min']:.4f}")
            print(f"  Advantage: mean={log_entry['advantage_mean']:.4f}, max={log_entry['advantage_max']:.4f}, min={log_entry['advantage_min']:.4f}")
            print(f"  Action: min={log_entry['action_min']:.4f}, max={log_entry['action_max']:.4f}, saturated={log_entry['action_saturated_pct']:.1f}%")
            print(f"  Policy loc: mean={log_entry['loc_mean']:.4f}, max={log_entry['loc_max']:.4f}, min={log_entry['loc_min']:.4f}")
            print(f"  Policy scale: mean={log_entry['scale_mean']:.4f}, max={log_entry['scale_max']:.4f}")
            print(f"  Log prob: mean={log_entry['log_prob_mean']:.4f}, min={log_entry['log_prob_min']:.4f}")
            print(f"  Learned std: {log_entry['learned_std']:.4f}")
            print(f"  Gradients: total_norm={log_entry['grad_norm_total']:.4f}, max={log_entry['grad_norm_max']:.4f}, mean={log_entry['grad_norm_mean']:.4f}")

        # Warning checks - print immediately if something looks suspicious
        warnings = []
        if abs(log_entry['value_max']) > 20:
            warnings.append(f"⚠️  Value max too high: {log_entry['value_max']:.2f}")
        if abs(log_entry['value_min']) < -10:
            warnings.append(f"⚠️  Value min too low: {log_entry['value_min']:.2f}")
        if abs(log_entry['loc_max']) > 15:
            warnings.append(f"⚠️  Policy loc exploding: {log_entry['loc_max']:.2f}")
        if log_entry['log_prob_min'] < -100:
            warnings.append(f"⚠️  Log prob very negative: {log_entry['log_prob_min']:.2f}")
        if log_entry['action_saturated_pct'] > 50:
            warnings.append(f"⚠️  >50% actions saturated: {log_entry['action_saturated_pct']:.1f}%")
        if abs(log_entry['loss_policy']) > 10:
            warnings.append(f"⚠️  Policy loss high: {log_entry['loss_policy']:.2f}")
        if log_entry['grad_norm_total'] > train_cfg["max_grad_norm"] * 10:
            warnings.append(f"⚠️  Gradients very large (clipped): {log_entry['grad_norm_total']:.2f}")
        if log_entry['grad_norm_max'] > 100:
            warnings.append(f"⚠️  Individual gradient exploding: {log_entry['grad_norm_max']:.2f}")

        if warnings:
            print(f"\n🔴 WARNINGS at batch {batch_idx}:")
            for w in warnings:
                print(f"  {w}")

        # Write detailed log to file
        log_file = dir_path / "training_log.txt"
        with open(log_file, 'a') as f:
            # Write header on first batch
            if batch_idx == 0:
                f.write(",".join(log_entry.keys()) + "\n")
            f.write(",".join(str(v) for v in log_entry.values()) + "\n")

        if total_frames % train_cfg["checkpoint_interval"] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{total_frames}.pt"
            torch.save({
                "total_frames": total_frames,
                "policy_state_dict": policy_module.state_dict(),
                "value_state_dict": value_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if total_frames % train_cfg["eval_interval"] == 0:
            eval_results = evaluate(eval_env, policy_module, device)
            eval_reward = eval_results["mean_reward"]

            writer.add_scalar("eval/reward", eval_reward, total_frames)
            print(f"Eval Reward: {eval_reward:.4f}")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_weights_path = dir_path / "best_weights.pt"
                torch.save({
                    "total_frames": total_frames,
                    "policy_state_dict": policy_module.state_dict(),
                    "value_state_dict": value_module.state_dict(),
                    "eval_reward": eval_reward,
                }, best_weights_path)
                print(f"New best model saved: {best_weights_path} (reward: {eval_reward:.4f})")

    pbar.close()

    final_checkpoint = checkpoint_dir / "final_model.pt"
    torch.save({
        "total_frames": total_frames,
        "policy_state_dict": policy_module.state_dict(),
        "value_state_dict": value_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_checkpoint)
    print(f"Training complete. Final model saved: {final_checkpoint}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to logs directory")
    args = parser.parse_args()

    train(args.dir)
