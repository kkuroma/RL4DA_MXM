#!/usr/bin/env python3
"""
Test script for TorchRL Multi-Agent ENKF Environment

This script verifies that the TorchRL environment wrapper is working correctly
by testing basic environment operations: reset, step, and spec compliance.

Usage:
    python scripts/test_torchrl_env.py --dir=logs/L96_1
    python scripts/test_torchrl_env.py --dir=logs/L96_1 --device=cuda
"""

import os
import sys
import argparse

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
from rl.torchrl_env import create_torchrl_multiagent_enkf_env


def test_env_creation(log_dir, device):
    """Test environment creation."""
    print("=" * 60)
    print("TEST 1: Environment Creation")
    print("=" * 60)

    try:
        env = create_torchrl_multiagent_enkf_env(
            logs_dir_path=log_dir,
            eval_mode=False,
            device=device,
        )
        print(f"✓ Environment created successfully")
        print(f"  - Number of agents: {env.num_agents}")
        print(f"  - Obs dim per agent: {env.obs_dim_per_agent}")
        print(f"  - Action dim per agent: {env.action_dim_per_agent}")
        print(f"  - Device: {env.device}")
        return env
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        raise


def test_specs(env):
    """Test environment specifications."""
    print("\n" + "=" * 60)
    print("TEST 2: Environment Specifications")
    print("=" * 60)

    try:
        # Test observation spec
        obs_spec = env.observation_spec
        print(f"✓ Observation spec:")
        print(f"  - Shape: {obs_spec['observation'].shape}")
        print(f"  - Dtype: {obs_spec['observation'].dtype}")
        print(f"  - Low: {obs_spec['observation'].space.low}")
        print(f"  - High: {obs_spec['observation'].space.high}")

        # Test action spec
        action_spec = env.action_spec
        print(f"✓ Action spec:")
        print(f"  - Shape: {action_spec['action'].shape}")
        print(f"  - Dtype: {action_spec['action'].dtype}")
        print(f"  - Low: {action_spec['action'].space.low}")
        print(f"  - High: {action_spec['action'].space.high}")

        # Test reward spec
        reward_spec = env.reward_spec
        print(f"✓ Reward spec:")
        print(f"  - Shape: {reward_spec['reward'].shape}")
        print(f"  - Dtype: {reward_spec['reward'].dtype}")

        # Test done spec
        done_spec = env.done_spec
        print(f"✓ Done spec:")
        print(f"  - Shape: {done_spec['done'].shape}")
        print(f"  - Dtype: {done_spec['done'].dtype}")

        return True
    except Exception as e:
        print(f"✗ Spec test failed: {e}")
        raise


def test_reset(env):
    """Test environment reset."""
    print("\n" + "=" * 60)
    print("TEST 3: Environment Reset")
    print("=" * 60)

    try:
        tensordict = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  - TensorDict keys: {list(tensordict.keys())}")
        print(f"  - Observation shape: {tensordict['observation'].shape}")
        print(f"  - Observation dtype: {tensordict['observation'].dtype}")
        print(f"  - Done: {tensordict['done']}")
        print(f"  - Terminated: {tensordict['terminated']}")

        # Check observation values are within spec
        obs = tensordict['observation']
        print(f"  - Observation min: {obs.min().item():.4f}")
        print(f"  - Observation max: {obs.max().item():.4f}")
        print(f"  - Observation mean: {obs.mean().item():.4f}")

        return tensordict
    except Exception as e:
        print(f"✗ Reset test failed: {e}")
        raise


def test_step(env, initial_tensordict):
    """Test environment step."""
    print("\n" + "=" * 60)
    print("TEST 4: Environment Step")
    print("=" * 60)

    try:
        # Sample random action
        action_spec = env.action_spec
        action = action_spec['action'].rand()
        print(f"✓ Sampled random action")
        print(f"  - Action shape: {action.shape}")
        print(f"  - Action dtype: {action.dtype}")
        print(f"  - Action min: {action.min().item():.4f}")
        print(f"  - Action max: {action.max().item():.4f}")

        # Create action tensordict
        action_td = initial_tensordict.clone()
        action_td['action'] = action

        # Step environment
        next_tensordict = env.step(action_td)
        print(f"✓ Environment step successful")
        print(f"  - TensorDict keys: {list(next_tensordict.keys())}")
        print(f"  - Next observation shape: {next_tensordict['observation'].shape}")
        print(f"  - Reward shape: {next_tensordict['reward'].shape}")
        print(f"  - Reward values: {next_tensordict['reward'].squeeze()}")
        print(f"  - Reward mean: {next_tensordict['reward'].mean().item():.4f}")
        print(f"  - Done: {next_tensordict['done']}")
        print(f"  - Terminated: {next_tensordict['terminated']}")

        return next_tensordict
    except Exception as e:
        print(f"✗ Step test failed: {e}")
        raise


def test_episode_rollout(env, num_steps=10):
    """Test a short episode rollout."""
    print("\n" + "=" * 60)
    print(f"TEST 5: Episode Rollout ({num_steps} steps)")
    print("=" * 60)

    try:
        tensordict = env.reset()
        total_reward = 0.0
        rewards_per_step = []

        for step in range(num_steps):
            # Sample random action
            action = env.action_spec['action'].rand()
            action_td = tensordict.clone()
            action_td['action'] = action

            # Step environment
            tensordict = env.step(action_td)

            # Accumulate rewards
            step_reward = tensordict['reward'].mean().item()
            total_reward += step_reward
            rewards_per_step.append(step_reward)

            # Check if done
            if tensordict['done'].item():
                print(f"  Episode ended at step {step + 1}")
                break

        print(f"✓ Episode rollout completed")
        print(f"  - Steps: {len(rewards_per_step)}")
        print(f"  - Total reward: {total_reward:.4f}")
        print(f"  - Mean reward per step: {np.mean(rewards_per_step):.4f}")
        print(f"  - Reward std: {np.std(rewards_per_step):.4f}")
        print(f"  - Min reward: {np.min(rewards_per_step):.4f}")
        print(f"  - Max reward: {np.max(rewards_per_step):.4f}")

        return True
    except Exception as e:
        print(f"✗ Episode rollout test failed: {e}")
        raise


def test_batch_consistency(env):
    """Test that environment maintains consistent behavior across resets."""
    print("\n" + "=" * 60)
    print("TEST 6: Batch Consistency")
    print("=" * 60)

    try:
        # Reset and step multiple times
        observations = []
        rewards = []

        for i in range(3):
            td = env.reset()
            observations.append(td['observation'].clone())

            action = env.action_spec['action'].rand()
            action_td = td.clone()
            action_td['action'] = action

            next_td = env.step(action_td)
            rewards.append(next_td['reward'].clone())

        print(f"✓ Multiple resets and steps successful")
        print(f"  - Number of resets: {len(observations)}")
        print(f"  - Observation shapes: {[obs.shape for obs in observations]}")
        print(f"  - Reward shapes: {[r.shape for r in rewards]}")

        # Check shapes are consistent
        obs_shapes = [obs.shape for obs in observations]
        reward_shapes = [r.shape for r in rewards]
        assert all(s == obs_shapes[0] for s in obs_shapes), "Inconsistent observation shapes"
        assert all(s == reward_shapes[0] for s in reward_shapes), "Inconsistent reward shapes"
        print(f"  - All shapes consistent ✓")

        return True
    except Exception as e:
        print(f"✗ Batch consistency test failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Test TorchRL Multi-Agent ENKF Environment')
    parser.add_argument('--dir', required=True, help='Log directory (e.g., logs/L96_1)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use (default: cpu)')
    args = parser.parse_args()

    # Convert to absolute path
    log_dir = os.path.abspath(args.dir)

    # Validate log directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    # Check required files
    required_files = [
        os.path.join(log_dir, "config.py"),
        os.path.join(log_dir, "precomputed_paths"),
        os.path.join(log_dir, "norm_dict.json")
    ]

    for required_file in required_files:
        if not os.path.exists(required_file):
            print(f"Error: Required file/directory not found: {required_file}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("TorchRL Multi-Agent ENKF Environment Test Suite")
    print("=" * 60)
    print(f"Log directory: {log_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    try:
        # Run tests
        env = test_env_creation(log_dir, args.device)
        test_specs(env)
        initial_td = test_reset(env)
        test_step(env, initial_td)
        test_episode_rollout(env, num_steps=10)
        test_batch_consistency(env)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TESTS FAILED ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
