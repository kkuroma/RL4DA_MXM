#!/usr/bin/env python3
"""
Generate data for EAKF training.

Usage:
    python generate_data.py --dir=data_dir [--force-regenerate]

Expects a config.py in data_dir with model-specific configurations.
"""

import os
import argparse
import json
import numpy as np
import importlib.util
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from enkf.solver import EAKFSolver


def load_config(data_dir):
    """Load configuration from data_dir/config.py"""
    config_path = os.path.join(data_dir, "config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def generate_trajectory(config, data_dir, force_regenerate=False):
    """Generate main trajectory path"""
    trajectory_path = os.path.join(data_dir, "trajectory.npy")

    if os.path.exists(trajectory_path) and not force_regenerate:
        print(f"Trajectory already exists: {trajectory_path}")
        return np.load(trajectory_path)

    print("Generating trajectory...")

    # Extract configuration
    model_class = config['model_class']
    model_params = config['params']
    total_steps = config['total_steps']
    warmup_steps = config['warmup_steps']
    starting_var = config['starting_var']
    dtda = config['dtda']
    use_solver_ivp = config['use_solver_ivp']

    # Generate random initial condition
    N = model_params['N']
    initial_state = np.random.randn(N) * np.sqrt(starting_var)

    # Initialize model
    model = model_class(params=model_params, dt=dtda, use_solve_ivp=use_solver_ivp)
    model.initialize(initial_state)

    # Generate trajectory
    trajectory = []
    for i in tqdm(range(total_steps), desc="Generating trajectory"):
        state, _ = model.step()
        trajectory.append(state.copy())

    trajectory = np.array(trajectory)

    # Save trajectory
    np.save(trajectory_path, trajectory)
    print(f"Trajectory saved to: {trajectory_path}")

    return trajectory


def generate_precomputed_paths(config, data_dir, trajectory, force_regenerate=False):
    """Generate precomputed EAKF paths"""
    precomputed_dir = os.path.join(data_dir, "precomputed_paths")
    visualizations_dir = os.path.join(data_dir, "visualizations")
    os.makedirs(precomputed_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    num_precomputed_paths = config['num_precomputed_paths']
    warmup_steps = config['warmup_steps']
    max_episode_length = config['max_episode_length']
    
    print(f"Generating {num_precomputed_paths} paths of length {max_episode_length}")

    # Check which paths already exist
    existing_paths = {}
    missing_indices = []
    for i in range(num_precomputed_paths):
        path_file = os.path.join(precomputed_dir, f"{i}.npy")
        if os.path.exists(path_file) and not force_regenerate:
            existing_paths[i] = np.load(path_file, allow_pickle=True).item()
        else:
            missing_indices.append(i)

    if not missing_indices:
        print("All precomputed paths already exist")
        paths = [existing_paths[i] for i in range(num_precomputed_paths)]
        return paths

    print(f"Found {len(existing_paths)} existing paths, generating {len(missing_indices)} missing paths...")

    print("Generating precomputed EAKF paths...")

    # Extract configuration for EAKF
    model_class = config['model_class']
    model_params = config['params']
    num_ensembles = config['num_ensembles']
    dtda = config['dtda']
    oda = config['oda']
    noise_strength = config['noise_strength']
    inflation = config['inflation']
    use_solver_ivp = config['use_solver_ivp']

    # Set up observation operator and noise
    N = model_params['N']
    H = np.eye(N)  # Full observation
    R = 0.1 * np.eye(N)  # Observation noise covariance

    # Sample starting points from trajectory (excluding warmup)
    valid_trajectory = trajectory[warmup_steps:]

    if len(valid_trajectory) < num_precomputed_paths:
        raise ValueError(f"Not enough trajectory points after warmup. Need {num_precomputed_paths}, have {len(valid_trajectory)}")

    # Sample indices uniformly
    indices = np.linspace(0, len(valid_trajectory) - 1, num_precomputed_paths, dtype=int)

    # Generate only missing paths
    for i in tqdm(missing_indices, desc="Generating missing precomputed paths"):
        idx = indices[i]
        # Get initial condition
        initial_condition = valid_trajectory[idx]

        # Initialize EAKF solver
        solver = EAKFSolver(
            model_class=model_class,
            model_params=model_params,
            initial_conditions=initial_condition,
            num_ensembles=num_ensembles,
            H=H,
            R=R,
            dtda=dtda,
            oda=oda,
            noise_strength=noise_strength,
            inflation=inflation,
            use_solver_ivp=use_solver_ivp
        )

        # Run EAKF for max_episode_length steps
        path_data = solver.run_eakf(max_episode_length, verbose=False)

        # Save individual path
        path_file = os.path.join(precomputed_dir, f"{i}.npy")
        np.save(path_file, path_data)

        # Generate and save visualization
        viz_file = os.path.join(visualizations_dir, f"{i}.png")
        solver.visualize(save_path=viz_file, title_suffix=f" - Path {i}")

        existing_paths[i] = path_data

    # Combine existing and newly generated paths in order
    paths = [existing_paths[i] for i in range(num_precomputed_paths)]

    print(f"Precomputed paths saved to: {precomputed_dir}")
    print(f"Visualizations saved to: {visualizations_dir}")
    return paths


def generate_norm_dict(paths, data_dir, force_regenerate=False):
    """Generate normalization dictionary"""
    norm_dict_path = os.path.join(data_dir, "norm_dict.json")

    if os.path.exists(norm_dict_path) and not force_regenerate:
        print(f"Normalization dict already exists: {norm_dict_path}")
        return

    print("Generating normalization dictionary...")

    # Get all keys from first path
    keys = list(paths[0].keys())

    # Calculate max absolute values for each key
    norm_dict = {}
    for key in keys:
        max_vals = []
        for path in paths:
            data = path[key]
            max_val = np.max(np.abs(data))
            max_vals.append(max_val)
        norm_dict[key] = float(np.max(max_vals))

    # Special handling for analysis_states normalization
    if 'analysis_states' in norm_dict and 'previous_analysis' in norm_dict:
        norm_dict['analysis_states'] = max(
            norm_dict['previous_analysis'],
            norm_dict['analysis_states']
        )

    # Save normalization dictionary
    with open(norm_dict_path, 'w') as f:
        json.dump(norm_dict, f, indent=2)

    print(f"Normalization dict saved to: {norm_dict_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate data for EAKF training')
    parser.add_argument('--dir', required=True, help='Data directory containing config.py')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force regeneration of existing files')

    args = parser.parse_args()
    data_dir = args.dir
    force_regenerate = args.force_regenerate

    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load configuration
    config = load_config(data_dir)

    # Generate trajectory
    trajectory = generate_trajectory(config, data_dir, force_regenerate)

    # Generate precomputed paths
    paths = generate_precomputed_paths(config, data_dir, trajectory, force_regenerate)

    # Generate normalization dictionary
    generate_norm_dict(paths, data_dir, force_regenerate)

    print("Data generation complete!")


if __name__ == "__main__":
    main()