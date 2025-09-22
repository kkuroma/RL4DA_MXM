import numpy as np
import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import sys

# Add src to path for imports
src_path = os.path.dirname(os.path.dirname(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)

from models.l96 import L96
from enkf.eakf_solver import EAKFSolver
from utils.normalization import DataNormalizer, create_enkf_normalizer
from utils.simple_visualization import visualize_path_simple


def generate_source_trajectory(config: Dict[str, Any], save_path: str) -> np.ndarray:
    """
    Generate a long source trajectory for sampling initial conditions.

    Args:
        config: Configuration dictionary
        save_path: Path to save the trajectory

    Returns:
        Generated trajectory array of shape (N, total_steps)
    """
    print("Generating source trajectory...")

    # Extract L96 parameters
    l96_params = config['l96_params']
    N = l96_params['N']
    F = l96_params['F']

    # Generate trajectory
    total_steps = config['total_steps']
    dt = config['dtda']

    # Create L96 model
    model = L96(params=l96_params, dt=dt)

    # Start with initial conditions
    initial_conditions = config['initial_conditions']
    print(f"Using initial conditions: {initial_conditions} (zero vector)")

    # Initialize trajectory storage
    trajectory = np.zeros((N, total_steps))

    # Initialize model with initial conditions
    model.initialize(np.array(initial_conditions))

    # Generate trajectory with progress bar
    for t in tqdm(range(total_steps), desc="Generating trajectory"):
        state = model.get_current_state()
        trajectory[:, t] = state
        model.step()

        # Check for NaN/inf
        if not np.isfinite(state).all():
            raise RuntimeError(f"Trajectory became unstable at step {t}")

    # Save trajectory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, trajectory)
    print(f"Source trajectory saved to: {save_path}")

    return trajectory


def generate_precomputed_enkf_paths(
    source_trajectory: np.ndarray,
    config: Dict[str, Any],
    save_dir: str
) -> List[Dict[str, np.ndarray]]:
    """
    Generate precomputed ENKF paths from source trajectory.

    Args:
        source_trajectory: Source trajectory array of shape (N, total_steps)
        config: Configuration dictionary
        save_dir: Directory to save precomputed paths

    Returns:
        List of ENKF path dictionaries
    """
    print("Generating precomputed ENKF paths...")

    N, total_steps = source_trajectory.shape

    # Extract configuration
    warmup_steps = config['warmup_steps']
    num_paths = config['num_precomputed_paths']
    sample_var = config['sample_var']

    # ENKF parameters
    num_ensembles = config['num_ensembles']
    H = config['H']
    R = config['R']
    dtda = config['dtda']
    oda = config['oda']
    noise_strength = config['noise_strength']
    inflation = config['inflation']
    l96_params = config['l96_params']

    # Sample starting points from stabilized part of trajectory
    stable_trajectory = source_trajectory[:, warmup_steps:]
    stable_steps = stable_trajectory.shape[1]

    print(f"Source trajectory shape: {source_trajectory.shape}")
    print(f"Using trajectory from step {warmup_steps} onwards: {stable_trajectory.shape}")
    print(f"Sampling {num_paths} starting points from {stable_steps} available points")

    # Sample random starting indices
    start_indices = np.random.choice(stable_steps - 1000, size=num_paths, replace=False)
    start_indices = np.sort(start_indices)  # Sort for better data locality

    paths = []

    for i, start_idx in enumerate(tqdm(start_indices, desc="Generating EAKF paths")):
        # Get initial conditions for this path
        true_initial = stable_trajectory[:, start_idx]

        # Create perturbed ensemble initial conditions
        ensemble_initial = true_initial[:, np.newaxis] + np.random.randn(N, num_ensembles) * sample_var

        # Create EAKF solver
        solver = EAKFSolver(
            model_class=L96,
            model_params=l96_params,
            initial_conditions=true_initial,
            num_ensembles=num_ensembles,
            H=H,
            R=R,
            dtda=dtda,
            oda=oda,
            noise_strength=noise_strength,
            inflation=inflation
        )

        # Override initial ensemble
        solver.true_state = true_initial.copy()
        solver.ensemble = ensemble_initial.copy()

        # Run EAKF simulation
        steps_per_obs = int(oda / dtda)
        max_observations = 500  # 500 steps per path as expected

        path_data = solver.run_eakf(
            num_assimilations=max_observations,
            verbose=False
        )

        # Store path data
        path = {
            'true_states': path_data['true_states'],
            'forecast_states': path_data['background_states'],  # background_states are the forecast states
            'analysis_states': path_data['analysis_states'],
            'observations': path_data['observations'],
            'start_index': start_idx + warmup_steps,  # Adjust for original trajectory
            'path_id': i
        }

        paths.append(path)

        # Save individual path
        path_file = os.path.join(save_dir, f"path_{i:03d}.npz")
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(path_file, **path)

    print(f"Generated {len(paths)} EAKF paths")

    # Create simple visualization for path 0 (like old codebase)
    try:
        print("Creating simple RMSE visualization for path 0...")
        if paths:
            viz_path = os.path.join(save_dir, "path_0_rmse.png")
            visualize_path_simple(paths[0], viz_path)
        print("Path visualization created successfully!")
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")

    return paths


def load_or_generate_data(config: Dict[str, Any], config_dir: str, force_regenerate: bool = False) -> Tuple[List[Dict[str, np.ndarray]], DataNormalizer]:
    """
    Load existing data or generate new data for training.

    Args:
        config: Configuration dictionary
        config_dir: Configuration directory
        force_regenerate: Force regeneration even if data exists

    Returns:
        Tuple of (paths, normalizer)
    """
    source_path = os.path.join(config_dir, config['source_path'])
    precomputed_dir = os.path.join(config_dir, config['precomputed_dir'])
    norm_factors_path = os.path.join(config_dir, config['norm_factors_path'])

    # Generate or load source trajectory
    if force_regenerate or not os.path.exists(source_path):
        print(f"Source trajectory not found at {source_path}. Generating...")
        source_trajectory = generate_source_trajectory(config, source_path)
    else:
        print(f"Loading existing source trajectory from {source_path}")
        source_trajectory = np.load(source_path)

    # Generate or load precomputed paths
    if force_regenerate or not os.path.exists(precomputed_dir):
        print("Precomputed paths directory not found. Generating...")
        paths = generate_precomputed_enkf_paths(source_trajectory, config, precomputed_dir)
    else:
        print(f"Loading existing precomputed paths from {precomputed_dir}")
        paths = load_precomputed_paths(precomputed_dir)

    # Generate or load normalization factors
    if force_regenerate or not os.path.exists(norm_factors_path):
        print("Computing normalization factors...")
        normalizer = create_enkf_normalizer(paths, config)
        normalizer.save(norm_factors_path)
    else:
        print(f"Loading existing normalization factors from {norm_factors_path}")
        normalizer = DataNormalizer()
        normalizer.load(norm_factors_path)

    print(f"Loaded {len(paths)} paths with normalizer")
    return paths, normalizer


def create_visualizations_for_existing_data(config_dir: str, force_regenerate: bool = False):
    """
    Create visualizations for existing precomputed data.

    Args:
        config_dir: Configuration directory containing data
        force_regenerate: Force regeneration of visualizations
    """
    print("Creating visualizations for existing data...")

    # Load configuration
    import sys
    config_path = os.path.join(config_dir, 'config.py')
    if os.path.exists(config_path):
        # Add config directory to path
        sys.path.insert(0, config_dir)
        try:
            import config as config_module
            config = config_module.config
        except ImportError as e:
            print(f"Error loading config: {e}")
            return
        finally:
            # Remove from path
            if config_dir in sys.path:
                sys.path.remove(config_dir)
    else:
        print(f"Config file not found at {config_path}")
        return

    # Load data
    try:
        paths, normalizer = load_or_generate_data(config, config_dir, force_regenerate=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load source trajectory if available
    source_path = os.path.join(config_dir, config.get('source_path', 'source_trajectory.npy'))
    source_trajectory = None
    if os.path.exists(source_path):
        try:
            source_trajectory = np.load(source_path)
            print(f"Loaded source trajectory: {source_trajectory.shape}")
        except Exception as e:
            print(f"Warning: Could not load source trajectory: {e}")

    # Create visualizations
    precomputed_dir = os.path.join(config_dir, config.get('precomputed_dir', 'precomputed_paths'))
    viz_dir = os.path.join(precomputed_dir, 'visualizations')

    # Check if visualizations already exist
    if not force_regenerate and os.path.exists(viz_dir):
        print(f"Visualizations already exist at {viz_dir}. Use force_regenerate=True to recreate.")
        return

    try:
        create_path_visualizations(paths, precomputed_dir, config, source_trajectory)
        print(f"Visualizations created successfully at {viz_dir}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


def load_precomputed_paths(precomputed_dir: str) -> List[Dict[str, np.ndarray]]:
    """Load precomputed paths from directory."""
    import pickle
    paths = []

    # Check for old .pkl format first, then new .npz format
    pkl_files = [f for f in os.listdir(precomputed_dir) if f.startswith('path_') and f.endswith('.pkl')]
    npz_files = [f for f in os.listdir(precomputed_dir) if f.startswith('path_') and f.endswith('.npz')]

    if pkl_files:
        print(f"Loading {len(pkl_files)} paths from legacy .pkl format")
        pkl_files.sort()
        for path_file in pkl_files:
            with open(os.path.join(precomputed_dir, path_file), 'rb') as f:
                path_data = pickle.load(f)

            # Convert to expected format (handle legacy naming)
            path = {
                'true_states': path_data['true_states'],
                'forecast_states': path_data.get('forecast_states', path_data.get('background_states')),
                'analysis_states': path_data['analysis_states'],
                'observations': path_data['observations'],
                'start_index': path_data.get('start_index', 0),
                'path_id': path_data.get('path_id', len(paths))
            }
            paths.append(path)

    elif npz_files:
        print(f"Loading {len(npz_files)} paths from .npz format")
        npz_files.sort()
        for path_file in npz_files:
            path_data = np.load(os.path.join(precomputed_dir, path_file))
            path = {
                'true_states': path_data['true_states'],
                'forecast_states': path_data['forecast_states'],
                'analysis_states': path_data['analysis_states'],
                'observations': path_data['observations'],
                'start_index': path_data['start_index'],
                'path_id': path_data['path_id']
            }
            paths.append(path)
    else:
        raise FileNotFoundError(f"No precomputed path files found in {precomputed_dir}")

    return paths


def verify_data_quality(paths: List[Dict[str, np.ndarray]], config: Dict[str, Any]) -> bool:
    """
    Verify the quality of generated data.

    Args:
        paths: List of ENKF paths
        config: Configuration dictionary

    Returns:
        True if data quality is acceptable
    """
    print("Verifying data quality...")

    issues = []

    for i, path in enumerate(paths):
        # Check for NaN/inf values
        for key, data in path.items():
            if isinstance(data, np.ndarray) and not np.isfinite(data).all():
                issues.append(f"Path {i}: {key} contains NaN/inf values")

        # Check shapes
        T = path['true_states'].shape[0]
        if path['forecast_states'].shape[0] != T:
            issues.append(f"Path {i}: Shape mismatch between true_states and forecast_states")

        if path['analysis_states'].shape[0] != T:
            issues.append(f"Path {i}: Shape mismatch between true_states and analysis_states")

        # Check ENKF improvement
        true_states = path['true_states']
        forecast_mean = np.mean(path['forecast_states'], axis=2)
        analysis_mean = np.mean(path['analysis_states'], axis=2)

        forecast_rmse = np.sqrt(np.mean((forecast_mean - true_states) ** 2))
        analysis_rmse = np.sqrt(np.mean((analysis_mean - true_states) ** 2))

        if analysis_rmse >= forecast_rmse:
            issues.append(f"Path {i}: ENKF not improving (forecast RMSE: {forecast_rmse:.3f}, analysis RMSE: {analysis_rmse:.3f})")

    if issues:
        print("Data quality issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return False

    print("Data quality verification passed!")
    return True


def print_data_statistics(paths: List[Dict[str, np.ndarray]], normalizer: DataNormalizer):
    """Print statistics about the generated data."""
    print("\n=== Data Statistics ===")

    # Basic statistics
    print(f"Number of paths: {len(paths)}")

    if paths:
        path = paths[0]
        T, N = path['true_states'].shape
        N_ens = path['forecast_states'].shape[2]

        print(f"Path length: {T} timesteps")
        print(f"State dimension: {N}")
        print(f"Ensemble size: {N_ens}")

        # ENKF performance statistics
        all_forecast_rmse = []
        all_analysis_rmse = []

        for path in paths:
            true_states = path['true_states']
            forecast_mean = np.mean(path['forecast_states'], axis=2)
            analysis_mean = np.mean(path['analysis_states'], axis=2)

            forecast_rmse = np.sqrt(np.mean((forecast_mean - true_states) ** 2))
            analysis_rmse = np.sqrt(np.mean((analysis_mean - true_states) ** 2))

            all_forecast_rmse.append(forecast_rmse)
            all_analysis_rmse.append(analysis_rmse)

        print(f"\nENKF Performance:")
        print(f"  Forecast RMSE: {np.mean(all_forecast_rmse):.3f} ± {np.std(all_forecast_rmse):.3f}")
        print(f"  Analysis RMSE: {np.mean(all_analysis_rmse):.3f} ± {np.std(all_analysis_rmse):.3f}")
        print(f"  Improvement: {np.mean(all_forecast_rmse) - np.mean(all_analysis_rmse):.3f}")

    # Normalization statistics
    print("\nNormalization Statistics:")
    stats = normalizer.get_stats()
    for data_type, stat in stats.items():
        print(f"  {data_type}: range [{stat['original_range'][0]:.2f}, {stat['original_range'][1]:.2f}] -> [-3, 3]")