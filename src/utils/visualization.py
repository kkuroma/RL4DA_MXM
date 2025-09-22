import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Dict, List, Any, Optional
import sys

# Add src to path for imports
src_path = os.path.dirname(os.path.dirname(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)


def visualize_enkf_path(path_data: Dict[str, np.ndarray], save_path: str, title: str = "ENKF Path"):
    """
    Visualize a single ENKF path showing RMSE comparison.

    Args:
        path_data: Dictionary containing 'true_states', 'forecast_states', 'analysis_states'
        save_path: Path to save the visualization
        title: Title for the plot
    """
    # Extract data from path
    truth = path_data["true_states"]
    forecast_mean = np.mean(path_data["forecast_states"], axis=2)  # Mean over ensemble
    analysis_mean = np.mean(path_data["analysis_states"], axis=2)  # Mean over ensemble

    # Compute RMSE
    rmse_forecast = np.sqrt(np.mean((truth - forecast_mean)**2, axis=1))
    rmse_analysis = np.sqrt(np.mean((truth - analysis_mean)**2, axis=1))

    # Create plot
    plt.figure(figsize=(12, 8))

    # Main RMSE plot
    plt.subplot(2, 2, 1)
    plt.plot(rmse_forecast, label="Truth vs Forecast", linewidth=2, color='red', alpha=0.8)
    plt.plot(rmse_analysis, label="Truth vs Analysis", linewidth=2, color='blue', alpha=0.8)
    plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
    plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=12, fontweight='bold')
    plt.title(f"{title} - RMSE Comparison", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Time series of first 3 variables (truth vs ensemble mean)
    plt.subplot(2, 2, 2)
    time_steps = np.arange(truth.shape[0])
    for i in range(min(3, truth.shape[1])):
        plt.plot(time_steps, truth[:, i], f'C{i}-', label=f'Truth X{i+1}', alpha=0.8)
        plt.plot(time_steps, analysis_mean[:, i], f'C{i}--', label=f'Analysis X{i+1}', alpha=0.6)
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title(f"{title} - State Variables")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Ensemble spread over time
    plt.subplot(2, 2, 3)
    ensemble_std = np.std(path_data["forecast_states"], axis=2)
    ensemble_spread = np.mean(ensemble_std, axis=1)  # Average spread across all variables
    plt.plot(time_steps, ensemble_spread, 'purple', linewidth=2, alpha=0.8)
    plt.xlabel("Time Steps")
    plt.ylabel("Ensemble Spread")
    plt.title(f"{title} - Ensemble Spread")
    plt.grid(True, alpha=0.3)

    # RMSE improvement histogram
    plt.subplot(2, 2, 4)
    rmse_improvement = rmse_forecast - rmse_analysis
    plt.hist(rmse_improvement, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(np.mean(rmse_improvement), color='red', linestyle='--',
                label=f'Mean: {np.mean(rmse_improvement):.3f}')
    plt.xlabel("RMSE Improvement")
    plt.ylabel("Frequency")
    plt.title(f"{title} - RMSE Improvement Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ENKF path visualization saved to: {save_path}")


def visualize_l96_trajectory(trajectory: np.ndarray, title: str = "L96 Trajectory",
                           save_path: Optional[str] = None, max_points: int = 10000):
    """
    Visualize L96 trajectory in multiple ways including 3D phase space.

    Args:
        trajectory: Array of shape (N, T) where N is state dimension, T is time steps
        title: Title for the plots
        save_path: Optional path to save the visualization
        max_points: Maximum number of points to plot for performance
    """
    # Limit points for better visualization
    if trajectory.shape[1] > max_points:
        indices = np.linspace(0, trajectory.shape[1]-1, max_points, dtype=int)
        traj_subset = trajectory[:, indices]
    else:
        traj_subset = trajectory

    fig = plt.figure(figsize=(15, 12))

    # 3D trajectory (if we have at least 3 variables)
    if trajectory.shape[0] >= 3:
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(traj_subset[0], traj_subset[1], traj_subset[2], 'b-', alpha=0.7, linewidth=0.5)
        ax1.scatter(traj_subset[0, 0], traj_subset[1, 0], traj_subset[2, 0],
                    color='green', s=100, label='Start')
        ax1.scatter(traj_subset[0, -1], traj_subset[1, -1], traj_subset[2, -1],
                    color='red', s=100, label='End')
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('X3')
        ax1.set_title(f'{title} - 3D Trajectory')
        ax1.legend()

    # Time series for first few variables
    time_steps = np.arange(traj_subset.shape[1])
    ax2 = fig.add_subplot(232)
    for i in range(min(5, traj_subset.shape[0])):
        ax2.plot(time_steps, traj_subset[i], label=f'X{i+1}', alpha=0.8)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Value')
    ax2.set_title(f'{title} - Time Series')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Phase space: X1 vs X2
    if trajectory.shape[0] >= 2:
        ax3 = fig.add_subplot(233)
        ax3.plot(traj_subset[0], traj_subset[1], 'b-', alpha=0.7, linewidth=0.5)
        ax3.scatter(traj_subset[0, 0], traj_subset[1, 0], color='green', s=50, label='Start')
        ax3.scatter(traj_subset[0, -1], traj_subset[1, -1], color='red', s=50, label='End')
        ax3.set_xlabel('X1')
        ax3.set_ylabel('X2')
        ax3.set_title(f'{title} - X1 vs X2')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Energy over time
    ax4 = fig.add_subplot(234)
    energy = np.sum(trajectory**2, axis=0)
    if len(energy) > max_points:
        energy_subset = energy[indices]
        time_subset = indices
    else:
        energy_subset = energy
        time_subset = time_steps
    ax4.plot(time_subset, energy_subset, 'purple', linewidth=1.5)
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Total Energy')
    ax4.set_title(f'{title} - Total Energy')
    ax4.grid(True, alpha=0.3)

    # Spatial mean and variance
    ax5 = fig.add_subplot(235)
    spatial_mean = np.mean(trajectory, axis=0)
    spatial_var = np.var(trajectory, axis=0)
    if len(spatial_mean) > max_points:
        mean_subset = spatial_mean[indices]
        var_subset = spatial_var[indices]
        time_subset = indices
    else:
        mean_subset = spatial_mean
        var_subset = spatial_var
        time_subset = time_steps
    ax5.plot(time_subset, mean_subset, 'g-', label='Spatial Mean', linewidth=1.5)
    ax5.plot(time_subset, var_subset, 'r-', label='Spatial Variance', linewidth=1.5)
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('Value')
    ax5.set_title(f'{title} - Spatial Statistics')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Hovmöller diagram (space-time plot)
    ax6 = fig.add_subplot(236)
    # Show first 20 variables and limited time for better visualization
    n_vars_show = min(20, trajectory.shape[0])
    n_time_show = min(1000, trajectory.shape[1])
    im = ax6.imshow(trajectory[:n_vars_show, :n_time_show],
                    aspect='auto', cmap='viridis', origin='lower')
    ax6.set_xlabel('Time Steps')
    ax6.set_ylabel('Variable Index')
    ax6.set_title(f'{title} - Hovmöller Diagram')
    plt.colorbar(im, ax=ax6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"L96 trajectory visualization saved to: {save_path}")
    else:
        plt.show()


def visualize_multiple_paths(paths: List[Dict[str, np.ndarray]], save_dir: str,
                           max_paths: int = 5, title_prefix: str = "ENKF Path"):
    """
    Visualize multiple ENKF paths for comparison.

    Args:
        paths: List of path dictionaries
        save_dir: Directory to save visualizations
        max_paths: Maximum number of paths to visualize
        title_prefix: Prefix for plot titles
    """
    os.makedirs(save_dir, exist_ok=True)

    # Individual path visualizations
    for i, path in enumerate(paths[:max_paths]):
        save_path = os.path.join(save_dir, f"path_{i:03d}.png")
        visualize_enkf_path(path, save_path, f"{title_prefix} {i}")

    # Combined RMSE comparison
    plt.figure(figsize=(15, 10))

    # Plot 1: RMSE comparison for all paths
    plt.subplot(2, 3, 1)
    forecast_rmses = []
    analysis_rmses = []

    for i, path in enumerate(paths[:max_paths]):
        truth = path["true_states"]
        forecast_mean = np.mean(path["forecast_states"], axis=2)
        analysis_mean = np.mean(path["analysis_states"], axis=2)

        rmse_forecast = np.sqrt(np.mean((truth - forecast_mean)**2, axis=1))
        rmse_analysis = np.sqrt(np.mean((truth - analysis_mean)**2, axis=1))

        forecast_rmses.append(rmse_forecast)
        analysis_rmses.append(rmse_analysis)

        plt.plot(rmse_forecast, alpha=0.5, color='red', linewidth=1)
        plt.plot(rmse_analysis, alpha=0.5, color='blue', linewidth=1)

    # Plot averages
    if forecast_rmses:
        avg_forecast = np.mean(forecast_rmses, axis=0)
        avg_analysis = np.mean(analysis_rmses, axis=0)
        plt.plot(avg_forecast, 'red', linewidth=3, label='Avg Forecast RMSE')
        plt.plot(avg_analysis, 'blue', linewidth=3, label='Avg Analysis RMSE')

    plt.xlabel("Time Steps")
    plt.ylabel("RMSE")
    plt.title("Multi-Path RMSE Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: RMSE improvement distribution across all paths
    plt.subplot(2, 3, 2)
    all_improvements = []
    for forecast_rmse, analysis_rmse in zip(forecast_rmses, analysis_rmses):
        improvements = forecast_rmse - analysis_rmse
        all_improvements.extend(improvements)

    if all_improvements:
        plt.hist(all_improvements, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.mean(all_improvements), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_improvements):.3f}')
        plt.xlabel("RMSE Improvement")
        plt.ylabel("Frequency")
        plt.title("RMSE Improvement Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 3: Average ensemble spread
    plt.subplot(2, 3, 3)
    all_spreads = []
    for path in paths[:max_paths]:
        ensemble_std = np.std(path["forecast_states"], axis=2)
        ensemble_spread = np.mean(ensemble_std, axis=1)
        all_spreads.append(ensemble_spread)
        plt.plot(ensemble_spread, alpha=0.5, color='purple', linewidth=1)

    if all_spreads:
        avg_spread = np.mean(all_spreads, axis=0)
        plt.plot(avg_spread, 'purple', linewidth=3, label='Average Spread')

    plt.xlabel("Time Steps")
    plt.ylabel("Ensemble Spread")
    plt.title("Multi-Path Ensemble Spread")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Statistics summary
    plt.subplot(2, 3, 4)
    stats_data = []
    for i, (forecast_rmse, analysis_rmse) in enumerate(zip(forecast_rmses, analysis_rmses)):
        mean_forecast = np.mean(forecast_rmse)
        mean_analysis = np.mean(analysis_rmse)
        improvement = mean_forecast - mean_analysis
        stats_data.append([mean_forecast, mean_analysis, improvement])

    if stats_data:
        stats_array = np.array(stats_data)
        x_pos = np.arange(len(stats_data))
        width = 0.35

        plt.bar(x_pos - width/2, stats_array[:, 0], width, label='Forecast RMSE', alpha=0.8, color='red')
        plt.bar(x_pos + width/2, stats_array[:, 1], width, label='Analysis RMSE', alpha=0.8, color='blue')

        plt.xlabel("Path Index")
        plt.ylabel("Mean RMSE")
        plt.title("Mean RMSE by Path")
        plt.legend()
        plt.xticks(x_pos, [f'Path {i}' for i in range(len(stats_data))])
        plt.grid(True, alpha=0.3)

    # Plot 5: Performance correlation
    plt.subplot(2, 3, 5)
    if len(forecast_rmses) > 1:
        for i in range(min(3, len(paths))):
            truth = paths[i]["true_states"]
            plt.plot(truth[:, 0], alpha=0.7, label=f'Path {i} Truth X1')
        plt.xlabel("Time Steps")
        plt.ylabel("X1 Value")
        plt.title("Truth Trajectories Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 6: Overall statistics
    plt.subplot(2, 3, 6)
    if forecast_rmses:
        overall_stats = {
            'Mean Forecast RMSE': np.mean([np.mean(rmse) for rmse in forecast_rmses]),
            'Mean Analysis RMSE': np.mean([np.mean(rmse) for rmse in analysis_rmses]),
            'Mean Improvement': np.mean([np.mean(f_rmse - a_rmse) for f_rmse, a_rmse in zip(forecast_rmses, analysis_rmses)]),
            'Improvement Std': np.std([np.mean(f_rmse - a_rmse) for f_rmse, a_rmse in zip(forecast_rmses, analysis_rmses)])
        }

        metrics = list(overall_stats.keys())
        values = list(overall_stats.values())
        colors = ['red', 'blue', 'green', 'orange']

        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title("Overall Statistics")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    # Save combined plot
    combined_save_path = os.path.join(save_dir, "combined_analysis.png")
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Multi-path visualization saved to: {save_dir}")
    print(f"Combined analysis saved to: {combined_save_path}")


def create_path_visualizations(paths: List[Dict[str, np.ndarray]], save_dir: str,
                             config: Dict[str, Any], source_trajectory: Optional[np.ndarray] = None):
    """
    Create comprehensive visualizations for ENKF paths and optionally source trajectory.

    Args:
        paths: List of ENKF path dictionaries
        save_dir: Directory to save all visualizations
        config: Configuration dictionary
        source_trajectory: Optional source trajectory to visualize
    """
    print("Creating comprehensive path visualizations...")

    # Create main visualization directory
    viz_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Visualize source trajectory if provided
    if source_trajectory is not None:
        source_viz_path = os.path.join(viz_dir, "source_trajectory.png")
        N = config.get('l96_params', {}).get('N', 20)
        F = config.get('l96_params', {}).get('F', 5.0)
        visualize_l96_trajectory(
            source_trajectory,
            title=f"L96 Source Trajectory (N={N}, F={F})",
            save_path=source_viz_path,
            max_points=5000
        )

    # Visualize individual paths and create multi-path analysis
    path_viz_dir = os.path.join(viz_dir, "enkf_paths")
    visualize_multiple_paths(paths, path_viz_dir, max_paths=min(10, len(paths)))

    # Create summary statistics file
    summary_path = os.path.join(viz_dir, "summary_statistics.txt")
    with open(summary_path, 'w') as f:
        f.write("ENKF Path Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Number of paths: {len(paths)}\n")
        if paths:
            path = paths[0]
            T, N = path['true_states'].shape
            N_ens = path['forecast_states'].shape[2]
            f.write(f"Path length: {T} timesteps\n")
            f.write(f"State dimension: {N}\n")
            f.write(f"Ensemble size: {N_ens}\n\n")

            # Compute overall statistics
            all_forecast_rmse = []
            all_analysis_rmse = []

            for path in paths:
                truth = path['true_states']
                forecast_mean = np.mean(path['forecast_states'], axis=2)
                analysis_mean = np.mean(path['analysis_states'], axis=2)

                forecast_rmse = np.sqrt(np.mean((truth - forecast_mean) ** 2))
                analysis_rmse = np.sqrt(np.mean((truth - analysis_mean) ** 2))

                all_forecast_rmse.append(forecast_rmse)
                all_analysis_rmse.append(analysis_rmse)

            f.write("ENKF Performance:\n")
            f.write(f"  Forecast RMSE: {np.mean(all_forecast_rmse):.3f} ± {np.std(all_forecast_rmse):.3f}\n")
            f.write(f"  Analysis RMSE: {np.mean(all_analysis_rmse):.3f} ± {np.std(all_analysis_rmse):.3f}\n")
            f.write(f"  Improvement: {np.mean(all_forecast_rmse) - np.mean(all_analysis_rmse):.3f}\n")
            f.write(f"  Improvement %: {(1 - np.mean(all_analysis_rmse)/np.mean(all_forecast_rmse))*100:.1f}%\n")

    print(f"All visualizations saved to: {viz_dir}")