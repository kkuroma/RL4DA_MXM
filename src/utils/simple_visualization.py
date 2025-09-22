import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict
import sys

# Add src to path for imports
src_path = os.path.dirname(os.path.dirname(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)


def visualize_path_simple(path_data: Dict[str, np.ndarray], save_path: str):
    """
    Simple visualization like the old codebase - just RMSE comparison for one path.

    Args:
        path_data: Dictionary containing 'true_states', 'forecast_states', 'analysis_states'
        save_path: Path to save the visualization
    """
    # Extract data from path (same logic as old codebase)
    truth = path_data["true_states"]
    background = path_data["forecast_states"].mean(axis=-1)  # Mean over ensemble
    analysis = path_data["analysis_states"].mean(axis=-1)    # Mean over ensemble

    # Compute RMSE (same logic as old codebase)
    rmse_background = np.sqrt(np.mean((truth - background)**2, axis=1))
    rmse_analysis = np.sqrt(np.mean((truth - analysis)**2, axis=1))

    # Create plot (exactly like old codebase)
    plt.figure(figsize=(10, 5))
    plt.plot(rmse_background, label="Truth vs Forecast", linewidth=2, color='red')
    plt.plot(rmse_analysis, label="Truth vs Posterior", linewidth=2, color='blue')

    plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
    plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=12, fontweight='bold')
    plt.title("RMSE Comparison: Truth vs Posterior & Forecast (Path 0)", fontsize=14, fontweight='bold')
    plt.legend(prop={'size': 16})
    plt.grid()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    print(f"Path visualization saved to: {save_path}")


def create_simple_path_visualization(config_dir: str):
    """Create simple path visualization for path 0 only."""
    # Load path 0
    path_file = os.path.join(config_dir, "precomputed_paths", "path_000.npz")
    if not os.path.exists(path_file):
        print(f"Path file not found: {path_file}")
        return

    path_data = np.load(path_file)

    # Create simple visualization
    save_path = os.path.join(config_dir, "precomputed_paths", "path_0_rmse.png")
    visualize_path_simple(path_data, save_path)