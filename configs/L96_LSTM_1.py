import numpy as np
import sys
import os

PROJECT_NAME = "L96_LSTM_1"

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'models'))

from eakf_solver import EAKFSolver
from models.l96 import L96

# Create solver instance for L96 - N=17, N_ens=20, F=5
def create_solver():
    # L96 parameters
    l96_params = {"N": 17, "F": 5.0}
    initial_conditions = np.ones(17) + 0.1 * np.random.randn(17)  # Small perturbations around 1
    num_ensembles = 20  # Full ensemble size
    
    # EAKF setup
    H = np.eye(17)  # Observe all variables
    R = 0.1 * np.eye(17)  # Observation error covariance
    dtda = 0.01  # Time step
    oda = 1.0   # Time between observations
    
    # Create EAKF solver
    solver = EAKFSolver(
        model_class=L96,
        model_params=l96_params,
        initial_conditions=initial_conditions,
        num_ensembles=num_ensembles,
        H=H, R=R, dtda=dtda, oda=oda,
        noise_strength=1.0,
        inflation=3.0,
        use_solver_ivp=False
    )
    solver.set_normed_factor()
    
    return solver

config = {
    # Solver configuration
    "solver": create_solver(),
    
    # RL Environment configuration
    "n_steps": 1000,
    "device": "cpu",
    "fixed_initial_conditions": False,
    
    # Agent configuration
    "agent_type": "custom_lstm",  # "mlp", "lstm", or "custom_lstm"
    "agent_kwargs": {
        "learning_rate": 5e-5,
        "lstm_hidden_size": 64,
        "num_lstm_layers": 2,
    },
    
    # Training configuration
    "total_timesteps": 4000000,
    
    # Evaluation configuration
    "eval_freq": 10000,
    "eval_episodes": 1,
    "eval_initial_condition": np.ones(17),  # Vector of ones for evaluation
    "eval_n_steps": 1000,
    
    # Logging configuration
    "tensorboard_log": f"./logs/{PROJECT_NAME}/tensorboard/",
    "instance_name": "L96_LSTM_1_v1",
    
    # Model saving configuration
    "save_path": f"./logs/{PROJECT_NAME}/weights",
    "save_freq": 50000,  # Save weights every N steps
    "save_best": True,   # Save best model based on eval reward
    
    # Vectorized environment configuration
    "n_envs": 1,  # Number of parallel environments
    
    # Visualization configuration
    "visualize_training": False,   # Enable training episode visualization
    "visualize_eval": False,       # Enable evaluation episode visualization
    "viz_save_path": f"./logs/{PROJECT_NAME}/visualizations/",  # Path to save visualizations
    
    # Normalization factors saving
    "norm_factors_save_path": f"./logs/{PROJECT_NAME}/norm_factors.json",
}