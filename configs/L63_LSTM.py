import numpy as np
import sys
import os

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'models'))

from eakf_solver import EAKFSolver
from models.l63 import L63

# Create solver instance based on eakf_demo.ipynb settings
def create_solver():
    # L63 parameters
    l63_params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0/3.0}
    initial_conditions = np.array([20.0, 30.0, 0.0])
    num_ensembles = 20  # Full ensemble size
    
    # EAKF setup
    H = np.eye(3)  # Observe all variables
    R = 0.1 * np.eye(3)  # Observation error covariance
    dtda = 0.01  # Time step
    oda = 1.0   # Time between observations
    
    # Create EAKF solver
    solver = EAKFSolver(
        model_class=L63,
        model_params=l63_params,
        initial_conditions=initial_conditions,
        num_ensembles=num_ensembles,
        H=H, R=R, dtda=dtda, oda=oda,
        noise_strength=1.0,
        inflation=1.50,
        use_solver_ivp=False
    )
    
    return solver

config = {
    # Solver configuration
    "solver": create_solver(),
    
    # RL Environment configuration
    "norm_factor": 60.0,
    "n_steps": 1000,
    "device": "cpu",
    "fixed_initial_conditions": False,
    
    # Agent configuration
    "agent_type": "custom_lstm",  # "mlp", "lstm", or "custom_lstm"
    "agent_kwargs": {
        "learning_rate": 3e-4,
        "lstm_hidden_size": 64,
        "num_lstm_layers": 2,
    },
    
    # Training configuration
    "total_timesteps": 4000000,
    
    # Evaluation configuration
    "eval_freq": 2500,
    "eval_episodes": 1,
    "eval_initial_condition": np.array([0.0, 0.0, 0.0]),  # Zero vector for evaluation
    "eval_n_steps": 1000,
    
    # Logging configuration
    "tensorboard_log": "./logs/L63_LSTM/tensorboard/",
    "instance_name": "L63_LSTM_v1",
    
    # Model saving configuration
    "save_path": "./logs/L63_LSTM/weights",
    "save_freq": 50000,  # Save weights every N steps
    "save_best": True,   # Save best model based on eval reward
    
    # Vectorized environment configuration
    "n_envs": 4,  # Number of parallel environments
}