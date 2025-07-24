import numpy as np
import torch
import argparse
import importlib.util
import os
import sys
from gymnasium import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from agents import create_mlp_agent, create_lstm_agent, create_custom_lstm_agent

class EAKF_RL_Env(Env):
    def __init__(self, solver, norm_factor=1.0, n_steps=1000, device='cpu', fixed_initial_condition=True):
        super(EAKF_RL_Env, self).__init__()
        
        self.solver = solver
        self.solver.step() # Perform initial solver step to populate data
        self.N = len(solver.true_initial)  # dimension of state space
        self.n_ens = solver.num_ensembles  # number of ensemble members
        self.norm_factor = norm_factor
        self.device = device
        self.n_steps = n_steps
        self.current_step = 0
        self.fixed_initial_condition = fixed_initial_condition
        self.ground_truth = self.solver.xa
        
        # Single agent handles all ensembles: obs (3*N*n_ens,), action (N*n_ens,)
        self.observation_space = spaces.Box(
            low=-norm_factor, high=norm_factor,
            shape=(3 * self.N * self.n_ens,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-norm_factor, high=norm_factor,
            shape=(self.N * self.n_ens,), dtype=np.float32
        )
    
    def _get_observation(self):
        """Get observation for RL agent."""
        # Get data from solver
        xa = self.solver.xa
        step_results = self.solver.step(custom_kalman_func=None)
        xb = step_results["background_ensemble"]
        derivs = step_results["derivatives"]
        self.ground_truth = step_results["analysis_ensemble"]
        obs = np.concatenate([xa.flatten(), xb.flatten(), derivs.flatten()]) / self.norm_factor
        return obs.astype(np.float32)
    
    def step(self, action):
        """Environment step with custom Kalman function from RL agent."""
        analysis_ensemble = (action * self.norm_factor).reshape(self.N, self.n_ens)
        rmse = np.sqrt(np.mean((self.ground_truth - analysis_ensemble) ** 2))
        reward = -rmse
        
        self.current_step += 1
        done = self.current_step >= self.n_steps
        
        # Get next observation
        next_obs = self._get_observation()
        
        return next_obs, reward, done, done, {'rmse': rmse, 'step': self.current_step}
    
    def reset(self, seed=None):
        """Reset environment."""
        super().reset(seed=seed)
        # Reset solver
        if self.fixed_initial_condition:
            self.solver.reset(None)
        else:
            self.solver.reset(np.random.randn(self.N))
        self.current_step = 0
        self.solver.step() # Perform initial solver step to populate data
        return self._get_observation(), {}

class ZeroInitEvalEnv(EAKF_RL_Env):
    """Evaluation environment that always starts with zero initial conditions."""
    
    def reset(self, seed=None):
        """Reset environment with zero initial conditions."""
        super().reset(seed=seed)
        
        # Always use zero initial conditions for evaluation
        zero_ic = np.zeros(self.N)
        self.solver.reset(zero_ic)
        
        self.current_step = 0
        self.solver.step()  # Perform initial solver step to populate data
        
        return self._get_observation(), {}

def load_config(config_path):
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def create_vectorized_env(config, n_envs=4):
    """Create vectorized environment."""
    def make_env():
        env = EAKF_RL_Env(
            solver=config["solver"],
            norm_factor=config["norm_factor"],
            n_steps=config["n_steps"],
            device=config["device"],
            fixed_initial_condition=config["fixed_initial_conditions"]
        )
        return Monitor(env)
    
    return DummyVecEnv([make_env for _ in range(n_envs)])

def create_eval_env(config):
    """Create evaluation environment with zero initial conditions."""
    eval_env = ZeroInitEvalEnv(
        solver=config["solver"],
        norm_factor=config["norm_factor"],
        n_steps=config["eval_n_steps"],
        device=config["device"],
        fixed_initial_condition=True
    )
    return Monitor(eval_env)

def setup_callbacks(config, eval_env):
    """Setup training callbacks."""
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config["save_path"], "best_model"),
        log_path=os.path.join(config["save_path"], "eval_logs"),
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["eval_episodes"],
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback for periodic saving
    if config["save_freq"] > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=config["save_freq"],
            save_path=os.path.join(config["save_path"], "checkpoints"),
            name_prefix=config["instance_name"]
        )
        callbacks.append(checkpoint_callback)
    
    return CallbackList(callbacks)

def train_from_config(config):
    """Main training function using config."""
    
    # Create directories
    os.makedirs(config["save_path"], exist_ok=True)
    os.makedirs(config["tensorboard_log"], exist_ok=True)
    
    # Create evaluation environment
    eval_env = create_eval_env(config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config, eval_env)
    
    # Create training environment
    if config.get("n_envs", 1) > 1:
        # Vectorized environment
        env = create_vectorized_env(config, n_envs=config["n_envs"])
    else:
        # Single environment
        env = EAKF_RL_Env(
            solver=config["solver"],
            norm_factor=config["norm_factor"],
            n_steps=config["n_steps"],
            device=config["device"],
            fixed_initial_condition=config["fixed_initial_conditions"]
        )
        env = Monitor(env)
    
    # Create agent based on config
    agent_type = config["agent_type"]
    agent_kwargs = config["agent_kwargs"].copy()
    
    # Add TensorBoard logging and device
    agent_kwargs["tensorboard_log"] = os.path.join(
        config["tensorboard_log"], 
        config["instance_name"]
    )
    agent_kwargs["device"] = config["device"]
    
    if agent_type == "mlp":
        model = create_mlp_agent(env, **agent_kwargs)
    elif agent_type == "lstm":
        model = create_lstm_agent(env, **agent_kwargs)
    elif agent_type == "custom_lstm":
        model = create_custom_lstm_agent(env, **agent_kwargs)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    
    # Train the model
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(config["save_path"], f"{config['instance_name']}_final")
    model.save(final_model_path)
    
    print(f"Training completed! Final model saved to: {final_model_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train RL agents for EAKF")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train model
    model = train_from_config(config)
    
    return model

if __name__ == "__main__":
    main()