import numpy as np
import torch
import argparse
import importlib.util
import os
import sys
import json
from gymnasium import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from agents import create_mlp_agent, create_lstm_agent, create_custom_lstm_agent

# TODO: 3 different models
# L96 - N=20, N_ens=20, F=5 (ground truth)
# Finish these 3 first
# L96 - N=17, N_ens=20, F=5
# L96 - N=20, N_ens=20, F=4.5
# L96 - N=20, N_ens=20, F=5.5
# Train this combined agent
# 3 agents - one for each of the last three, trained using RL as well?
# p(x,y,z | x_a)
# meta-agent? combine these three (MoE esque) - pick one of the 3 agents

class EAKF_RL_Env(Env):
    def __init__(self, solver, n_steps=1000, device='cpu', fixed_initial_condition=True, 
                 visualize_episodes=False, viz_save_path=None):
        super(EAKF_RL_Env, self).__init__()
        
        self.solver = solver
        self.N = len(solver.true_initial)  # dimension of state space
        self.norm_factors = solver.normed_factors
        self.solver.step() # Perform initial solver step to populate data
        self.n_ens = solver.num_ensembles  # number of ensemble members
        # norm_factor is now computed dynamically in get_normed_factor()
        self.device = device
        self.n_steps = n_steps
        self.current_step = 0
        self.fixed_initial_condition = fixed_initial_condition
        self.ground_truth = self.solver.xa
        self.alpha = 0
        self.visualize_episodes = visualize_episodes
        self.viz_save_path = viz_save_path
        self.current_ep = 0
        # Single agent handles all ensembles: obs (3*N*n_ens,), action (N*n_ens,)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(4 * self.N * self.n_ens,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(self.N * self.n_ens,), dtype=np.float32
        )
        self.last_reward = -1e10
        self.curr_reward = 0
    
    def _get_observation(self):
        """Get observation for RL agent."""
        # Get data from solver
        xa = self.solver.xa / self.norm_factors["analysis_states"]
        step_results = self.solver.step(custom_kalman_func=None)
        xb = step_results["background_ensemble"] / self.norm_factors["background_states"]
        xo = step_results["ensemble_observation"] / self.norm_factors["observations"]
        dx = step_results["derivatives"] / self.norm_factors["derivatives"]
        self.ground_truth = step_results["analysis_ensemble"] / self.norm_factors["analysis_states"]
        obs = np.concatenate([xa.flatten(), xb.flatten(), xo.flatten(), dx.flatten()])
        return obs.astype(np.float32)
    
    def step(self, action):
        """Environment step with custom Kalman function from RL agent."""
        analysis_ensemble = action.reshape(self.N, self.n_ens)
        rmse = np.sqrt(np.mean((self.ground_truth - analysis_ensemble) ** 2))
        reward = -rmse
        self.curr_reward += reward
        analysis_combined = (self.ground_truth * (1-self.alpha) + analysis_ensemble * self.alpha) * self.norm_factors["analysis_states"]
        self.solver.xa = analysis_combined
        self.solver.analysis_states[-1] =  analysis_combined
        self.current_step += 1
        done = self.current_step >= self.n_steps
        
        # Visualize episode if done and visualization is enabled
        if done:
            # update alpha, increase if model performs better, decrease otherwise
            self.curr_reward /= self.current_step
            if self.curr_reward > self.last_reward:
                self.alpha += 1/1000 # alpha = 1 at the end of training
                self.alpha = min(self.alpha, 1)
            else:
                self.alpha -= 1/100
                self.alpha = max(self.alpha, 0)
            self.last_reward = self.curr_reward
            self.curr_reward = 0
            # (experimental) visualization
            if self.visualize_episodes and self.viz_save_path and self.current_ep%10 == 0:
                title_suffix = f" - Episode {self.current_ep}"
                save_path = os.path.join(self.viz_save_path, f"episode_{self.current_ep:04d}.png")
                self.solver.visualize(save_path=save_path, title_suffix=title_suffix)
        
        # Get next observation
        next_obs = self._get_observation()
        
        return next_obs, reward, done, done, {'rmse': rmse, 'step': self.current_step}
    
    def reset(self, seed=None, initial_condition=None):
        """Reset environment."""
        super().reset(seed=seed)
        self.current_ep += 1
        self.current_step = 0
        # Reset solver
        if self.fixed_initial_condition:
            self.solver.reset(None)
        else:
            if initial_condition is None:
                initial_condition = np.random.randn(self.N)*5
            self.solver.reset(initial_condition)
        self.solver.step() # Perform initial solver step to populate data
        return self._get_observation(), {}

class ZeroInitEvalEnv(EAKF_RL_Env):
    """Evaluation environment that always starts with zero initial conditions."""
    
    def reset(self, seed=None):
        """Reset environment with zero initial conditions."""
        # Always use zero initial conditions for evaluation
        self.fixed_initial_condition = False
        return super().reset(seed=seed, initial_condition=np.zeros(self.N)+5.0)
    
    def step(self, action):
        """Override step to use eval-specific visualization."""
        next_obs, reward, done, truncated, info = super().step(action)
        
        # Visualize eval episode if done and visualization is enabled
        if done and self.visualize_episodes and self.viz_save_path:
            title_suffix = f" - Eval Episode {self.current_ep}"
            save_path = os.path.join(self.viz_save_path, f"eval_episode_{self.current_ep:04d}.png")
            self.solver.visualize(save_path=save_path, title_suffix=title_suffix)
        
        return next_obs, reward, done, truncated, info

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
            n_steps=config["n_steps"],
            device=config["device"],
            fixed_initial_condition=config["fixed_initial_conditions"],
            visualize_episodes=config.get("visualize_training", False),
            viz_save_path=config.get("viz_save_path")
        )
        return Monitor(env)
    
    return DummyVecEnv([make_env for _ in range(n_envs)])

def create_eval_env(config):
    """Create evaluation environment with zero initial conditions."""
    eval_env = ZeroInitEvalEnv(
        solver=config["solver"],
        n_steps=config["eval_n_steps"],
        device=config["device"],
        fixed_initial_condition=True,
        visualize_episodes=config.get("visualize_eval", False),
        viz_save_path=config.get("viz_save_path")
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
    if config.get("viz_save_path"):
        os.makedirs(config["viz_save_path"], exist_ok=True)
    
    # Save normalization factors to JSON
    if config.get("norm_factors_save_path"):
        os.makedirs(os.path.dirname(config["norm_factors_save_path"]), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        norm_factors_serializable = {}
        for key, value in config["solver"].normed_factors.items():
            if isinstance(value, np.ndarray):
                norm_factors_serializable[key] = value.tolist()
            else:
                norm_factors_serializable[key] = value
        
        with open(config["norm_factors_save_path"], 'w') as f:
            json.dump(norm_factors_serializable, f, indent=2)
        
        print(f"Normalization factors saved to: {config['norm_factors_save_path']}")
    
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
            n_steps=config["n_steps"],
            device=config["device"],
            fixed_initial_condition=config["fixed_initial_conditions"],
            visualize_episodes=config.get("visualize_training", False),
            viz_save_path=config.get("viz_save_path")
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