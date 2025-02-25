from rl_env import RLEnv
from l96 import L96
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

class Train:
    def __init__(self, N=8, F=5, Nens=8, action_coef=1, observation_coef=1, seed=0):
        self.N = N
        self.F = F
        self.Nens = Nens

        self.l96_system = L96(self.N, self.F)
        derivative_func = lambda x0, t: self.l96_system.dx(x0, t) # pass self

        observation_dimension = self.N

        identity = np.identity(self.N, dtype=np.float32)
        noise = 0.1
        initial_condition = lambda : np.ones(self.N, dtype=np.float32) + np.random.multivariate_normal(np.zeros(self.N), 0.1 * identity)

        '''
        indiv_action_bounds = action_coef * np.ones(self.N, dtype=np.float32)
        indiv_obs_bounds = observation_coef * np.ones(self.N, dtype=np.float32)
        action_bounds = np.concat([indiv_action_bounds[:] for _ in range(self.Nens)])
        observation_bounds = np.concat([indiv_obs_bounds[:]] + [indiv_action_bounds[:] for _ in range(self.Nens)])
        action_space = gym.spaces.Box(low=-action_bounds, high=action_bounds, dtype=np.float32)
        observation_space = gym.spaces.Box(low=-observation_bounds, high=observation_bounds, dtype=np.float32)
        '''

        action_space = gym.spaces.Box(low=-1, high=1, shape=(self.Nens * self.N,), dtype="float32")
        observation_space = gym.spaces.Box(low=-1, high=1, shape=((self.Nens + 1) * self.N,), dtype="float32")

        #initial_ensemble_noise = (np.zeros(self.N), identity)
        initial_ensemble_noise = (np.zeros(self.N), np.eye(observation_dimension) * noise)
        termination_rule = lambda t, ens: t > 100

        self.rl_environment = RLEnv(derivative_func, state_dimension=self.N, observation_dimension=self.N, Nens=self.Nens,
            action_space=action_space, observation_space=observation_space, H=identity, noise=noise, initial_condition=initial_condition,
            initial_ensemble_noise=initial_ensemble_noise, termination_rule=termination_rule, seed=seed)

def main():
    device = torch.device("cpu")

    def make_env(i):
        def wrapper():
            training_handler = Train(seed=i)
            training_handler
            return training_handler.rl_environment
        return wrapper

    n_epochs = 150 # number of episodes
    epoch_length = 500 # length of each episode
    eval_freq = 1500 # training steps before evaluating

    training_env = Train().rl_environment
    eval_env = Train(seed=1).rl_environment
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
        log_path='./logs/', eval_freq=eval_freq, deterministic=True,
        render=False)

    model = PPO("MlpPolicy", training_env,
        n_steps=epoch_length,
        n_epochs=n_epochs,
        batch_size=epoch_length // 20,
        verbose=2
    )
    model.learn(
        total_timesteps=epoch_length * n_epochs,
        log_interval=1,
        progress_bar=False,
        callback=eval_callback
    )
    model.save("lorenz96")


if __name__ == '__main__':
    main()
