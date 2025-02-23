from rl_env import RLEnv
from l96 import L96
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.logger import configure

class Train:
    def __init__(self, N=40, F=5, Nens=20, action_coef=10, observation_coef=10, seed=0):
        self.N = N
        self.F = F
        self.Nens = Nens

        self.l96_system = L96(self.N, self.F)
        derivative_func = lambda x0, t: self.l96_system.dx(x0, t) # pass self

        observation_dimension = self.N

        identity = np.identity(self.N, dtype=np.float32)
        noise = 0.2
        initial_condition = lambda : np.ones(self.N, dtype=np.float32) + np.random.multivariate_normal(np.zeros(self.N), 0.1 * identity)

        indiv_action_bounds = action_coef * np.ones(self.N, dtype=np.float32)
        indiv_obs_bounds = observation_coef * np.ones(self.N, dtype=np.float32)
        action_bounds = np.concat([indiv_action_bounds[:] for _ in range(self.Nens)])
        observation_bounds = np.concat([indiv_obs_bounds[:]] + [indiv_action_bounds[:] for _ in range(self.Nens)])
        action_space = gym.spaces.Box(low=-action_bounds, high=action_bounds, dtype=np.float32)
        observation_space = gym.spaces.Box(low=-observation_bounds, high=observation_bounds, dtype=np.float32)

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

    #training_handler = Train()
    #env = training_handler.rl_environment
    #assert isinstance(env.action_space, gym.spaces.box.Box), (f"proved class type {env.action_space}")
    #model = SAC("MlpPolicy", env, verbose=1)
    num_cpu = 50
    n_epochs = 15

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecNormalize(env)
    env = VecMonitor(env, filename="log.txt")
    logger = configure('logs', ["stdout", "csv"])
    model = PPO("MlpPolicy", env,
        n_epochs=n_epochs,
        verbose=1
    )
    model.set_logger(logger)
    model.learn(
        total_timesteps=10000,
        log_interval=4,
        progress_bar=False
    )
    model.save("lorenz96")


if __name__ == '__main__':
    main()
