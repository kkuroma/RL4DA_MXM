from rl_env import RLEnv
from l96 import L96
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

class Train:
    def __init__(self, N=40, F=5, Nens=20):
        self.N = N
        self.F = F
        self.Nens = Nens

        self.l96_system = L96(self.N, self.F)
        derivative_func = lambda x0, t: self.l96_system.dx(x0, t) # pass self

        observation_dimension = self.N

        identity = np.identity(self.N, dtype=np.float32)
        initial_condition = lambda : np.ones(self.N, dtype=np.float32) + np.random.multivariate_normal(np.zeros(self.N), 0.1 * identity)

        indiv_action_bounds = 0.5 * np.ones(self.N, dtype=np.float32)
        action_bounds = np.concat([indiv_action_bounds[:] for _ in range(self.Nens)])
        observation_bounds = np.concat([indiv_action_bounds[:] for _ in range(self.Nens + 1)])
        action_space = gym.spaces.Box(low=-action_bounds, high=action_bounds, dtype=np.float32)
        observation_space = gym.spaces.Box(low=-observation_bounds, high=observation_bounds, dtype=np.float32)

        initial_ensemble_noise = (np.zeros(self.N), identity)
        termination_rule = lambda t, ens: t > 100

        self.rl_environment = RLEnv(derivative_func, state_dimension=self.N, observation_dimension=self.N, Nens=self.Nens,
            action_space=action_space, observation_space=observation_space, H=identity, noise=identity, initial_condition=initial_condition,
            initial_ensemble_noise=initial_ensemble_noise, termination_rule=termination_rule)

def main():
    device = torch.device("cpu")

    training_handler = Train()
    env = training_handler.rl_environment
    assert isinstance(env.action_space, gym.spaces.box.Box), (f"proved class type {env.action_space}")
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("lorenz96")


if __name__ == '__main__':
    main()
