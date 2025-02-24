import gymnasium as gym
import numpy as np
from enkf import eakf

class RLEnv(gym.Env):
    def __init__(self, derivative_func, dt=0.1, state_dimension=None, observation_dimension=None, Nens=40, action_space=None, observation_space=None, H=None, noise=None, initial_condition=None,
        initial_ensemble_noise=(None, None), termination_rule=None, ground_truth_forward=None, seed=None):
        super(RLEnv, self).__init__()

        self.dx = derivative_func # dx(x, t) is derivative at time t at pos x
        self.dt = dt # each timestep
        self.state_dimension = state_dimension
        self.observation_dimension = observation_dimension
        self.ensemble_size = Nens # number of elements in the ensemble
        self.observation_matrix = H # projects each (ground truth) state vector to observation space
        #self.noise_covariance = noise # covariance matrix for observations. obs = H * truth + w, w ~ N(noise)
        self.noise_variance = noise
        self.noise_covariance = np.eye(observation_dimension) * noise
        self.initial_condition = initial_condition # function that returns a (possibly random?) initial condition
        self.initial_ensemble_noise = initial_ensemble_noise # (mean, covariance) of noise added to initial condition to generate initial ensemble members
        self.termination_rule = termination_rule # termination_rule(t, ensemble) returns whether or not to terminate per step
        self.ground_truth_forward = ( # Used to get the next ground truth state (when using a dataset). Otherwise just uses runge kutta 4.
            ground_truth_forward if ground_truth_forward != None
            else lambda x0, t, dt: runge_kutta_4(self.dx, x0, t, dt)
        )
        self.action_space = action_space
        self.observation_space = observation_space

        self.T = 0
        self.last_obs = None
        self.seed = seed

    def step(self, action):
        H = self.observation_matrix

        # see how the enkf model would have updated the ensemble
        zens = np.stack(self.ensembles, 1)
        zens = eakf(self.ensemble_size, self.observation_dimension, zens, H, self.noise_variance, False, None, self.last_obs)

        # get rmse between enkf-predicted update and action to get score
        zens = np.concatenate(np.unstack(zens))
        rmse = np.sqrt(np.mean((zens - action)**2))

        # Compute forecast, observe next step to get error
        self.ensembles = np.split(action, self.ensemble_size)
        priors = [
            runge_kutta_4(self.dx, ensemble, self.T, self.dt)
            for ensemble in self.ensembles
        ]
        forecast_mean = sum(priors) / self.ensemble_size

        observation = H @ self.ground_truth + self.np_random.multivariate_normal(np.zeros(self.observation_dimension), self.noise_covariance)
        error = H @ forecast_mean - observation # error between observation and prior

        # Construct input vector (ensembles + error between forecast and observation)
        input_vector = np.concat((error, np.concat(self.ensembles)))
        #input_vector = np.concat((error, np.concat(priors)))
        self.T += self.dt

        return input_vector, -rmse, self.termination_rule(self.T, self.ensembles), False, self.__get_info()

    def reset(self, seed=None, **kwargs):
        if not self.seed:
            super().reset(seed=seed)
        else:
            super().reset(seed=self.seed)

        self.T = 0

        ensemble_mean, ensemble_covariance = self.initial_ensemble_noise
        H = self.observation_matrix

        # initial condition
        self.ground_truth = self.initial_condition()
        self.ensembles = [self.ground_truth + self.np_random.multivariate_normal(ensemble_mean, ensemble_covariance) for _ in range(self.ensemble_size)]

        # compute forecast, observe next step to get error
        priors = [
            runge_kutta_4(self.dx, ensemble, self.T, self.dt)
            for ensemble in self.ensembles
        ]
        forecast_mean = sum(priors) / self.ensemble_size
        self.ground_truth = self.ground_truth_forward(self.ground_truth, self.T, self.dt)
        observation = H @ self.ground_truth + self.np_random.multivariate_normal(np.zeros(self.observation_dimension), self.noise_covariance)
        error = H @ forecast_mean - observation

        # construct our RL model input vector
        input_vector = np.concat((error, np.concat(self.ensembles)))
        self.last_obs = observation
        self.T += self.dt

        return input_vector, self.__get_info()

    def __get_info(self):
        return {
            "time": self.T,
            "ground_truth": self.ground_truth,
            "ens_mean": np.mean(self.ensembles)
        }

def runge_kutta_4(func, x0, t, dt):
    '''Apply runge-kutta to a function'''
    k1 = func(x0, t)
    k2 = func(x0 + (dt / 2.0) * k1, t + (dt / 2.0))
    k3 = func(x0 + (dt / 2.0) * k2, t + (dt / 2.0))
    k4 = func(x0 + dt * k3, t + dt)

    return x0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
