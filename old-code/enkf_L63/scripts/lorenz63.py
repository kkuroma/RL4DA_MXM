import numpy as np
from scipy.integrate import solve_ivp

'''
This class implements the Lorenz 63 model
'''

class L63:
    def __init__(self, sigma, rho, beta, dt, use_solve_ivp=False):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.initialized = False
        self.states = []
        self.times = []
        self.use_solve_ivp = use_solve_ivp

    def initialize(self, values):
        self.reset()
        self.states.append(np.array(values))
        self.times.append(0.0)
        self.initialized = True

    def reset(self):
        self.initialized = False
        self.states = []
        self.times = []

    def derivatives(self, t, state):
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]

    def step(self, step_size=1e-2):
        if self.initialized and self.use_solve_ivp:
            sol = solve_ivp(self.derivatives, [0, self.dt], self.states[-1], method='RK45')
            self.states.append(sol.y[:, -1])
            self.times.append(self.times[-1] + self.dt)
            return sol.y[:, -1], self.times[-1] + self.dt
        elif self.initialized:
            derivatives = self.derivatives(0, self.states[-1])
            new_states = self.states[-1]+np.array(derivatives)*self.dt
            self.states.append(new_states)
            self.times.append(self.times[-1] + self.dt)
            return new_states, self.times[-1] + self.dt