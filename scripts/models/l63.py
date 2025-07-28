import numpy as np
from scipy.integrate import solve_ivp

class L63:
    """
    Modular Lorenz 63 model implementation.
    Provides derivative() and step() methods for use with EAKF solver.
    """
    
    def __init__(self, params=None, dt=0.01, use_solve_ivp=False):
        """
        Initialize Lorenz 63 model.
        
        Args:
            params: Dictionary with parameters (sigma, rho, beta) or None for defaults
            dt: Time step for integration
            use_solve_ivp: Use scipy's solve_ivp for integration (default: False, uses Euler)
        """
        if params is None:
            params = self.get_default_params()
        
        self.sigma = params['sigma']
        self.rho = params['rho']
        self.beta = params['beta']
        self.dt = dt
        self.use_solve_ivp = use_solve_ivp
        self.initialized = False
        self.states = []
        self.times = []
        self.N = 3

    def initialize(self, initial_state):
        """
        Initialize model with given initial state.
        
        Args:
            initial_state: 3D initial state vector [x, y, z]
        """
        self.reset()
        self.states.append(np.array(initial_state, dtype=float))
        self.times.append(0.0)
        self.initialized = True

    def reset(self):
        """Reset model state history."""
        self.initialized = False
        self.states = []
        self.times = []

    def derivatives(self, t, state):
        """
        Compute Lorenz 63 derivatives.
        
        Args:
            t: Time (not used in autonomous system)
            state: 3D state vector [x, y, z]
            
        Returns:
            numpy.ndarray: Derivative vector [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        # print(x,y,z,self.rho,self.sigma,self.beta)
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])

    def step(self):
        """
        Advance model by one time step.
        
        Returns:
            tuple: (new_state, new_time)
        """
        if not self.initialized:
            raise ValueError("Model must be initialized before stepping")
        
        current_state = self.states[-1]
        current_time = self.times[-1]
        
        if self.use_solve_ivp:
            # Use scipy's adaptive solver
            sol = solve_ivp(
                self.derivatives, 
                [current_time, current_time + self.dt], 
                current_state, 
                method='RK45',
                dense_output=False
            )
            new_state = sol.y[:, -1]
        else:
            # Use simple Euler method
            derivatives = self.derivatives(current_time, current_state)
            new_state = current_state + derivatives * self.dt
        
        new_time = current_time + self.dt
        
        # Store new state and time
        self.states.append(new_state)
        self.times.append(new_time)
        
        return new_state.copy(), new_time

    def get_current_state(self):
        """Get current state without advancing."""
        if not self.initialized:
            raise ValueError("Model must be initialized")
        return self.states[-1].copy()

    def get_current_time(self):
        """Get current time."""
        if not self.initialized:
            raise ValueError("Model must be initialized")
        return self.times[-1]

    def get_history(self):
        """Get complete state and time history."""
        return np.array(self.states), np.array(self.times)

    @staticmethod
    def get_default_params():
        """Return default Lorenz 63 parameters."""
        return {
            'sigma': 10.0,
            'rho': 28.0, 
            'beta': 8.0/3.0
        }

    @staticmethod
    def get_chaotic_initial():
        """Return a typical initial condition in the chaotic regime."""
        return np.array([1.0, 1.0, 1.0])