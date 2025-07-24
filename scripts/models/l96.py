import numpy as np
from scipy.integrate import solve_ivp

class L96:
    """
    Modular Lorenz 96 model implementation.
    Provides derivative() and step() methods for use with EAKF solver.
    """
    
    def __init__(self, params=None, dt=0.01, use_solve_ivp=False):
        """
        Initialize Lorenz 96 model.
        
        Args:
            params: Dictionary with parameters (N, F) or None for defaults
            dt: Time step for integration
            use_solve_ivp: Use scipy's solve_ivp for integration (default: False, uses RK4)
        """
        if params is None:
            params = self.get_default_params()
        
        self.N = params['N']
        self.F = params['F']
        self.dt = dt
        self.use_solve_ivp = use_solve_ivp
        self.initialized = False
        self.states = []
        self.times = []

    def initialize(self, initial_state):
        """
        Initialize model with given initial state.
        
        Args:
            initial_state: N-dimensional initial state vector
        """
        if len(initial_state) != self.N:
            raise ValueError(f"Initial state must have {self.N} dimensions")
        
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
        Compute Lorenz 96 derivatives.
        
        Args:
            t: Time (not used in autonomous system)
            state: N-dimensional state vector
            
        Returns:
            numpy.ndarray: Derivative vector
        """
        X = np.array(state)
        dXdt = np.zeros(self.N)
        
        # Lorenz 96 equations with periodic boundary conditions
        for i in range(self.N):
            # Use modulo for periodic boundary conditions
            dXdt[i] = (X[(i+1) % self.N] - X[(i-2) % self.N]) * X[(i-1) % self.N] - X[i] + self.F
        
        return dXdt

    def _rk4_step(self, state, dt):
        """
        Perform one RK4 integration step.
        
        Args:
            state: Current state
            dt: Time step
            
        Returns:
            numpy.ndarray: New state after RK4 step
        """
        k1 = self.derivatives(0, state)
        k2 = self.derivatives(0, state + 0.5 * dt * k1)
        k3 = self.derivatives(0, state + 0.5 * dt * k2)
        k4 = self.derivatives(0, state + dt * k3)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

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
            # Use RK4 method (more stable than Euler for L96)
            new_state = self._rk4_step(current_state, self.dt)
        
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

    def get_energy(self):
        """Compute total energy of current state."""
        if not self.initialized:
            raise ValueError("Model must be initialized")
        return 0.5 * np.sum(self.states[-1]**2)

    @staticmethod
    def get_default_params():
        """Return default Lorenz 96 parameters."""
        return {
            'N': 20,
            'F': 9.0
        }

    @staticmethod
    def get_random_initial(N=40, amplitude=1.0):
        """
        Generate random initial condition.
        
        Args:
            N: Number of variables
            amplitude: Amplitude of random perturbations
            
        Returns:
            numpy.ndarray: Random initial state
        """
        return amplitude * np.random.randn(N)

    @staticmethod
    def get_perturbed_initial(N=40, base_value=1.0, perturbation=0.1):
        """
        Generate initial condition with small perturbations around base value.
        
        Args:
            N: Number of variables
            base_value: Base value for all variables
            perturbation: Amplitude of random perturbations
            
        Returns:
            numpy.ndarray: Perturbed initial state
        """
        return base_value * np.ones(N) + perturbation * np.random.randn(N)