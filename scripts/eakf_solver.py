import numpy as np
from tqdm import tqdm

class EAKFSolver:
    """
    Environment-like EAKF solver that expects models with derivative() and step() methods.
    Computes ground truth and allows for step-by-step assimilation with potential RL intervention.
    Uses the same algorithm as enkf_L63/scripts/eakf.py but with step-by-step functionality.
    """
    
    def __init__(
        self, 
        model_class,
        model_params, 
        initial_conditions, 
        num_ensembles, 
        H, 
        R, 
        dtda, 
        oda, 
        noise_strength=1.0, 
        inflation=1.05,
        use_solver_ivp=False,
        ):
        """
        Initialize EAKF solver.
        
        Args:
            model_class: Class that implements derivative(t, state) and step() methods
            model_params: Parameters for model initialization (dict or legacy params)
            initial_conditions: True initial state
            num_ensembles: Number of ensemble members
            H: Observation operator matrix
            R: Observation error covariance matrix
            dtda: Time step between assimilations
            oda: Time between observations
            noise_strength: Initial ensemble perturbation strength
            inflation: Multiplicative inflation factor
        """
        self.model_class = model_class
        self.model_params = model_params
        self.true_initial = np.array(initial_conditions)
        self.num_ensembles = num_ensembles
        self.H = H
        self.R = R
        self.dtda = dtda
        self.oda = oda
        self.inflation = inflation
        self.use_solver_ivp = use_solver_ivp
        
        # Initialize ensemble with noise
        self.ensemble_initials = [
            self.true_initial + np.random.randn(len(self.true_initial)) * noise_strength 
            for _ in range(num_ensembles)
        ]
        
        self.reset()
    
    def reset(self, initial_conditions=None):
        """
        Reset the solver to initial conditions.
        
        Args:
            initial_conditions: new initial conditions if not None
        """
        if initial_conditions is not None:
            self.true_initial = np.array(initial_conditions)
        # Initialize true model - handle both dict and legacy parameter formats
        
        self.true_model = self.model_class(self.model_params, self.dtda, self.use_solver_ivp)
        self.true_model.initialize(self.true_initial)
        
        # Advance true model to first observation time
        for _ in range(int(self.oda / self.dtda)):
            self.true_state, _ = self.true_model.step()
        
        # Initialize ensemble analysis states
        self.xa = np.array(self.ensemble_initials).T
        
        # Storage for diagnostics
        self.true_states = []
        self.background_states = []
        self.analysis_states = []
        self.derivatives = []
        self.observations = []
        self.previous_analysis = []
        
        self.current_step = 0
        
    def get_state(self):
        """Get current state information for RL agent."""
        return {
            'true_state': self.true_state.copy(),
            'analysis_ensemble': self.xa.copy(),
            'step': self.current_step
        }
    
    def kalman(self, **kwargs):
        """
        Default Kalman update function (overridable by user).
        Implements the algorithm from lines 61-84 in enkf_L63/scripts/eakf.py
        
        Args:
            **kwargs: Dictionary containing xa, xb, Y, H, R, derivs
            
        Returns:
            numpy.ndarray: Updated analysis ensemble
        """
        xb = kwargs['xb']
        H = kwargs['H']
        R = kwargs['R']
        Y = kwargs['Y']
        # Compute mean states
        xb_mean = np.mean(xb, axis=1, keepdims=True)
        Hx_mean = np.mean(H @ xb, axis=1, keepdims=True)
        
        # Compute error covariance matrices
        state_dim = xb.shape[0]
        obs_dim = H.shape[0]
        PHt = np.zeros((state_dim, obs_dim))
        HpfHt = np.zeros((obs_dim, obs_dim))
        
        for l in range(self.num_ensembles):
            PHt += np.outer(xb[:, l] - xb_mean[:, 0], H @ xb[:, l] - Hx_mean[:, 0])
            HpfHt += np.outer(H @ xb[:, l] - Hx_mean[:, 0], H @ xb[:, l] - Hx_mean[:, 0])
        
        PHt /= (self.num_ensembles - 1)
        HpfHt /= (self.num_ensembles - 1)
        
        # Compute Kalman Gain
        K = PHt @ np.linalg.pinv(HpfHt + R)
        
        # Update ensemble members with Kalman Gain
        return xb + K @ (Y - H @ xb)
    
    def step(self, custom_kalman_func=None):
        """
        Perform one complete EAKF step (one observation at a time).
        
        Args:
            custom_kalman_func: Optional custom kalman function with signature 
                                kalman(**kwargs) -> x_a, where kwargs contains:
                                xa, xb, Y, H, R, derivs
            
        Returns:
            dict: Contains all step information
        """
        # Store true state
        self.true_states.append(self.true_state.copy())
        
        # Store previous analysis
        self.previous_analysis.append(self.xa.copy())
        
        # Compute and store background derivatives at analysis points
        derivs = np.zeros_like(self.xa)
        for l in range(self.num_ensembles):
            if isinstance(self.model_params, dict):
                temp_model = self.model_class(self.model_params, self.dtda, self.use_solver_ivp)
            else:
                temp_model = self.model_class(*self.model_params, self.dtda, self.use_solver_ivp)
            derivs[:, l] = temp_model.derivatives(0, self.xa[:, l])
        self.derivatives.append(derivs.copy())
        
        # Generate observation with noise
        observation = self.H @ self.true_state + np.linalg.cholesky(self.R) @ np.random.randn(self.H.shape[0])
        self.observations.append(observation.copy())
        
        # Generate perturbed observations for the ensemble
        Y = observation[:, np.newaxis] + np.linalg.cholesky(self.R) @ np.random.randn(self.H.shape[0], self.num_ensembles)
        
        # Propagate each ensemble member forward in time
        xb = np.zeros_like(self.xa)
        for l in range(self.num_ensembles):
            if isinstance(self.model_params, dict):
                model = self.model_class(self.model_params, self.dtda, self.use_solver_ivp)
            else:
                model = self.model_class(*self.model_params, self.dtda, self.use_solver_ivp)
            model.initialize(self.xa[:, l])
            for _ in range(int(self.oda / self.dtda)):
                xb[:, l], _ = model.step()
        
        # Store background states
        self.background_states.append(xb.copy())
        
        # Apply inflation
        xb_mean = np.mean(xb, axis=1, keepdims=True)
        xb = xb_mean + (xb - xb_mean) * self.inflation
        
        # Prepare dictionary of parameters for Kalman update
        kalman_params = {
            'xa': self.xa.copy(),
            'xb': xb.copy(),
            'Y': Y.copy(),
            'H': self.H,
            'R': self.R,
            'derivs': self.derivatives[-1].copy() if self.derivatives else None
        }
        
        # Apply Kalman update (either custom or default)
        if custom_kalman_func is not None:
            self.xa = custom_kalman_func(**kalman_params)
        else:
            self.xa = self.kalman(**kalman_params)
        
        # Store analysis states
        self.analysis_states.append(self.xa.copy())
        
        # Advance true state to next observation time
        if isinstance(self.model_params, dict):
            self.true_model = self.model_class(self.model_params, self.dtda, self.use_solver_ivp)
        else:
            self.true_model = self.model_class(*self.model_params, self.dtda, self.use_solver_ivp)
        self.true_model.initialize(self.true_state)
        for _ in range(int(self.oda / self.dtda)):
            self.true_state, _ = self.true_model.step()
        
        self.current_step += 1
        
        return {
            'true_state': self.true_states[-1].copy(),
            'background_ensemble': self.background_states[-1].copy(),
            'analysis_ensemble': self.analysis_states[-1].copy(),
            'derivatives': self.derivatives[-1].copy(),
            'observation': self.observations[-1].copy(),
            'step': self.current_step - 1
        }
    
    def run_eakf(self, num_assimilations, verbose=True, custom_kalman_func=None):
        """
        Run full EAKF for specified number of assimilation cycles.
        
        Args:
            num_assimilations: Number of assimilation cycles
            verbose: Show progress bar
            custom_kalman_func: Optional custom kalman function
            
        Returns:
            dict: Complete solution with all diagnostics
        """
        iterator = tqdm(range(num_assimilations)) if verbose else range(num_assimilations)
        
        for _ in iterator:
            self.step(custom_kalman_func)
        
        return {
            "true_states": np.array(self.true_states),
            "background_states": np.array(self.background_states),
            "derivatives": np.array(self.derivatives),
            "analysis_states": np.array(self.analysis_states),
            "previous_analysis": np.array(self.previous_analysis),
            "observations": np.array(self.observations)
        }
    
    def get_diagnostics(self):
        """Get current diagnostics and history."""
        return {
            "true_states": np.array(self.true_states) if self.true_states else None,
            "background_states": np.array(self.background_states) if self.background_states else None,
            "derivatives": np.array(self.derivatives) if self.derivatives else None,
            "analysis_states": np.array(self.analysis_states) if self.analysis_states else None,
            "previous_analysis": np.array(self.previous_analysis) if self.previous_analysis else None,
            "observations": np.array(self.observations) if self.observations else None,
            "current_step": self.current_step
        }