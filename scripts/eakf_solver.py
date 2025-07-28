import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
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
        self.noise_strength = noise_strength
        
        # Pre-allocate arrays for performance
        self.state_dim = len(initial_conditions)
        self.obs_dim = H.shape[0]
        self.steps_per_oda = int(oda / dtda)
        
        # Create reusable model instances for ensemble members
        self._ensemble_models = []
        for _ in range(num_ensembles):
            if isinstance(model_params, dict):
                model = model_class(model_params, dtda, use_solver_ivp)
            else:
                model = model_class(*model_params, dtda, use_solver_ivp)
            self._ensemble_models.append(model)
        
        # Create single model for derivatives computation
        if isinstance(model_params, dict):
            self._derivative_model = model_class(model_params, dtda, use_solver_ivp)
        else:
            self._derivative_model = model_class(*model_params, dtda, use_solver_ivp)
            
        # Pre-allocate arrays
        self._xb_array = np.zeros((self.state_dim, num_ensembles))
        self._derivs_array = np.zeros((self.state_dim, num_ensembles))
        self._noise_array = np.zeros(self.obs_dim)
        self._obs_noise_array = np.zeros((self.obs_dim, num_ensembles))
        
        # Cache Cholesky decomposition for performance
        self._chol_R = np.linalg.cholesky(R)
        
        self.reset()
        
    def set_normed_factor(self):
        print("Generating normalization factor...")
        true_initial = self.true_initial
        results = []
        for _ in range(20):
            self.reset(np.random.randn(self.true_initial.shape[0])*5)
            results.append(self.run_eakf(100, verbose=True))
        self.reset(true_initial)
        self.normed_factors = {
            key: np.max([np.max(np.abs(result[key])) for result in results])
            for key in results[0].keys()
        }
    
    def reset(self, initial_conditions=None):
        """
        Reset the solver to initial conditions.
        
        Args:
            initial_conditions: new initial conditions if not None
        """
        if initial_conditions is not None:
            self.true_initial = np.array(initial_conditions)
        # Initialize true model (reuse existing instance)
        if not hasattr(self, 'true_model'):
            if isinstance(self.model_params, dict):
                self.true_model = self.model_class(self.model_params, self.dtda, self.use_solver_ivp)
            else:
                self.true_model = self.model_class(*self.model_params, self.dtda, self.use_solver_ivp)
        
        self.true_model.initialize(self.true_initial)
        
        # Advance true model to first observation time
        for _ in range(self.steps_per_oda):
            self.true_state, _ = self.true_model.step()
        
        # Initialize ensemble analysis states
        self.ensemble_initials = [
            self.true_initial + np.random.randn(len(self.true_initial)) * self.noise_strength 
            for _ in range(self.num_ensembles)
        ]
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
        
        # Compute and store background derivatives at analysis points (optimized)
        for l in range(self.num_ensembles):
            self._derivs_array[:, l] = self._derivative_model.derivatives(0, self.xa[:, l])
        self.derivatives.append(self._derivs_array.copy())
        
        # Generate observation with noise (optimized with cached Cholesky)
        self._noise_array[:] = np.random.randn(self.obs_dim)
        observation = self.H @ self.true_state + self._chol_R @ self._noise_array
        self.observations.append(observation.copy())
        
        # Generate perturbed observations for the ensemble (optimized with cached Cholesky)
        self._obs_noise_array[:] = np.random.randn(self.obs_dim, self.num_ensembles)
        Y = observation[:, np.newaxis] + self._chol_R @ self._obs_noise_array
        
        # Propagate each ensemble member forward in time (optimized)
        for l in range(self.num_ensembles):
            model = self._ensemble_models[l]
            model.initialize(self.xa[:, l])
            for _ in range(self.steps_per_oda):
                self._xb_array[:, l], _ = model.step()
        
        # Store background states
        self.background_states.append(self._xb_array.copy())
        
        # Apply inflation
        xb_mean = np.mean(self._xb_array, axis=1, keepdims=True)
        xb = xb_mean + (self._xb_array - xb_mean) * self.inflation
        
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
        
        # Advance true state to next observation time (reuse existing model)
        self.true_model.initialize(self.true_state)
        for _ in range(self.steps_per_oda):
            self.true_state, _ = self.true_model.step()
        
        self.current_step += 1
        
        return {
            'true_state': self.true_states[-1].copy(),
            'background_ensemble': self.background_states[-1].copy(),
            'analysis_ensemble': self.analysis_states[-1].copy(),
            'derivatives': self.derivatives[-1].copy(),
            'observation': self.observations[-1].copy(),
            'step': self.current_step - 1,
            "ensemble_observation": Y,
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
    
    def visualize(self, save_path=None, title_suffix="", show_plot=False):
        """
        Visualize RMSE comparison between truth vs forecast and truth vs posterior.
        
        Args:
            save_path: Path to save the plot (if None, plot is not saved)
            title_suffix: Additional text to add to the plot title
            show_plot: Whether to display the plot (default: False)
        """
        if not self.true_states or not self.background_states or not self.analysis_states:
            print("Warning: No data to visualize. Run the solver first.")
            return
        
        # Convert to numpy arrays and compute means
        truth = np.array(self.true_states)
        background = np.array(self.background_states).mean(axis=-1)  # Mean over ensemble
        analysis = np.array(self.analysis_states).mean(axis=-1)      # Mean over ensemble
        
        # Compute RMSE
        rmse_background = np.sqrt(np.mean((truth - background)**2, axis=1))
        rmse_analysis = np.sqrt(np.mean((truth - analysis)**2, axis=1))
        
        # Create plot
        plt.figure(figsize=(10, 5))
        plt.plot(rmse_background, label="Truth vs Forecast", linewidth=2, color='red')
        plt.plot(rmse_analysis, label="Truth vs Posterior", linewidth=2, color='blue')
        
        plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
        plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=12, fontweight='bold')
        plt.title(f"RMSE Comparison: Truth vs Posterior & Forecast{title_suffix}", fontsize=14, fontweight='bold')
        plt.legend(prop={'size': 16})
        plt.grid()
        
        # Save plot if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close to free memory