from tqdm import tqdm
import numpy as np

class eakf_solver:
    def __init__(self, params, initial_conditions, num_ensembles, model_class, noise_strength=1.0):
        self.model_class = model_class
        self.num_ensembles = num_ensembles
        self.params = params
        self.true_initial = initial_conditions
        self.ensemble_initials = [self.true_initial + np.random.randn(3)*noise_strength for _ in range(num_ensembles)]
    
    def run_eakf(self, H, R, dtda, oda, num_assimilations, verbose=1, inflation=1.05):
        # H, R = observation map and covariance
        # dtda = time between analyses
        sigma, rho, beta = self.params
        # Generate states from true initial, this will be used later
        model = self.model_class(sigma, rho, beta, dtda, False)
        model.initialize(self.true_initial)
        for _ in range(int(oda/dtda)):
            xt, _ = model.step()
        
        xa = np.array(self.ensemble_initials).T  # Ensemble states
        
        true_states = []
        background_states = []
        derivatives = []
        analysis_states = []
        previous_analysis = []
        observations = []
        
        if verbose==1:
            pbar = tqdm(range(num_assimilations))
        else: 
            pbar = range(num_assimilations)
        for k in pbar:
            # Store true state
            true_states.append(xt)
            
            # Store background state
            previous_analysis.append(xa.copy())
            
            # Compute and store background derivatives
            derivs = np.array([model.derivatives(0, xa[:, l]) for l in range(self.num_ensembles)]).T
            derivatives.append(derivs)
            
            # Generate observations with noise
            observation = H @ xt + np.linalg.cholesky(R) @ np.random.randn(3)
            observations.append(observation)
            
            # Generate perturbed observations for the ensemble
            Y = observation[:, np.newaxis] + np.linalg.cholesky(R) @ np.random.randn(3, self.num_ensembles)
            
            # Propagate each ensemble member forward in time
            xb = np.zeros_like(xa)
            for l in range(self.num_ensembles):
                model.initialize(xa[:, l])
                for _ in range(int(oda/dtda)):
                    xb[:, l], _ = model.step()
            
            # Store background states
            background_states.append(xb.copy())
            
            # Compute mean states
            xb_mean = np.mean(xb, axis=1, keepdims=True)
            xb = xb_mean + (xb-xb_mean) * inflation
            xb_mean = np.mean(xb, axis=1, keepdims=True)
            Hx_mean = np.mean(H @ xb, axis=1, keepdims=True)
            
            # Compute error covariance matrices
            PHt = np.zeros((3, 3))
            HpfHt = np.zeros((3, 3))
            
            for l in range(self.num_ensembles):
                PHt += np.outer(xb[:, l] - xb_mean[:, 0], H @ xb[:, l] - Hx_mean[:, 0])
                HpfHt += np.outer(H @ xb[:, l] - Hx_mean[:, 0], H @ xb[:, l] - Hx_mean[:, 0])
            
            PHt /= (self.num_ensembles - 1)
            HpfHt /= (self.num_ensembles - 1)
            
            # Compute Kalman Gain
            K = PHt @ np.linalg.pinv(HpfHt + R)
            
            # Update ensemble members with Kalman Gain
            xa = xb + K @ (Y - H @ xb)
            
            # Store analysis/filtered states
            analysis_states.append(xa.copy())
            
            # Compute next true state
            model.initialize(xt)
            for _ in range(int(oda/dtda)):
                xt, _ = model.step()
        
        return {
            "true_states": np.array(true_states),
            "background_states": np.array(background_states),
            "derivatives": np.array(derivatives),
            "analysis_states": np.array(analysis_states),
            "previous_analysis": np.array(previous_analysis),
            "observations": np.array(observations)
        }