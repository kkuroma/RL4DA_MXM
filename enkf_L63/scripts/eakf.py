from tqdm import tqdm
import numpy as np

class eakf_solver:
    def __init__(self, params, initial_conditions, num_ensembles, model_class):
        self.model_class = model_class
        self.num_ensembles = num_ensembles
        sigma, rho, beta = params
        self.params = params
        model = model_class(sigma, rho, beta, 100)
        model.initialize(initial_conditions)
        # pick the last point of a system as a true reference
        self.true_initial, _ = model.step()
        self.ensemble_initials = []
        for _ in range(num_ensembles):
            # Induce randomness to create ensemble members
            self.ensemble_initials.append(self.true_initial + np.random.randn(3))
    
    def run_eakf(self, H, R, dtda, num_assimilations, verbose=1):
        # H, R = observation map and covariance
        # dtda = time between analyses
        sigma, rho, beta = self.params
        # Generate states from true initial, this will be used later
        model = self.model_class(sigma, rho, beta, dtda)
        model.initialize(self.true_initial)
        xt, _ = model.step()
        
        xa = np.array(self.ensemble_initials).T  # Ensemble states
        
        true_states = []
        uncleaned_states = []
        uncleaned_derivatives = []
        cleaned_states = []
        background_states = []
        
        if verbose:
            pbar = tqdm(range(num_assimilations))
        else: 
            pbar = range(num_assimilations)
        for k in pbar:
            # Store true state
            true_states.append(xt)
            
            # Store uncleaned state
            uncleaned_states.append(xa.copy())
            
            # Compute and store uncleaned derivatives
            derivs = np.array([model.derivatives(0, xa[:, l]) for l in range(self.num_ensembles)]).T
            uncleaned_derivatives.append(derivs)
            
            # Generate observations with noise
            observation = H @ xt + np.linalg.cholesky(R) @ np.random.randn(3)
            
            # Generate perturbed observations for the ensemble
            Y = np.tile(observation[:, np.newaxis], (1, self.num_ensembles))
            
            # Propagate each ensemble member forward in time
            xb = np.zeros_like(xa)
            for l in range(self.num_ensembles):
                model.initialize(xa[:, l])
                xb[:, l], _ = model.step()
            
            # Store background states
            background_states.append(xb.copy())
            
            # Compute mean states
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
            K = PHt @ np.linalg.inv(HpfHt + R)
            
            # Update ensemble members with Kalman Gain
            xa = xb + K @ (Y - H @ xb)
            
            # Store cleaned/filtered states
            cleaned_states.append(xa.copy())
            
            # Compute next true state
            model.initialize(xt)
            xt, _ = model.step()
        
        return {
            "true_states": np.array(true_states),
            "uncleaned_states": np.array(uncleaned_states),
            "uncleaned_derivatives": np.array(uncleaned_derivatives),
            "cleaned_states": np.array(cleaned_states),
            "background_states": np.array(background_states)
        }