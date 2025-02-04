'''
A deterministic flavor of EnKF: 
An Ensemble Adjustment Kalman Filter for Data Assimilation(EAKF)-Anderson,2001 or 
Ensemble Data Assimilation without Perturbed Observations (EnSRF)-Whitaker, 2002
(EAKF and EnSRF are equivalent.)
'''

import numpy as np

def eakf(ensemble_size, nobsgrid, zens, Hk, obs_error_var, localize, CMat, zobs):
    '''
        ensemble_size:  # of ensemble elts
        nobsgrid:       dim of observation vector
        zens:           ensemble matrix (each row is an ensemble elt)
        Hk:             observation matrix, maps observation vector to model
        obs_error_var:  variance of observation (inherent)
        localize:       flag for whether to do localization (prevent far-away data points from being correlated)
        CMat:           Localization matrix (gets Hadamard'd with Kalman gain matrix)
        zobs:           Our observation at this time step
    '''
    rn = 1.0 / (ensemble_size - 1) # normalization constant
    
    for iobs in range(0, nobsgrid): # for each observation coef:
        xmean = np.mean(zens, axis=0) 
        xprime = zens - xmean # normalized ensemble
        hxens = (Hk[iobs, :] * zens.T).T  # gets the `iobs`-th value from each ensemble member (`iobs`-th column), 40*1
        hxmean = np.mean(hxens, axis=0)
        hxprime = hxens - hxmean # normalize the `iobs`-th values
        hpbht = (hxprime.T * hxprime * rn)[0, 0] # rn * ||hxprime||^2
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var))) # Can't parse this well but used in calculating the increment?
        pbht = (xprime.T * hxprime) * rn
    
        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        mean_inc = (kfgain * (zobs[0, iobs] - hxmean)).T # kfgain * observation error
        prime_inc = - (gainfact * kfgain * hxprime.T).T # 

        zens = zens + mean_inc + prime_inc
    return zens
