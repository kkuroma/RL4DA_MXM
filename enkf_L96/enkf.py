'''
A deterministic flavor of EnKF: 
An Ensemble Adjustment Kalman Filter for Data Assimilation(EAKF)-Anderson,2001 or 
Ensemble Data Assimilation without Perturbed Observations (EnSRF)-Whitaker, 2002
(EAKF and EnSRF are equivalent.)
'''

import numpy as np

def eakf(ensemble_size, nobsgrid, zens, Hk, obs_error_var, localize, CMat, zobs):
    rn = 1.0 / (ensemble_size - 1)
    
    for iobs in range(0, nobsgrid):
        xmean = np.mean(zens, axis=0) 
        xprime = zens - xmean
        hxens = (Hk[iobs, :] * zens.T).T  # 40*1
        hxmean = np.mean(hxens, axis=0)
        hxprime = hxens - hxmean
        hpbht = (hxprime.T * hxprime * rn)[0, 0]
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
        pbht = (xprime.T * hxprime) * rn
    
        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        mean_inc = (kfgain * (zobs[0, iobs] - hxmean)).T
        prime_inc = - (gainfact * kfgain * hxprime.T).T

        zens = zens + mean_inc + prime_inc
    return zens