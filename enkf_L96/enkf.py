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
        print('xmean: ', xmean.shape)
        xprime = zens - xmean # normalized ensemble
        print('xprime: ', xprime.shape)
        hxens = (zens.T.dot(Hk[iobs, :])).T  # gets the `iobs`-th value from each ensemble member (`iobs`-th column), 40*1
        print('hxens: ', hxens.shape)
        hxmean = np.mean(hxens, axis=0)
        print('hxmean: ', hxmean.shape)
        hxprime = hxens - hxmean # normalize the `iobs`-th values
        print('hxprime: ', hxprime.shape)
        hpbht = (hxprime.T.dot(hxprime) * rn) # rn * ||hxprime||^2
        print('hpbht: ', hpbht)
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var))) # Can't parse this well but used in calculating the increment?
        print('gainfact: ', gainfact.shape)
        pbht = (xprime.dot( hxprime)) * rn
        print('pbht: ', pbht.shape)
    
        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        print('kfgain: ', kfgain.shape)

        mean_inc = (kfgain * (zobs[0, iobs] - hxmean)).T # kfgain * observation error
        mean_inc = np.stack([mean_inc for _ in range(ensemble_size)]).T
        print('mean_inc: ', mean_inc.shape)
        prime_inc = - (gainfact * kfgain.reshape(nobsgrid, 1) @ hxprime.reshape(1, ensemble_size)) # 
        print('prime_inc: ', prime_inc.shape)

        zens = zens + mean_inc + prime_inc
    return zens

if __name__ == '__main__':
    ensemble_size = 40
    model_size = 20
    nobsgrid = 20
    zens = np.ones((nobsgrid, ensemble_size))
    Hk = np.eye(nobsgrid)
    #Hk = np.asmatrix(np.zeros((nobsgrid, model_size)))
    print('Hk: ', Hk.shape)
    #for iobs in range(0, nobsgrid):
    #    x1 = obs_grids[iobs] 
    #    Hk[iobs, x1] = 1.0
    zobs = np.array([np.ones(nobsgrid)])
    eakf(ensemble_size, nobsgrid, zens, Hk, 1, False, None, zobs)
