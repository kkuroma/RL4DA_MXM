'''
EnKF to Lorenz 96 system 

Observations are partial and noisy.
Assuming independent observation errors, observations are serially assimilated.
'''

import numpy as np
from construct_GC import construct_GC
from enkf import eakf
from os.path import dirname, join as pjoin

np.random.seed(0)

# --------------------- load data --------------------------
data_dir = '../Data/'
icsfname = pjoin(data_dir, 'l96_ics.npy')
obsfname = pjoin(data_dir, 'l96_obs.npy')
truthfname = pjoin(data_dir, 'l96_truth.npy')

# get truth
ztruth = np.load(truthfname)
print('truth size:', ztruth.shape)

# get initial condition
zics_total = np.load(icsfname)
print('ics size:', zics_total.shape)

# get observation
zobs_total = np.load(obsfname)
print('obs size:', zobs_total.shape)

# ------------------ model parameters -----------------
model_size = 40  # Nx
L96 = L96(Nx=model_size)

# ------------------- observation parameters ------------------
obs_density = 10 # observations are partial: uniformlly distributed for every 'obs_density' model grids 
obs_error_var = 0.04 # observation error variance
model_grids = np.arange(0, model_size)
obs_grids = model_grids[model_grids % obs_density == 0] # 0, 10, 20, 30
nobsgrid = len(obs_grids)

R = np.mat(obs_error_var * np.eye(nobsgrid, nobsgrid)) # Observation identity matrix scaled by obs_error_var

# make observation operator H
Hk = np.mat(np.zeros((nobsgrid, model_size))) # Maps observation into the model?
for iobs in range(0, nobsgrid):
    x1 = obs_grids[iobs] 
    Hk[iobs, x1] = 1.0

time_steps = 10000 # total number of model integration steps
obs_freq_timestep = 20 # observations are infrequent in time: available every 'obs_freq_timestep' time steps
model_times = np.arange(0, time_steps)
obs_times = model_times[model_times % obs_freq_timestep == 0]
nobstime = len(obs_times)

# --------------------- DA parameters -----------------------
# analysis period
iobsbeg = 5
iobsend = int(time_steps/obs_freq_timestep)

# eakf parameters
ensemble_size = 40
ens_mem_beg = 1
ens_mem_end = ens_mem_beg + ensemble_size
inflation_values = [1.05]
ninf = len(inflation_values)
localization_values = [16]
nloc = len(localization_values)
localize = 1

# ---------------------- assimilation -----------------------
prior_mse = np.zeros((nobstime,ninf,nloc))
analy_mse = np.zeros((nobstime,ninf,nloc))
spread_mse = np.zeros((nobstime,ninf,nloc))
prior_err = np.zeros((ninf,nloc))
analy_err = np.zeros((ninf,nloc))

# tuning inflation and localization by grid search
for iinf in range(ninf): # only 1 pass
    inflation_value = inflation_values[iinf]
    print('inflation:',inflation_value)
    
    for iloc in range(nloc): # only 1 pass
        localization_value = localization_values[iloc] # localization matrix cutoff distance
        print('localization:',localization_value)

        CMat = np.mat(construct_GC(localization_value, model_size, obs_grids)) # Matrix used to "localize" Kalman gain matrix (restrict covariance, see yt video)
        zens = np.mat(zics_total[ens_mem_beg: ens_mem_end, :])  # ensemble are drawn from ics set - this is our initial ensemble
        zeakf_prior = np.zeros((model_size, nobstime)) # these four are for analysis purposes I think
        zeakf_analy = np.empty((model_size, nobstime))
        prior_spread = np.empty((model_size, nobstime))
        analy_spread = np.empty((model_size, nobstime))

        for iassim in tqdm(range(0, nobstime), desc="assimilation step"):
            # print(iassim)

            # EnKF step
            obsstep = iassim * obs_freq_timestep + 1
            zens = np.mat(zens)
            zeakf_prior[:, iassim] = np.mean(zens, axis=0)  # prior ensemble mean - this is our forecast (computed by taking the mean of the ensembles)
            zobs = np.mat(zobs_total[iassim, :]) # get observation

            # inflation (Relaxaiton To Prior Perturbations)
            ensmean = np.mean(zens, axis=0) # forecast yet again
            ensp = zens - ensmean
            zens = ensmean + ensp * inflation_value # Increase the "spread" of the ensembles by a factor of inflation_value

            prior_spread[:, iassim] = np.std(zens, axis=0, ddof=1)

            # serial update
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
            zens = eakf(ensemble_size, nobsgrid, zens, Hk, obs_error_var, localize, CMat, zobs)

            # save analysis
            zeakf_analy[:, iassim] = np.mean(zens, axis=0)

            # save analysis spread
            analy_spread[:, iassim] = np.std(zens, axis=0, ddof=1)

            # ensemble model advance, but no advance model from the last obs
            if iassim < nobstime - 1:
                zens = L96.forward_ens(ensemble_size, zens, obs_freq_timestep)

        zt = ztruth.T
        prior_mse[:, iinf, iloc] = np.mean((zt - zeakf_prior) ** 2, axis=0)
        analy_mse[:, iinf, iloc] = np.mean((zt - zeakf_analy) ** 2, axis=0)
        spread_mse[:, iinf, iloc] = np.mean(analy_spread ** 2, axis=0)
        prior_err[iinf, iloc] = np.mean(prior_mse[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err[iinf, iloc] = np.mean(analy_mse[iobsbeg - 1: iobsend, iinf, iloc])

anasave = {
'mean_analy': zeakf_analy,
'mean_prior': zeakf_prior,
'spread_analy': analy_spread,
'spread_pior': prior_spread,
'mse_prior': prior_mse,
'mse_analy': analy_mse,
}
np.savez('../Data/L96_analy.npz', **anasave)

prior_err = np.nan_to_num(prior_err, nan=999)
analy_err = np.nan_to_num(analy_err, nan=999)

# # uncomment these in tuning inflation and localization
# minerr = np.min(prior_err)
# inds = np.where(prior_err == minerr)
# print('min prior error = {0:.6f}, inflation = {1:.3f}, localizaiton = {2:d}'.format(minerr, inflation_values[0][0], localization_values[inds[1][0]]))
# minerr = np.min(analy_err)
# inds = np.where(analy_err == minerr)
# ind = inds[0][0]
# print('min analy error = {0:.6f}, inflation = {1:.3f}, localizaiton = {2:d}'.format(minerr, inflation_values[0][0], localization_values[inds[1][0]]))

# uncomment these if there are only one inflation and one localization
print('prior time mean mse = {0:.6f}, inflation = {1:.3f}, localizaiton = {2:d}'.format(prior_err[0,0], inflation_values[0], localization_values[0]))
print('analy time mean mse = {0:.6f}, inflation = {1:.3f}, localizaiton = {2:d}'.format(analy_err[0,0], inflation_values[0], localization_values[0]))
print('prior mse = {0:.6f}, inflation = {1:.3f}, localizaiton = {2:d}'.format(np.mean(((zt - zeakf_prior)[:, iobsbeg:]) ** 2), inflation_values[0], localization_values[0]))
print('analy mse = {0:.6f}, inflation = {1:.3f}, localizaiton = {2:d}'.format(np.mean(((zt - zeakf_analy)[:, iobsbeg:]) ** 2), inflation_values[0], localization_values[0]))
