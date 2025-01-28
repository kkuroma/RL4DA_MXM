import numpy as np


def construct_GC(cut, l, ylocs):
    """
    Construct the Gaspari and Cohn localization matrix for a 1D field. 
    The localization matrix is multiplied as the Schur product to the Kalman gain matrix.

    more on localization: Distance-Dependent Filtering of Background Error Covariance Estimates in an Ensemble Kalman Filter - Hamill, 2001

    Parameters:
        cut (float): Localization cutoff distance.
        l (array): 1D coordinates of the model grids [x1, x2, ...].
        ylocs (array): 1D coordinates of the observations locations [y1, y2, ...].

    Returns:
        np.ndarray: Localization matrix of shape (len(ylocs), len(l)).
    """
    nobs = len(ylocs)
    V = np.zeros((nobs, l))

    for iobs in range(0, nobs):
        yloc = ylocs[iobs]
        for iCut in range(0, l):
            dist = min(abs(iCut - yloc), abs(iCut - l - yloc), abs(iCut + l - yloc))
            r = dist / (0.5 * cut)

            if dist >= cut:
                V[iobs, iCut] = 0.0
            elif 0.5*cut <= dist < cut:
                V[iobs, iCut] = r**5 / 12.0 - r**4 / 2.0 + r**3 * 5.0 / 8.0 + r**2 * 5.0 / 3.0 - 5.0 * r + 4.0 - 2.0 / (3.0 * r)
            else:
                V[iobs, iCut] = r**5 * (-0.25) + r**4 / 2.0 + r**3 * 5.0/8.0 - r**2 * 5.0/3.0 + 1.0

    return V