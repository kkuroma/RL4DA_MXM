'''
A code framework for Lorenz 96 system (https://en.wikipedia.org/wiki/Lorenz_96_model)
'''

import numpy as np


class L96:
    def __init__(self, dt, Nx):
        # Intialize simulation Setting: time step dt, number of grids Nx, etc.
        self.Nx = Nx
        self.dt = dt
    
    def forward_ens(self, ens, u0_ens, Nt):
        # Forward ensemble members: ensemble size ens, initial conditions u0_ens, number of time steps Nt, etc.
        Nx = self.Nx
        dt = self.dt
        u0_ens = u0_ens.A
        U_ens = np.zeros((ens, Nx))
        for e in range(ens):
            u0 = u0_ens[e, :]
            for n in range(Nt):
                # un = 
                # u0 = 

            U_ens[e, :] = un
        return U_ens

