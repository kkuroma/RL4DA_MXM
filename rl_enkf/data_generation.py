import numpy as np
import pandas as pd
from l96 import L96
from enkf import eakf
from tqdm import tqdm

def generate_l96(N, F, timesteps, dt):
    '''Generate l96 data using runge-kutta approx.
    Initial conditions will be initialized to [F, ..., F] + N(0, 0.01)
    '''
    system = L96(N, F)
    u0 = np.ones(N) * F + np.random.multivariate_normal(np.zeros(N), np.eye(N) * 0.01)
    data = [u0]
    t = 0

    for i in tqdm(range(timesteps)):
        u = data[-1]
        u = runge_kutta_4(system.dx, u, t, dt)
        data.append(u)
        t += dt

    return np.array(data)

def runge_kutta_4(func, x0, t, dt):
    '''Apply runge-kutta to a function'''
    k1 = func(x0, t)
    k2 = func(x0 + (dt / 2.0) * k1, t + (dt / 2.0))
    k3 = func(x0 + (dt / 2.0) * k2, t + (dt / 2.0))
    k4 = func(x0 + dt * k3, t + dt)

    return x0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

if __name__ == '__main__':
    N = 40
    F = 8
    timesteps = 3600 * 20
    dt = 0.05
    data = generate_l96(N, F, timesteps, dt)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f'./data/lorenz96_N{N}_F{F}.csv')
