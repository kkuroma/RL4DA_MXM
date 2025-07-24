import numpy as np

def mae(x,y):
    return np.mean(np.abs(x-y))

def mse(x,y):
    return np.mean(np.abs(x-y)**2)

def rmse(x,y):
    return np.sqrt(np.mean(np.abs(x-y)**2))

def generate_close_initial_conditions(base_point, num_points, perturbation=1):
    base_point = np.array(base_point)
    perturbations = np.random.randn(num_points, 3)*perturbation
    initial_conditions = base_point + perturbations
    return initial_conditions