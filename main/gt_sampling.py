import matplotlib.pyplot as plt
import numpy as np
import torch
import bbobtorch
import pyswarms as ps

def sample_from_problem_with_pso(problem, 
                                 n_dim, 
                                 seed, 
                                 n_samples=5000, 
                                 particles=20, 
                                 iters=250, 
                                 options={'c1': 0.8, 'c2': 0.3, 'w': 0.95},
                                 lower_bound=-5,
                                 upper_bound=5
                                 ):
    if n_samples != particles * iters:
        raise ValueError('n_samples must be equal to particles * iters')
    np.random.seed(seed)
    bounds=[np.array([lower_bound] * n_dim), np.array([upper_bound] * n_dim)]
    options = options
    optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=n_dim, options=options, bounds=(bounds[0], bounds[1]))
    optimizer.optimize(lambda x: problem(torch.tensor(x, dtype=torch.float32)).numpy(), iters=iters, verbose=False)
    samples = np.array(optimizer.pos_history).reshape((n_samples, 2))

    return samples


def random_sample(n_samples, n_dim, seed, lower_bound=-3, upper_bound=5):
    np.random.seed(seed)
    samples_random = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_samples, n_dim))
    return samples_random

def get_sample(f_number, n_samples, n_dim, seed=42, lower_bound=-5, upper_bound=5, method='random', particles=20, iters=250, options={'c1': 0.8, 'c2': 0.3, 'w': 0.95}):
    if f_number == 1:
        problem = bbobtorch.create_f01(n_dim, seed=42)
    elif f_number == 2:
        problem = bbobtorch.create_f03(n_dim, seed=42)
    elif f_number == 3:
        problem = bbobtorch.create_f24(n_dim, seed=42)
    else:
        raise ValueError('We only use BBOB functions be 1, 3 and 24')

    if method == 'random':
        return random_sample(n_samples, n_dim, seed, lower_bound, upper_bound)
    elif method == 'pso':
        return sample_from_problem_with_pso(problem, n_dim, seed, n_samples=n_samples, particles=particles, iters=iters, options=options, lower_bound=lower_bound, upper_bound=upper_bound)
    else:
        raise ValueError('method must be either random or pso')
    

