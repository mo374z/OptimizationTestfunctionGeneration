import matplotlib.pyplot as plt
import numpy as np
import torch
import bbobtorch
import pyswarms as ps

def sample_from_problem_with_pso(problem, 
                                 n_dim, 
                                 seed, 
                                 n_samples=1000, 
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


def random_sample(n_samples, n_dim, seed, lower_bound=-5, upper_bound=5):
    np.random.seed(seed)
    samples_random = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_samples, n_dim))
    return samples_random

def get_gradient(problem, x, h=1e-5):
    gradient = []
    for i in range(len(x[0])):
        # Create a perturbed point for the i-th dimension
        perturbed_x = x.clone()
        perturbed_x[0][i] += h
        
        # Calculate function values at the original and perturbed points
        y_original = problem(x)
        y_perturbed = problem(perturbed_x)
        
        # Calculate finite difference approximation of the partial derivative
        partial_derivative = (y_perturbed - y_original) / h
        gradient.append(partial_derivative.item())
    return gradient

def get_second_derivative(problem, x, h=1e-5):
    second_derivative = []
    for i in range(len(x[0])):
        # Create perturbed points for the i-th dimension
        perturbed_x_left = x.clone()
        perturbed_x_right = x.clone()
        perturbed_x_left[0][i] -= h
        perturbed_x_right[0][i] += h
        
        # Calculate function values at the original and perturbed points
        y_original = problem(x)
        y_left = problem(perturbed_x_left)
        y_right = problem(perturbed_x_right)
        
        # Calculate finite difference approximation of the second derivative
        second_derivative_partial = (y_left - 2 * y_original + y_right) / (h ** 2)
        second_derivative.append(second_derivative_partial.item())
    return second_derivative

def get_sample(problem, n_samples, n_dim, seed=42, lower_bound=-5, upper_bound=5, method='random', particles=20, iters=50, options={'c1': 0.8, 'c2': 0.3, 'w': 0.95}):
    if method == 'random':
        result = random_sample(n_samples, n_dim, seed, lower_bound, upper_bound)
    elif method == 'pso':
        result = sample_from_problem_with_pso(problem, n_dim, seed, n_samples=n_samples, particles=particles, iters=iters, options=options, lower_bound=lower_bound, upper_bound=upper_bound)
    else:
        raise ValueError('method must be either random or pso')
    
    result = torch.tensor(result, dtype=torch.float64).float()
    gradients = np.array([])
    second_derivatives = np.array([])

    for i in range(n_samples):
        tensor_sample = torch.tensor(result[i], dtype=torch.float64).reshape((1, n_dim)).float()
        gradient = get_gradient(problem, tensor_sample)
        second_derivative = get_second_derivative(problem, tensor_sample)
        gradients = np.append(gradients, gradient)
        second_derivatives = np.append(second_derivatives, second_derivative)

    gradients = gradients.reshape((n_samples, n_dim))
    second_derivatives = second_derivatives.reshape((n_samples, n_dim))
    
    return result, problem(result), gradients, second_derivatives
