import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def random_search_optimization(function, n_dim, num_iterations=100, search_range=(-5.0, 5.0)):
    best_inputs = []
    best_outputs = []

    best_input = None
    best_output = float('inf')

    for _ in range(num_iterations):
        # Generate a random tensor input within the search range
        random_input = torch.rand(n_dim) * (search_range[1] - search_range[0]) + search_range[0]

        # Compute the output of the function for the random input
        output = function(random_input)

        # Check if the current output is better than the best found so far
        if output < best_output:
            best_output = output
            best_input = random_input

        # Save the best input and output at this iteration
        best_inputs.append(best_input)
        best_outputs.append(best_output.item())

    return best_inputs, best_outputs


def gradient_descent_optimization(function, n_dim, num_iterations=100, learning_rate=0.01, search_range=(-5.0, 5.0), epsilon=1e-4):
    best_inputs = []
    best_outputs = []

    # Initialize random input tensor within the search range
    best_input = torch.rand(n_dim) * (search_range[1] - search_range[0]) + search_range[0]
    best_input.requires_grad = False  # Set requires_grad to False, no autograd

    best_output = function(best_input)

    for _ in range(num_iterations):
        # Compute gradients numerically using finite differences
        gradients = torch.zeros_like(best_input)
        for i in range(n_dim):
            perturbed_input = best_input.clone()
            perturbed_input[i] += epsilon
            perturbed_output = function(perturbed_input)

            # check whether best_output and perturbed_output are a tensor
            # if not, convert them to tensor
            if not isinstance(best_output, torch.Tensor):
                best_output = torch.Tensor(best_output)
            if not isinstance(perturbed_output, torch.Tensor):
                perturbed_output = torch.Tensor(perturbed_output)
            gradients[i] = (perturbed_output - best_output) / epsilon

        # Update the input using the computed gradients with gradient descent
        with torch.no_grad():
            best_input -= learning_rate * gradients

        # Compute the output after the update
        output = function(best_input)

        if not isinstance(output, torch.Tensor):
            output = torch.Tensor(output)

        # Update the best output and input if needed
        if output < best_output:
            best_output = output
            best_inputs.append(best_input.detach().clone().cpu().numpy())  # Store the best input
            best_outputs.append(best_output.item())

    return best_inputs, best_outputs


def evolutionary_optimization(function, n_dim, num_iterations=100, search_range=(-5.0, 5.0)):
    best_inputs = []
    best_outputs = []

    def callback_func(xk, convergence):
        best_inputs.append(xk)

    def wrap_function(x):
        return function(torch.Tensor(x)).item()

    result = differential_evolution(wrap_function, 
        bounds=[(-5,5), (-5,5)], 
        maxiter=num_iterations, 
        callback=callback_func, popsize=10, tol=1e-9)

    # calculate best outputs
    for i in range(len(best_inputs)):
        best_outputs.append(wrap_function(best_inputs[i]))

    return best_inputs, best_outputs   


def perform_optimization(type, function, n_dim, num_iterations):
    if type == "Random":
        return random_search_optimization(function, n_dim, num_iterations)
    elif type == "Gradient":
        return gradient_descent_optimization(function, n_dim, num_iterations)
    elif type == "Evolutionary":
        return evolutionary_optimization(function, n_dim, num_iterations)
    
def plot_optimization(functions:list, optimization_type, n_dim=2, n_times=20, i_evaluations=100):

    for elem in functions:

        results = []

        # Perform random search 'n_times' times and store the results
        for _ in range(n_times):
            best_inputs, best_outputs = perform_optimization(optimization_type, elem[0], n_dim, num_iterations=i_evaluations)
            results.append(np.array(best_outputs))

        # append the elements of results to the same length as the longest
        # Find the maximum length
        max_length = max(len(arr) for arr in results)

        # Initialize a list to store the padded arrays
        padded_array_list = []

        # Loop through the original list and pad arrays
        for arr in results:
            last_element = arr[-1]
            padding_length = max_length - len(arr)
            padded_arr = np.pad(arr, (0, padding_length), mode='constant', constant_values=last_element)
            padded_array_list.append(padded_arr)

        results = padded_array_list

        # Calculate the mean and interquartile range (IQR) for each evaluation
        mean_values = np.mean(results, axis=0)
        q1 = np.percentile(results, 75, axis=0) 
        q3 = np.percentile(results, 25, axis=0)

        x = [i for i in range(1, len(mean_values)+1)]

        plt.plot(x, mean_values, label=f'{elem[1]}')
        plt.fill_between(x, q1, q3,  alpha=0.2)

    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.xlim(0, len(mean_values))
    plt.title(f'Evaluation using {optimization_type} optimization')
    plt.grid(True)
    
    return plt.gca()