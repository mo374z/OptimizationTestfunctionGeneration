import torch
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


def gradient_descent_optimization(function, n_dim, num_iterations=100, learning_rate=0.01, search_range=(-5.0, 5.0)):
    best_inputs = []
    best_outputs = []

    # Initialize random input tensor within the search range
    best_input = torch.rand(n_dim) * (search_range[1] - search_range[0]) + search_range[0]
    best_input.requires_grad = True  # Set requires_grad to True for autograd

    best_output = function(best_input)

    for _ in range(num_iterations):
        # Perform a forward pass to compute the output
        output = function(best_input)

        # Clear gradients from the previous iteration
        best_input.grad = None

        # Compute gradients using automatic differentiation
        output.backward()

        # Update the parameters using the computed gradients with gradient descent
        with torch.no_grad():
            best_input -= learning_rate * best_input.grad

        # Update the best output and input if needed
        if output < best_output:
            best_output = output
            best_inputs.append(best_input.detach().clone().cpu().numpy())  # Store the best input (detach to avoid gradients)
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