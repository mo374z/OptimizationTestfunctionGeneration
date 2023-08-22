import matplotlib.pyplot as plt
import torch
import bbobtorch
import numpy as np
from sklearn.neighbors import KDTree


def create_problem(f_number, n_dim, seed):
    if f_number == 1:
        problem = bbobtorch.create_f01(n_dim, seed=seed)
    elif f_number == 3:
        problem = bbobtorch.create_f03(n_dim, seed=seed)
    elif f_number == 24:
        problem = bbobtorch.create_f24(n_dim, seed=seed)
    else:
        raise ValueError('We only use BBOB functions be 1, 3 and 24')

    return problem


def plot_sampled_data(samples, results, f_number: str):	
    plt.scatter(samples[:, 0], samples[:, 1], c=results, cmap='inferno', s=1)

    # Add color bar for reference
    colorbar = plt.colorbar()
    colorbar.set_label('Function Value', rotation=270, labelpad=15)

    # Add labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Custom Samples with Function Values ({f_number})')
    return plt.gca()


def plot_simulated_meshgrid(X, Y, mesh_results, model: str):
    plt.pcolormesh(X, Y, mesh_results, cmap='inferno', shading='nearest')
    plt.colorbar(label='Function Value')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Simulated Function from {model}')

    return plt.gca()

def plot_ground_truth(n_dim, problem, f_name, xlim=(-5, 5), step=0.01):
    ranges = [torch.arange(xlim[0], xlim[1] + step, step=step) for _ in range(n_dim)]
    meshgrid = torch.meshgrid(*ranges)
    points = torch.stack(meshgrid, dim=-1).view(-1, n_dim)

    gt_results = problem(points)

    grid_size = int((xlim[1] - xlim[0]) / step) + 1
    x = points[:, 0].reshape(grid_size, grid_size)
    y = points[:, 1].reshape(grid_size, grid_size)
    z = gt_results.reshape(grid_size, grid_size)

    # Create a pcolormesh plot
    plt.pcolormesh(x, y, z, cmap='inferno', shading='nearest')
    plt.colorbar()
    plt.plot(problem.x_opt[0], problem.x_opt[1], 'rx', markersize=10, label='x_opt')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Ground Truth of {f_name} Function')
    return plt.gca()


def plot_collage(samples, results, problem, problem_name, model_name, X, Y, mesh_results):
    # Plot the sampled points
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plot_sampled_data(samples, results, problem_name)

    # Plot the predicted function values using a contour plot
    plt.subplot(1, 3, 2)
    plot_simulated_meshgrid(X, Y, mesh_results, model_name)

    # Plot the ground truth using a contour plot
    plt.subplot(1, 3, 3)
    plot_ground_truth(2, problem, f_name=problem_name)

    plt.tight_layout()
    return plt.gca()

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def nearest_point(x1, x2, point_set):
    min_distance = float('inf')
    nearest_point = None
    
    for point in point_set:
        distance = euclidean_distance([x1, x2], point)	
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
    
    return nearest_point

def nearest_point_with_target(x1, x2, point_set, target):
    min_distance = float('inf')
    nearest_point = None
    target_value = None
    
    for point, target_val in zip(point_set, target):
        distance = euclidean_distance([x1, x2], point)	
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
            target_value = target_val
    
    return nearest_point, target_value


# def test_function(X, X_train, X_train_grads, model):
#     ''' X 2D array of shape (2, n) '''
#     X_grads = np.zeros_like(X)
#     for i in range(len(X)):
#         X_grads[i, 0], X_grads[i, 1] = nearest_point_with_target(X[i,0], X[i,1], X_train, X_train_grads)[1]
    
#     X_in = np.concatenate((X, X_grads), axis=1)
#     return model.predict(X_in)


def test_function(X, X_train, X_train_grads, model):
    # Build a KD-tree on X_train for efficient nearest neighbor searches
    tree = KDTree(X_train)
    
    # Find the nearest neighbors and their corresponding gradients
    _, indices = tree.query(X, k=1)
    nearest_gradients = X_train_grads[indices]
    
    # Combine original points with nearest gradients
    X_in = np.concatenate((X, nearest_gradients), axis=1)
    
    # Make predictions using the model
    predictions = model.predict(X_in)
    
    return predictions