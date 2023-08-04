import matplotlib.pyplot as plt
import torch
import bbobtorch


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


def plot_collage(samples, results, problem, problem_name, X, Y, mesh_results):
    # Plot the sampled points
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plot_sampled_data(samples, results, problem_name)

    # Plot the predicted function values using a contour plot
    plt.subplot(1, 3, 2)
    plot_simulated_meshgrid(X, Y, mesh_results, "NN Model")

    # Plot the ground truth using a contour plot
    plt.subplot(1, 3, 3)
    plot_ground_truth(2, problem, f_name=problem_name)

    plt.tight_layout()
    return plt.gca()