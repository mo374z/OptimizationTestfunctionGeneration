import matplotlib.pyplot as plt
import pandas as pd
import torch
import bbobtorch
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from utils.optimizer import perform_optimization


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


def plot_simulated_meshgrid(X, Y, mesh_results, model: str, colorbar=True):
    plt.pcolormesh(X, Y, mesh_results, cmap='inferno', shading='nearest')
    if colorbar: plt.colorbar(label='Function Value')
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

# def euclidean_distance(point1, point2):
#     return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# def nearest_point(x1, x2, point_set):
#     min_distance = float('inf')
#     nearest_point = None
    
#     for point in point_set:
#         distance = euclidean_distance([x1, x2], point)	
#         if distance < min_distance:
#             min_distance = distance
#             nearest_point = point
    
#     return nearest_point

# def nearest_point_with_target(x1, x2, point_set, target):
#     min_distance = float('inf')
#     nearest_point = None
#     target_value = None
    
#     for point, target_val in zip(point_set, target):
#         distance = euclidean_distance([x1, x2], point)	
#         if distance < min_distance:
#             min_distance = distance
#             nearest_point = point
#             target_value = target_val
    
#     return nearest_point, target_value


# def test_function(X, X_train, X_train_grads, model):
#     ''' X 2D array of shape (2, n) '''
#     X_grads = np.zeros_like(X)
#     for i in range(len(X)):
#         X_grads[i, 0], X_grads[i, 1] = nearest_point_with_target(X[i,0], X[i,1], X_train, X_train_grads)[1]
    
#     X_in = np.concatenate((X, X_grads), axis=1)
#     return model.predict(X_in)


def test_function(X, X_train, X_train_grads, model, method='nearest_neighbor', gradient_estimator=None):
    if method=='nearest_neighbor':
        # Build a KD-tree on X_train for efficient nearest neighbor searches
        tree = KDTree(X_train)

        # Find the nearest neighbors and their corresponding gradients
        _, indices = tree.query(X, k=1)
        estimated_gradients = X_train_grads[indices].reshape(-1, 2)
    elif method=='estimator':
        estimated_gradients = gradient_estimator.predict(X)
    else:
        raise ValueError('Method must be either nearest_neighbor or estimator')

    # Combine original points with nearest gradients
    X_in = np.concatenate((X, estimated_gradients), axis=1)
    
    # Make predictions using the model
    predictions = model.predict(X_in)
    
    return predictions


def calculate_eval_metrics(functions:list, optims:list, n_trials, n_dim=2, max_iters_optim=100):
    df_nr_iter = pd.DataFrame(columns=[f[1] for f in functions])
    df_optim_loc = pd.DataFrame(columns=[f[1] for f in functions])
    df_optim_val = pd.DataFrame(columns=[f[1] for f in functions])
    df_mean_optim_vals = pd.DataFrame(columns=[f[1] for f in functions])

    for optimization_type in optims:
        row_iters = []
        row_optim_loc = []
        row_optim_val = []
        row = []
        for function in functions:
            trials_iters = []
            trials_optim_loc = []
            trials_optim_val = []
            trials = []
            for i in range(n_trials):
                res = []
                res = perform_optimization(optimization_type, function[0], n_dim=2, num_iterations=100,)
                trials_iters.append(len(res[1]))
                trials_optim_loc.append(res[0][-1] if type(res[0][-1]) == np.ndarray else res[0][-1].numpy())
                trials_optim_val.append(res[1][-1])
                trials.append(res[1])
            row_iters.append(f"{np.round(np.mean(trials_iters),2)}±{np.round(np.std(trials_iters),2)}")
            row_optim_loc.append(f"{np.round(np.mean(trials_optim_loc, axis=0),2)}±{np.round(np.std(trials_optim_loc, axis=0),2)}")
            row_optim_val.append(f"{np.round(np.mean(trials_optim_val),2)}±{np.round(np.std(trials_optim_val),2)}")
            max_length = max(len(arr) for arr in trials)
            padded_array_list = []
            for arr in trials:
                last_element = arr[-1]
                padding_length = max_length - len(arr)
                padded_arr = np.pad(arr, (0, padding_length), mode='constant', constant_values=last_element)
                padded_array_list.append(padded_arr)
            row.append(padded_array_list)
        df_nr_iter.loc[len(df_nr_iter)] = row_iters
        df_optim_loc.loc[len(df_optim_loc)] = row_optim_loc
        df_optim_val.loc[len(df_optim_val)] = row_optim_val
        df_mean_optim_vals.loc[len(df_mean_optim_vals)] = row

    df_nr_iter.index = df_optim_loc.index = df_optim_val.index = df_mean_optim_vals.index = optims

    # problem: different lengths of the arrays
    # correct by padding with last element
    df_mean_optim_vals_ = pd.DataFrame(columns=[f[1] for f in functions])
    for row in df_mean_optim_vals.iterrows():
        row_ = []
        for i in range (len(row[1])):
            max_length = 100
            padded_array_list = []
            for arr in row[1][i]:
                last_element = arr[-1]
                padding_length = max_length - len(arr)
                if padding_length > 0:
                    padded_arr = np.pad(arr, (0, padding_length), mode='constant', constant_values=last_element)
                    padded_array_list.append(padded_arr.tolist())
                else:
                    padded_array_list.append(arr.tolist())
            row_.append(padded_array_list)
        df_mean_optim_vals_.loc[len(df_mean_optim_vals_)] = row_
    df_mean_optim_vals_.index = optims

    df_r = df_mean_optim_vals_.copy()
    for row in df_r.iterrows():
        for col in df_r.columns:
            scores = []
            for i in range(len(df_r.loc[row[0], col])):
                score = pearsonr(df_mean_optim_vals_.loc[row[0], col][i], df_mean_optim_vals_.loc[row[0],'Groundtruth'][i])
                scores.append(score[0] if not np.isnan(score[0]) else 0) # use zero as corr if one of the arrays is constant
            df_r.loc[row[0], col] = f"{np.round(np.mean(scores),2)}±{np.round(np.std(scores),2)}"

    df_mse = df_mean_optim_vals_.copy()
    for row in df_mse.iterrows():
        for col in df_mse.columns:
            scores = []
            for i in range(len(df_mse.loc[row[0], col])):
                scores.append(mean_squared_error(df_mean_optim_vals_.loc[row[0], col][i], df_mean_optim_vals_.loc[row[0],'Groundtruth'][i]))
            df_mse.loc[row[0], col] = f"{np.round(np.mean(scores),2)}±{np.round(np.std(scores),2)}"
    
    return df_nr_iter, df_optim_loc, df_optim_val, df_r, df_mse