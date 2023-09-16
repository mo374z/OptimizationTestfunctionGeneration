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
    '''
        Create a BBOB problem (F01, F03, and  F24)

        f_number is the name of the BBOB problem (1=F01, 3=F03, or 24=F24)
        n_dim is the number of dimensions of the problem (2 is used for the BBOB functions)
        seed is the seed used to create the problem        
    '''

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
    '''
        Create a scatter plot of the sampled points
        
        samples are the sampled points (x1, x2) from the bbob function using random sampling
        results are the function values (y) of the sampled points
        f_number is the name of the BBOB problem (F01, F03, or F24)
    '''

    plt.scatter(samples[:, 0], samples[:, 1], c=results, cmap='inferno', s=1)

    # Add color bar for reference
    colorbar = plt.colorbar()
    colorbar.set_label('Function Value', rotation=270, labelpad=15)

    # Add labels and title
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.title(f'Custom Samples with Function Values ({f_number})')
    return plt.gca()



def plot_simulated_meshgrid(X, Y, mesh_results, model: str, colorbar=True):
    '''
        Create a contour plot of the simulated function values

        X is the meshgrid of the x1 values forming the grid
        Y is the meshgrid of the x2 values forming the grid
        mesh_results is the predicted function values of the model
        model is the name of the model that was used to predict the function values 
        colorbar is a boolean indicating whether a colorbar should be added to the plot
    '''

    # Create a heatmap-like plot of simulated function values on a meshgrid defined by X and Y
    plt.pcolormesh(X, Y, mesh_results, cmap='inferno', shading='nearest')
    if colorbar: plt.colorbar(label='Function Value')
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.title(f'Simulated Function from {model}')

    return plt.gca()

def plot_ground_truth(n_dim, problem, f_name, xlim=(-5, 5), step=0.01):
    '''
        Create a contour plot of the ground truth function

        n_dim is the number of dimensions of the problem (2 is used for the BBOB functions)
        problem is the BBOB problem from the bbobtorch package  
        f_name is the name of the BBOB problem (F01, F03, or F24)
        xlim is the range of the x values
        step is the step size between the x values
    '''

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

    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.title(f'Ground Truth of {f_name} Function')
    return plt.gca()


def plot_collage(samples, results, problem, problem_name, model_name, X, Y, mesh_results):
    '''
        Create a plot colage with sampled points, predicted function values and ground truth

        samples are the sampled points (x) from the bbob function using random sampling
        results are the function values (y) of the sampled points
        problem is are the sampled points from the bbob function
        problem_name is the name of the bbob problem that was used to sample the points (F01, F03, or F24)
        X is the meshgrid of the x1 values forming the grid
        Y is the meshgrid of the x2 values forming the grid
        mesh_results is the predicted function values of the model
    
    '''
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    # Plot the sampled points
    plot_sampled_data(samples, results, problem_name)

    plt.subplot(1, 3, 2)
    # Plot the predicted function values using a contour plot
    plot_simulated_meshgrid(X, Y, mesh_results, model_name)

    # Plot the ground truth using a contour plot
    plt.subplot(1, 3, 3)
    plot_ground_truth(2, problem, f_name=problem_name)

    plt.tight_layout()
    return plt.gca()


def plot_collage_results(problem, problem_name, model_name, model_name_2, X, Y, mesh_results, mesh_results_2):
    '''
        Create a plot colage with sampled points, predicted function values and ground truth

        problem is are the sampled points from the bbob function
        problem_name is the name of the bbob problem that was used to sample the points (F01, F03, or F24)
        model_name is the name of the first model that was used to predict the function values
        model_name_2 is the name of the second model that was used to predict the function values
        X is the meshgrid of the x1 values forming the grid
        Y is the meshgrid of the x2 values forming the grid
        mesh_results is the predicted function values of the first model
        mesh_results_2 is the predicted function values of the second model
    '''	

    # Plot the sampled points
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    # Plot the predicted function of model 1 using a contour plot
    plot_simulated_meshgrid(X, Y, mesh_results, model_name)

    plt.subplot(1, 3, 2)
    # Plot the ground truth using a contour plot
    plot_ground_truth(2, problem, f_name=problem_name)

    plt.subplot(1, 3, 3)
     # Plot the predicted function of model 2 using a contour plot
    plot_simulated_meshgrid(X, Y, mesh_results_2, model_name_2)

    plt.tight_layout()
    return plt.gca()

def test_function(X, X_train, X_train_grads, model, method='nearest_neighbor', gradient_estimator=None):
    ''' 
        Predict the function values for an input X using a model trained on X_train and X_train_grads,
        but estimating the gradients with a specified method.
    
        X is the input with n dims
        X_train is the sample on which the model was trained on
        X_train_grads is the gradients of the sample on which the model was trained on
        model is the model that predicts the function values based on the training input and gradients
        method is the method used to estimate the gradients of the input X (either nearest_neighbor or estimator)
        gradient_estimator is the estimator used to estimate the gradients of the input X (only needed if method is estimator)
    '''    

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

def calculate_eval_metrics(functions:list, optims:list, n_trials, n_dim=2,  seed=42, epsilon=5e-4): #max_iters_optim=100, XXX check if this is needed
    ''' 
        Calculate the number of iterations, the optimal location, the optimal value, 
        the correlation and the MSE between the optimization curves for each function and optimization method

        functions is a list of tuples with the BBOB function and the name of the function
        optims is a list of the optimization methods
        n_trials is the number of trials to perform for each function and optimization method
        n_dim is the number of dimensions of the problem (2 is used for the BBOB functions)
        seed is the seed used to create the problem
        epsilon is the epsilon used for the optimization methods
    '''
    
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
            torch.manual_seed(seed)
            for i in range(n_trials):
                eval_seed = torch.randint(0, 1000, (1,)).item()
                res = []
                res = perform_optimization(optimization_type, function[0], n_dim=n_dim, num_iterations=100, seed=eval_seed, epsilon=epsilon)
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