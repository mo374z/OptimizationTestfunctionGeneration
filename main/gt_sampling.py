import matplotlib.pyplot as plt
import numpy as np
import torch
import bbobtorch

n_dim = 2
n_samples = 1000
f_number = 1
seed = 42

if f_number == 1:
    problem = bbobtorch.create_f01(n_dim, seed=42)
elif f_number == 2:
    problem = bbobtorch.create_f03(n_dim, seed=42)
elif f_number == 3:
    problem = bbobtorch.create_f24(n_dim, seed=42)
else:
    raise ValueError('We only use BBOB functions be 1, 3 and 24')

