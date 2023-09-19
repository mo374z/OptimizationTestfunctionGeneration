# OptimizationTestfunctionGeneration

Goal of the project is to train a Machine Learning model on samples of distinct Ground Truth Functions to approximate the Ground Truth Function. The quality of the outcome is evaluated with focus on optimization algorithms.

![Methodical Approach](https://github.com/mo374z/OptimizationTestfunctionGeneration/blob/main/method.png)

## Installation
1. clone the repo
1. install environment via 
    ```bash
    conda env create --file environment.yml
    ```

## Usage

- main/sampling_test.ipynb: a comprehensive study of our sampling method (see chapter 3.2 Sampling)

- **main**: code for the simulations of the **bbobtorch functions** f1, f3 and f24 (see chapter 4 Testfunction Simulation) and evaluation in (see chapter 5 Evaluation)

- For the training of the **Neural Networks** via **MSE** and **Taylor-Loss** there are seperate notebooks.

- utils can be found in the **utils** folder, consisting of further **utils**, **groundtruth sampling** and **optimizers**.


________

