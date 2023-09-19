# OptimizationTestfunctionGeneration

Goal of the project is to train a Machine Learning model on samples of distinct Ground Truth Functions to simulate the Ground Truth Function. The quality of the outcome is evaluated with focus on optimization algorithms.
The project summary can be found in the paper attached to this repository.

## Methodical approach
![workflow](https://github.com/mo374z/OptimizationTestfunctionGeneration/assets/87517800/5d80da44-4f8c-46f2-9867-3b33f7b4a5a9)

## Installation
1. clone the repo
1. install environment via 
    ```bash
    conda env create --file environment.yml
    ```
By using the environment for execution an error-free process can be guaranteed.

## Structure

- **main**: contains code for the simulation and evaluation of the **bbobtorch functions** f1, f3 and f24 (see chapter 4 Testfunction Simulation and chapter 5 Evaluation)
    - main/sampling_test.ipynb: a comprehensive study of our sampling method (see chapter 3.2 Sampling)
    - for the training of the **Neural Networks** via **MSE** and **Taylor-Loss** there are seperate notebooks
- **misc**: miscellaneous files which are not used in the final results but created during the project progress
- **models**: pre-trained models for import
- **plots**: visualizations used in the course of the paper
- **utils**: consists of utilities and functions, which are used in several notebooks

________

