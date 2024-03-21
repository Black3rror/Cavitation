# cavitation

**A study on the Grundfos Cavitation dataset.**

This project is the codebase for the study on the Grundfos Cavitation dataset. It essentially has three parts:
- Data visualization: Visualizing the data and its statistics
- SVM model: Training and testing SVM models
- Neural Network model: Visualizing the performance of an existing neural network model

For the project to work, the dataset should be placed in the *data/raw* directory. Since the Grundfos Cavitation dataset is proprietary, it is not included in this repository. Further, the deep learning models (called neural network models in this project) are created, trained, tested, and saved in another project that will be published in a later stage and linked here. This project only visualizes the performance of the saved neural network model.

## How to use

1. Clone the repository
2. Setup the project
    - If using *Makefile*, run `make setup_project`
    - In case of any issues, execute the commands in the *setup_project* target of the *Makefile* manually
3. Run the project
    - For visualizing the data and its statistics, run `make visualize_data` or `python -m cavitation.data_visualization`
        - You can change the configs in *cavitation/configs/data_visualization_config.yaml*
    - For running the SVM experiments, run `run_svm_experiment` or `python -m cavitation.svm_experiments`
        - You can change the configs in *cavitation/configs/svm_model_config.yaml*
        - You can change the list of experiments in *cavitation/svm_experiments.py*
    - For visualizing the performance of the neural network model, run `visualize_nn_model` or `python -m cavitation.nn_model_visualization`
        - The neural network model should already be saved and the path to the saved model should be provided in the config file
        - You can change the configs in *cavitation/configs/nn_model_visualization_config.yaml*

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make setup_project` or `make requirements`
|
├── README.md            <- The top-level README for developers using this project.
|
├── .gitignore           <- The gitignore file
|
├── .pre-commit-config.yaml <- Configuration file for pre-commit hooks
│
├── pyproject.toml       <- Project configuration file
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
|
├── requirements_test.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── cavitation           <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
|   |
│   ├── configs          <- Configuration files for the project
│   │   ├── data_visualization_config.yaml
│   │   ├── nn_model_visualization_config.yaml
│   │   └── svm_model_config.yaml
│   │
│   ├── data
│   │   └── get_data.py  <- Load the data
|   |
│   ├── logger
│   │   └── easy_logger.py <- An easy-to-use logger
│   │
│   ├── data_visualization.py <- Visualizing the data and its statistics
|   |
│   ├── nn_model_visualization.py <- Visualizing the performance of an existing neural network model
|   |
│   ├── svm_experiments.py <- Script for running the SVM experiments
|   |
│   └── svm_model.py     <- SVM model implementations, training and prediction
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [DL_project_template](https://github.com/Black3rror/DL_project_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for
starting a Deep Learning Project.
