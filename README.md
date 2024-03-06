# cavitation

A study on the cavitation dataset.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make setup_project` or `make requirements`
├── README.md            <- The top-level README for developers using this project.
│
├── models               <- Trained and serialized models, model predictions, or model summaries
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
├── cavitation  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── get_data.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── train_model.py   <- script for training the model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [DL_project_template](https://github.com/Black3rror/DL_project_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for
starting a Deep Learning Project.
