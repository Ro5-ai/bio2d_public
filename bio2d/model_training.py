from bio2d.evaluation import scorer
from bio2d.baseline_models import BaselineModelInitializer

import pandas as pd
import numpy as np


def model_prediction(model, task_type: str, X: np.array):
    """Make predictions using the specified model on input data X.

    Args:
        model: A trained machine learning model with a `predict` or `predict_proba` method. Usually an sklearn model, but can also be anything as long as it has the corresponding prediction method defined.
        task_type (str): The type of prediction task. Should be either 'binary' or 'regression'.
        X: The input data to make predictions on.

    Returns:
        An array of predicted values, either as probabilities (for binary classification) or as
        continuous values (for regression).
    """

    if task_type == "binary":
        return model.predict_proba(X)[:, 1]
    if task_type == "regression":
        return model.predict(X)
    


def train_and_evaluate_model(
    task_type: str,
    model_type: str,
    train_data: tuple,
    test_data: tuple = None,
    hyperparameters=None,
):
    """Train a machine learning model using the specified model type and hyperparameters.

    Args:
        task_type (str): The type of task, either 'regression' or 'classification'.
        dataset (str): The name or identifier of the dataset used.
        model_type (str): The model type, e.g. 'catboost', 'lightgbm', 'random_forest', 'support_vector_machine'.
        train_data (tuple): A tuple (features, values) of NumPy arrays for training.
        test_data (tuple, optional): A tuple (features, values) of NumPy arrays for testing. Defaults to None.
        hyperparameters (dict, optional): Model-specific hyperparameters. Defaults to None.

    Raises:
        ValueError: If the model type cannot be initialized for the given task type.
    """

    train_values = train_data['values']
    train_features = train_data['features']

    train_iqr = None
    if task_type == 'regression':
        Q1 = np.percentile(train_values, 25)
        Q3 = np.percentile(train_values, 75)
        train_iqr = Q3 - Q1

    model_init = BaselineModelInitializer()
    try:
        model = getattr(model_init, f"init_{model_type}")(task_type)
    except ValueError as e:
        print(f"Failed to initialize {model_type} model with task type {task_type}: {e}")

    if hyperparameters is not None and model_type == 'catboost':
        model.set_params(
            iterations=float(hyperparameters['iterations']),
            depth=float(hyperparameters['depth']),
            learning_rate=float(hyperparameters['learning_rate']),
            bagging_temperature=float(hyperparameters['bagging_temperature']),
            l2_leaf_reg=float(hyperparameters['l2_leaf_reg']),
        )

    if model_type == 'mpnn':
        model.fit(train_data['smiles'], train_values, features=train_features)
    else:
        model.fit(train_features, train_values)

    if test_data is not None:
        if model_type == 'mpnn':
            y_pred = model.predict(test_data['smiles'], features = test_data['features'])['target']
        else:
            y_pred = model_prediction(model, task_type, test_data['features'])
        scores = scorer(y_pred, test_data['values'], task_type, iqr=train_iqr)
        print(scores)
