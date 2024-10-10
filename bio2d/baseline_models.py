import os
import subprocess

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR


class BaselineModelInitializer:
    """
    A class for initializing baseline models.

    Args:
        class_weight (str): Type of class weighting (default 'balanced').
        random_state (int): Random seed (default 1234).
    """

    def __init__(self, class_weight="balanced", random_state=1234):
        self.class_weight = class_weight
        self.random_state = random_state

    def init_random_forest(self, task_type):
        """
        Initialize Random Forest Classifier or Regressor.

        Args:
            task_type (str): Type of the task, 'binary' or 'regression'.

        Returns:
            obj: Random Forest Classifier or Regressor.
        """
        if task_type == "binary":
            return RandomForestClassifier(class_weight=self.class_weight, random_state=self.random_state, max_features="sqrt")
        elif task_type == "regression":
            return RandomForestRegressor(random_state=self.random_state, max_features=1.0)
        else:
            raise ValueError(f"Invalid task type for random forest: {task_type}")

    def init_support_vector_machine(self, task_type):
        """
        Initialize Support Vector Machine Classifier or Regressor.

        Args:
            task_type (str): Type of the task, 'binary' or 'regression'.

        Returns:
            obj: Support Vector Machine Classifier or Regressor.
        """
        if task_type == "binary":
            return SVC(class_weight=self.class_weight, probability=True)
        elif task_type == "regression":
            return SVR()
        else:
            raise ValueError(f"Invalid task type for support vector machine: {task_type}")

    def init_linear_regression(self, task_type):
        """
        Initialize Linear Regression.

        Args:
            task_type (str): Type of the task, 'binary' or 'regression'.

        Returns:
            obj: Linear Regression.
        """
        if task_type == "binary":
            return LogisticRegression(class_weight=self.class_weight)
        elif task_type == "regression":
            return LinearRegression()
        else:
            raise ValueError(f"Invalid task type for linear regression: {task_type}")

    def init_lightgbm(self, task_type):
        """
        Initialize LightGBM.

        Args:
            task_type (str): Type of the task, 'binary' or 'regression'.

        Returns:
            obj: LightGBM model.
        """
        if task_type == "binary":
            return LGBMClassifier(class_weight=self.class_weight)
        elif task_type == "regression":
            return LGBMRegressor()
        else:
            raise ValueError(f"Invalid task type for LGBM: {task_type}")

    def init_xgboost(self, task_type):
        """
        Initialize XGBoost.

        Args:
            task_type (str): Type of the task, 'binary' or 'regression'.

        Returns:
            obj: XGBoost model.
        """
        if task_type == "binary":
            return xgboost.XGBClassifier(use_label_encoder=False)
        elif task_type == "regression":
            return xgboost.XGBRegressor()
        else:
            raise ValueError(f"Invalid task type for XGBoost: {task_type}")

    def init_mpnn(self, task_type, save_dir=None):
        return ChemPropModel(dataset_type=task_type, save_dir=save_dir)

    def init_catboost(self, task_type):
        """
        Initialize CatBoost Classifier or Regressor.

        Args:
            task_type (str): Type of the task, 'binary' or 'regression'.

        Returns:
            obj: CatBoost Classifier or Regressor.
        """
        params = {
            'random_strength': 2,
            'random_seed': 123,
            'verbose': 0,
        }
        if task_type == "binary":
            params['loss_function'] = 'Logloss'
            return CatBoostClassifier(**params)
        elif task_type == "regression":
            params['loss_function'] = 'MAE'
            return CatBoostRegressor(**params)
        else:
            raise ValueError(f"Invalid task type for CatBoost: {task_type}")

    def init_knn(self, task_type, n_neighbors=1):
        """
        Initialize K-Nearest Neighbors Classifier or Regressor.

        Args:
            task_type (str): Type of the task, 'binary' or 'regression'.
            n_neighbors (int): Number of neighbors to use.

        Returns:
            obj: K-Nearest Neighbors Classifier or Regressor.
        """
        if task_type == "binary":
            return KNeighborsClassifier(n_neighbors=n_neighbors)
        elif task_type == "regression":
            return KNeighborsRegressor(n_neighbors=n_neighbors)
        else:
            raise ValueError(f"Invalid task type for KNN: {task_type}")


class ChemPropModel:
    def __init__(self, dataset_type, save_dir, features_path=None):
        if save_dir is None:
            save_dir = 'mpnn_checkpoints/'

        self.dataset_type = dataset_type
        if dataset_type == 'binary':
            self.dataset_type = 'classification'
        else:
            self.dataset_type = dataset_type

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir

        self.features_path = features_path

        self.temporary_dir = 'mpnn_output'
        if not os.path.exists(self.temporary_dir):
            os.mkdir(self.temporary_dir)

    def fit(self, smiles_list, targets, features=None):
        df = pd.DataFrame(
            {
                'smiles': smiles_list,
                'target': targets,
            }
        )

        data_csv_path = os.path.join(self.temporary_dir, 'data.csv')
        df.to_csv(data_csv_path, index=False)

        arguments = ['chemprop_train', '--data_path', data_csv_path, '--dataset_type', self.dataset_type, '--save_dir', self.save_dir]

        if features is not None:
            features_path = os.path.join(self.temporary_dir, 'train_features.csv')
            pd.DataFrame(features).to_csv(features_path, index=False)
            arguments.extend(['--features_path', features_path])

        subprocess.run(arguments)

        return self.save_dir

    def predict(self, smiles_list, checkpoint_dir=None, features=None):
        df = pd.DataFrame({'smiles': smiles_list})
        test_path = os.path.join(self.temporary_dir, 'data.csv')
        df.to_csv(test_path, index=False)

        preds_path = os.path.join(self.temporary_dir, 'preds.csv')

        if checkpoint_dir is None:
            checkpoint_dir = self.save_dir

        arguments = ['chemprop_predict', '--test_path', test_path, '--preds_path', preds_path, '--checkpoint_dir', checkpoint_dir]
        if features is not None:
            features_path = os.path.join(self.temporary_dir, 'test_features.csv')
            pd.DataFrame(features).to_csv(features_path, index=False)
            arguments.extend(['--features_path', features_path])

        subprocess.run(arguments)
        return pd.read_csv(preds_path)
