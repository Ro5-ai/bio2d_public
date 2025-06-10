# catboost_wrapper.py

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from dataset_config import get_dataset_config

class CatBoostSklearnRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, dataset_name, **kwargs):
        """
        If hyperparameters are not provided via kwargs, we load the defaults based on the dataset.
        """
        self.dataset_name = dataset_name
        config = get_dataset_config(dataset_name)
        defaults = config.get("catboost_params", {})
        # Merge defaults with any explicit kwargs (kwargs take precedence)
        self.params = {**defaults, **kwargs}
        self.model = CatBoostRegressor(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y, verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        self.model = CatBoostRegressor(**self.params)
        return self

